"""Delivery fallback: when the browser lost the reply, send it as a Telegram voice note.

Rock's case (17 Oct 2026): Herefordshire reception. The phone drops the WebSocket in the background or the
signal goes; the reply is already synthesised, so the service forwards the same audio as a voice note to the
person's Telegram chat (the bot and chat the Hermes gateway already uses). Nothing for the agent to decide.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import httpx

LOG = logging.getLogger("voice.fallback")


class TelegramVoiceNote:
    def __init__(self, cfg: dict) -> None:
        self.token = str(cfg.get("telegram_bot_token") or "").strip()
        self.chat_id = str(cfg.get("telegram_chat_id") or "").strip()
        self.api = str(cfg.get("telegram_api") or "https://api.telegram.org").rstrip("/")

    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, text: str, clips: list[tuple[str, bytes]], note: str = "") -> str:
        """Send the reply: one OGG/Opus voice note made from the clips (plus the text as caption), or a text
        message when there is no audio. Returns what was sent ("voice", "text") or raises."""
        if not self.configured():
            raise RuntimeError("telegram fallback not configured")
        caption = (note + "\n" if note else "") + text
        if clips:
            try:
                ogg = await _to_ogg_opus(clips)
                async with httpx.AsyncClient(timeout=60) as c:
                    r = await c.post(f"{self.api}/bot{self.token}/sendVoice", data={"chat_id": self.chat_id, "caption": caption[:1024]},
                                     files={"voice": ("reply.ogg", ogg, "audio/ogg")})
                if r.status_code == 200:
                    return "voice"
                LOG.warning("sendVoice failed %s: %s", r.status_code, r.text[:200])
            except Exception as e:  # noqa: BLE001 - fall through to text
                LOG.warning("voice note failed: %s", e)
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self.api}/bot{self.token}/sendMessage", data={"chat_id": self.chat_id, "text": caption[:4000]})
        if r.status_code != 200:
            raise RuntimeError(f"sendMessage failed {r.status_code}: {r.text[:200]}")
        return "text"


async def _to_ogg_opus(clips: list[tuple[str, bytes]], timeout: float = 60.0) -> bytes:
    """Concatenate the sentence clips (any container ffmpeg reads) into one OGG/Opus file, Telegram's voice format."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, (mime, data) in enumerate(clips):
            ext = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav", "audio/ogg": "ogg",
                   "audio/webm": "webm", "audio/aac": "aac", "audio/mp4": "m4a"}.get(mime.split(";")[0].strip().lower(), "bin")
            p = Path(d) / f"c{i:04d}.{ext}"
            p.write_bytes(data)
            paths.append(p)
        lst = Path(d) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in paths))
        out = Path(d) / "reply.ogg"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "32k", "-vbr", "on", str(out),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("ffmpeg timed out")
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')[-300:]}")
        return out.read_bytes()
