"""The brain is any OpenAI-compatible /v1/chat/completions endpoint (stage 4 bridge, DeepSeek, Azure OpenAI...).

The transcript is the state: the brain holds history keyed by ``prompt_cache_key`` (the bridge maps
it to one Hermes session), so the voice providers can change between turns without loss.
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

import httpx

LOG = logging.getLogger("voice.brain")

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$|\n+")
_MD_JUNK = re.compile(r"[*_`#>]+")
_TOOL_LINE = re.compile(r"^\s*>?\s*[⚙✗].*$", re.M)


class BrainError(Exception):
    pass


class Brain:
    def __init__(self, cfg: dict):
        self.url = (cfg.get("url") or "").rstrip("/")
        self.key = cfg.get("api_key") or ""
        self.model = cfg.get("model", "mahadeva")
        self.timeout = float(cfg.get("timeout_seconds", 120))
        self.system_prompt = cfg.get("system_prompt") or ""
        self.first_turn_prefix = cfg.get("first_turn_prefix") or ""
        self.send_history = bool(cfg.get("send_history", False))
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0))

    def configured(self) -> bool:
        return bool(self.url)

    async def stream(self, session: str, user_text: str, history: list[dict] | None = None,
                     first_turn: bool = False) -> AsyncIterator[str]:
        """Yield assistant text deltas (reasoning and tool lines are dropped)."""
        text = user_text
        if first_turn and self.first_turn_prefix:
            text = f"{self.first_turn_prefix.strip()}\n\n{user_text}"
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.send_history and history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})
        body = {"model": self.model, "stream": True, "prompt_cache_key": session, "messages": messages}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        try:
            async with self._client.stream("POST", f"{self.url}/chat/completions", json=body, headers=headers) as r:
                if r.status_code >= 400:
                    detail = (await r.aread())[:300].decode(errors="replace")
                    raise BrainError(f"brain HTTP {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as e:
            raise BrainError(f"brain unreachable: {e}") from e

    async def close(self) -> None:
        await self._client.aclose()


def speakable(text: str) -> str:
    """Strip markdown and the bridge's tool lines so TTS does not read asterisks aloud."""
    text = _TOOL_LINE.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)       # [label](url) -> label
    text = re.sub(r"https?://\S+", "a link", text)
    text = _MD_JUNK.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class SentenceSplitter:
    """Feed streamed deltas, get complete sentences out as soon as they close (TTS starts early)."""

    def __init__(self, min_chars: int = 24):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            m = next((m for m in _SENTENCE_END.finditer(self.buf) if m.end() >= self.min_chars), None)
            if not m:
                break
            head, self.buf = self.buf[:m.end()], self.buf[m.end():]
            s = speakable(head)
            if s:
                out.append(s)
        return out

    def flush(self) -> list[str]:
        s = speakable(self.buf)
        self.buf = ""
        return [s] if s else []
