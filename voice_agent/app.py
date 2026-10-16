"""voice-agent service: WebSocket voice loop, HTTP helpers, the standalone page and the widget."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import AudioError, to_wav16k, wav_duration_seconds
from .brain import Brain, BrainError, SentenceSplitter, speakable
from .config import HERE, load_config
from .providers import build_providers
from .waterfall import Exhausted, Waterfall

logging.basicConfig(level=os.environ.get("VOICE_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("voice.app")

CFG = load_config()
SERVICE = CFG.get("service") or {}
DATA_DIR = Path(SERVICE.get("data_dir") or os.environ.get("VOICE_DATA_DIR", "/data"))
TOKENS = {t for t in (SERVICE.get("tokens") or []) if t}
LANGUAGE = SERVICE.get("language", "en-GB")
MAX_SECONDS = float(SERVICE.get("max_utterance_seconds", 60))
PROVIDERS = build_providers(CFG)
WF = Waterfall(CFG, PROVIDERS, DATA_DIR)
BRAIN = Brain(CFG.get("brain") or {})
WEB = HERE / "web"

app = FastAPI(title="voice-agent", version=__version__, docs_url=None, redoc_url=None)


# -- auth -----------------------------------------------------------------------------------
def _ok_token(token: str | None) -> bool:
    if not TOKENS:
        return True
    return any(secrets.compare_digest(token or "", t) for t in TOKENS)


def _require(token: str | None, authorization: str | None = None) -> None:
    if authorization and authorization.lower().startswith("bearer "):
        token = token or authorization[7:].strip()
    if not _ok_token(token):
        raise HTTPException(401, "bad or missing token")


@app.on_event("startup")
async def _startup() -> None:
    if not TOKENS:
        LOG.warning("no service tokens configured: the page and API are open to anyone who can reach the port")
    if not BRAIN.configured():
        LOG.warning("brain.url is empty: the service will transcribe and speak but cannot answer")
    LOG.info("profile=%s legs=%s", SERVICE.get("profile"), {k: WF.candidates(k) for k in WF.legs})


@app.on_event("shutdown")
async def _shutdown() -> None:
    for p in PROVIDERS.values():
        await p.close()
    await BRAIN.close()


# -- pages ----------------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "version": __version__, "profile": SERVICE.get("profile"),
            "brain": BRAIN.configured(), "stt": WF.candidates("stt"), "tts": WF.candidates("tts")}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/voice.js")
async def widget_js() -> FileResponse:
    return FileResponse(WEB / "voice.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/nhs")
@app.get("/nhs/")
async def nhs_index() -> FileResponse:
    return FileResponse(WEB / "nhs" / "index.html")


app.mount("/nhs", StaticFiles(directory=str(WEB / "nhs")), name="nhs")
app.mount("/docs", StaticFiles(directory=str(HERE / "docs")), name="docs")


# -- JSON API (tests, scripts, other front ends) ---------------------------------------------
@app.get("/api/status")
async def status(token: str | None = Query(None), authorization: str | None = Header(None)) -> dict:
    _require(token, authorization)
    return {"version": __version__, "profile": SERVICE.get("profile"), "language": LANGUAGE,
            "brain": {"url": BRAIN.url, "model": BRAIN.model, "configured": BRAIN.configured()},
            "waterfall": WF.status()}


@app.post("/api/stt")
async def api_stt(file: UploadFile = File(...), token: str | None = Form(None),
                  authorization: str | None = Header(None)) -> dict:
    _require(token, authorization)
    try:
        wav = await to_wav16k(await file.read())
    except AudioError as e:
        raise HTTPException(400, str(e))
    try:
        name, res = await _stt(wav)
    except Exhausted as e:
        raise HTTPException(503, str(e)[:500])
    return {"text": res.text, "provider": name, "seconds": res.seconds, "usd": res.usd}


@app.post("/api/tts")
async def api_tts(request: Request, token: str | None = Query(None), authorization: str | None = Header(None)) -> Response:
    _require(token, authorization)
    body = await request.json()
    text = speakable(str(body.get("text", "")))
    if not text:
        raise HTTPException(400, "text required")
    try:
        name, res = await _tts(text, body.get("provider"))
    except Exhausted as e:
        raise HTTPException(503, str(e)[:500])
    return Response(res.audio, media_type=res.mime, headers={"X-Provider": name, "X-Cost-USD": f"{res.usd:.6f}"})


@app.post("/api/probe")
async def api_probe(token: str | None = Query(None), authorization: str | None = Header(None),
                    providers: str | None = Query(None)) -> dict:
    """Round-trip each provider: its TTS says a test phrase, its STT must hear it back. No bundled clip needed."""
    _require(token, authorization)
    phrase = "Testing one two three. The quick brown fox."
    wanted = providers.split(",") if providers else list(PROVIDERS)
    out: dict[str, Any] = {}
    for name in wanted:
        p = PROVIDERS.get(name)
        if p is None:
            out[name] = {"ok": False, "error": "unknown provider"}
            continue
        if not p.configured():
            out[name] = {"ok": False, "error": "no key"}
            continue
        row: dict[str, Any] = {}
        t0 = time.monotonic()
        try:
            tts = await asyncio.wait_for(p.synthesize(phrase, LANGUAGE), 60)
            row["tts_ms"] = int((time.monotonic() - t0) * 1000)
            row["tts_bytes"] = len(tts.audio)
            wav = await to_wav16k(tts.audio)
            row["audio_seconds"] = round(wav_duration_seconds(wav), 2)
            t1 = time.monotonic()
            stt = await asyncio.wait_for(p.transcribe(wav, LANGUAGE), 90)
            row["stt_ms"] = int((time.monotonic() - t1) * 1000)
            row["heard"] = stt.text
            heard = stt.text.lower()
            row["ok"] = ("two" in heard or "2" in heard) and "fox" in heard
            WF.record(name, usd=tts.usd + stt.usd, stt_seconds=stt.seconds, tts_chars=tts.chars)
            if row["ok"]:
                WF.breaker.reset(name)
        except Exception as e:  # noqa: BLE001
            row["ok"] = False
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        out[name] = row
    return out


@app.post("/api/reset")
async def api_reset(token: str | None = Query(None), authorization: str | None = Header(None),
                    provider: str | None = Query(None)) -> dict:
    _require(token, authorization)
    for name in ([provider] if provider else list(PROVIDERS)):
        WF.breaker.reset(name)
    return {"ok": True}


async def _stt(wav: bytes, on_fallthrough=None):
    name, res = await WF.run("stt", lambda p: p.transcribe(wav, LANGUAGE), on_fallthrough)
    WF.record(name, usd=res.usd, stt_seconds=res.seconds)
    return name, res


async def _tts(text: str, only: str | None = None, on_fallthrough=None):
    if only:
        p = PROVIDERS[only]
        res = await p.synthesize(text, LANGUAGE)
        WF.record(only, usd=res.usd, tts_chars=res.chars)
        return only, res
    name, res = await WF.run("tts", lambda p: p.synthesize(text, LANGUAGE), on_fallthrough)
    WF.record(name, usd=res.usd, tts_chars=res.chars)
    return name, res


# -- the voice loop over WebSocket ----------------------------------------------------------
class Turn:
    """One utterance: STT -> brain stream -> sentence TTS, with audio sent as it is ready."""

    def __init__(self, ws: WebSocket, session: str, history: list[dict]):
        self.ws = ws
        self.session = session
        self.history = history
        self.cancelled = False

    async def send(self, **msg: Any) -> None:
        if not self.cancelled:
            await self.ws.send_text(json.dumps(msg))

    async def fallthrough(self, leg: str):
        async def _cb(name: str, reason: str, nxt: str | None) -> None:
            await self.send(type="status", leg=leg, text=f"{name} failed ({reason[:80]}); "
                            + (f"using {nxt}" if nxt else "no provider left"))
        return _cb

    async def run(self, audio: bytes | None, typed: str | None) -> None:
        t0 = time.monotonic()
        try:
            if typed is not None:
                text, stt_name = typed.strip(), "typed"
            else:
                wav = await to_wav16k(audio or b"")
                secs = wav_duration_seconds(wav)
                if secs > MAX_SECONDS:
                    raise AudioError(f"utterance too long ({secs:.0f} s > {MAX_SECONDS:.0f} s)")
                if secs < 0.25:
                    await self.send(type="transcript", text="", provider="none", note="too short")
                    await self.send(type="done", turn_ms=0)
                    return
                stt_name, res = await _stt(wav, await self.fallthrough("stt"))
                text = res.text
            await self.send(type="transcript", text=text, provider=stt_name, ms=int((time.monotonic() - t0) * 1000))
            if not text:
                await self.send(type="done", turn_ms=int((time.monotonic() - t0) * 1000))
                return
            if not BRAIN.configured():
                raise BrainError("no brain configured")
            splitter = SentenceSplitter(int((CFG.get("brain") or {}).get("min_sentence_chars", 24)))
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            speaker = asyncio.create_task(self._speak(tts_queue))
            reply = ""
            try:
                async for delta in BRAIN.stream(self.session, text, self.history, first_turn=not self.history):
                    if self.cancelled:
                        break
                    reply += delta
                    await self.send(type="token", text=delta)
                    for sentence in splitter.feed(delta):
                        await tts_queue.put(sentence)
                for sentence in splitter.flush():
                    await tts_queue.put(sentence)
            finally:
                await tts_queue.put(None)
                await speaker
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            del self.history[:-20]
            await self.send(type="done", turn_ms=int((time.monotonic() - t0) * 1000), reply=reply)
        except (AudioError, BrainError, Exhausted) as e:
            LOG.warning("turn failed: %s", e)
            await self.send(type="error", text=str(e)[:400])
            await self.send(type="done", turn_ms=int((time.monotonic() - t0) * 1000))

    async def _speak(self, queue: asyncio.Queue[str | None]) -> None:
        seq = 0
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            if self.cancelled:
                continue
            try:
                name, res = await _tts(sentence, None, await self.fallthrough("tts"))
            except Exhausted as e:
                await self.send(type="error", text=f"no voice available: {str(e)[:200]}", leg="tts")
                await self.send(type="sentence", text=sentence, provider="none", seq=seq)
                seq += 1
                continue
            seq += 1
            await self.send(type="sentence", text=sentence, provider=name, seq=seq)
            await self.send(type="audio", seq=seq, mime=res.mime, b64=base64.b64encode(res.audio).decode(), provider=name)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str | None = Query(None), session: str | None = Query(None)) -> None:
    if not _ok_token(token):
        await ws.close(code=4401)
        return
    await ws.accept()
    session = session or f"voice-{uuid.uuid4().hex[:12]}"
    history: list[dict] = []
    chunks: list[bytes] = []
    current: asyncio.Task | None = None
    turn: Turn | None = None
    await ws.send_text(json.dumps({"type": "ready", "session": session, "profile": SERVICE.get("profile"),
                                   "stt": WF.candidates("stt"), "tts": WF.candidates("tts"),
                                   "brain": BRAIN.model if BRAIN.configured() else None}))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                chunks.append(msg["bytes"])
                continue
            try:
                data = json.loads(msg.get("text") or "{}")
            except ValueError:
                continue
            kind = data.get("type")
            if kind == "cancel":
                if turn:
                    turn.cancelled = True
                chunks = []
                await ws.send_text(json.dumps({"type": "cancelled"}))
            elif kind in ("utterance_end", "text"):
                if current and not current.done():
                    await ws.send_text(json.dumps({"type": "error", "text": "still answering the last turn"}))
                    chunks = []
                    continue
                audio = b"".join(chunks) if kind == "utterance_end" else None
                chunks = []
                turn = Turn(ws, session, history)
                current = asyncio.create_task(turn.run(audio, data.get("text") if kind == "text" else None))
            elif kind == "reset":
                history.clear()
                session = f"voice-{uuid.uuid4().hex[:12]}"
                await ws.send_text(json.dumps({"type": "ready", "session": session}))
            elif kind == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        if turn:
            turn.cancelled = True
        if current and not current.done():
            current.cancel()


def main() -> None:
    import uvicorn
    uvicorn.run("voice_agent.app:app", host=SERVICE.get("host", "0.0.0.0"), port=int(SERVICE.get("port", 8090)),
                log_level="info", ws_max_size=32 * 1024 * 1024)


if __name__ == "__main__":
    main()
