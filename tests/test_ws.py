"""The WebSocket loop end to end with fake providers and a fake OpenAI-compatible brain."""
import asyncio
import base64
import json
import os
import struct
import wave
import io
import threading

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
import uvicorn


def _wav(seconds=1.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture(scope="module")
def brain_server():
    fake = FastAPI()
    seen = []

    @fake.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        seen.append(body)
        text = body["messages"][-1]["content"].split("\n")[-1]

        async def gen():
            for piece in [f"You said {text}. ", "**Second** sentence here. ", "Bye."]:
                yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    cfg = uvicorn.Config(fake, host="127.0.0.1", port=18077, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    import time
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    yield seen
    server.should_exit = True


@pytest.fixture()
def app(tmp_path, monkeypatch, brain_server):
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOICE_TOKEN", "secret")
    monkeypatch.setenv("BRAIN_URL", "http://127.0.0.1:18077/v1")
    monkeypatch.setenv("VOICE_PROFILE", "mahadeva")
    monkeypatch.setenv("VOICE_FAIL_PROVIDERS", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("AZURE_SPEECH_KEY", "az")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import importlib
    import voice_agent.app as appmod
    appmod = importlib.reload(appmod)
    from voice_agent.providers import STTResult, TTSResult

    async def fake_transcribe(self, wav, language):
        return STTResult(text="hello mahadeva", seconds=1.0, usd=0.001)

    async def fake_synth(self, text, language):
        return TTSResult(audio=b"ID3fake", mime="audio/mpeg", chars=len(text), usd=0.001)
    for name in ("openai", "azure"):
        p = appmod.PROVIDERS[name]
        monkeypatch.setattr(type(p), "transcribe", fake_transcribe)
        monkeypatch.setattr(type(p), "synthesize", fake_synth)
    monkeypatch.setattr(type(appmod.PROVIDERS["local"]), "configured", lambda self: False)
    return appmod


def test_healthz_and_auth(app):
    c = TestClient(app.app)
    h = c.get("/healthz").json()
    assert h["ok"] and h["stt"] == ["openai", "azure"] and h["brain"]
    assert c.get("/api/status").status_code == 401
    st = c.get("/api/status?token=secret").json()
    assert st["waterfall"]["legs"]["tts"]["order"] == ["openai", "azure", "gemini", "local"]
    assert c.get("/").status_code == 200 and "VoiceAgent" in c.get("/voice.js").text
    assert c.get("/nhs/").status_code == 200 and c.get("/nhs/reach.html").status_code == 200
    assert "DPIA" in c.get("/docs/nhs-ig-note.md").text


def test_ws_turn_with_fallthrough(app, brain_server):
    c = TestClient(app.app)
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect("/ws?token=wrong"):
            pass
    assert e.value.code == 4401
    with c.websocket_connect("/ws?token=secret") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready" and ready["brain"] == "mahadeva"
        ws.send_bytes(_wav(1.2))
        ws.send_text(json.dumps({"type": "utterance_end"}))
        events = []
        while True:
            m = json.loads(ws.receive_text())
            events.append(m)
            if m["type"] == "done":
                break
        kinds = [e["type"] for e in events]
        assert "status" in kinds, kinds                      # openai forced failure -> azure
        tr = next(e for e in events if e["type"] == "transcript")
        assert tr["text"] == "hello mahadeva" and tr["provider"] == "azure"
        sentences = [e for e in events if e["type"] == "sentence"]
        audio = [e for e in events if e["type"] == "audio"]
        assert len(sentences) >= 2 and len(audio) == len(sentences)
        assert all("**" not in s["text"] for s in sentences)
        assert base64.b64decode(audio[0]["b64"]) == b"ID3fake" and audio[0]["provider"] == "azure"
        assert events[-1]["reply"].startswith("You said hello mahadeva")
        # the brain got the session key and the voice prefix on the first turn only
        assert brain_server[-1]["prompt_cache_key"] == ready["session"]
        assert "voice interface" in brain_server[-1]["messages"][-1]["content"]
        ws.send_text(json.dumps({"type": "text", "text": "second turn"}))
        while json.loads(ws.receive_text())["type"] != "done":
            pass
        assert brain_server[-1]["messages"][-1]["content"] == "second turn"
    st = c.get("/api/status?token=secret").json()["waterfall"]
    assert st["providers"]["openai"]["state"] == "cooling"
    assert st["providers"]["azure"]["spent_usd"] > 0


def test_http_tts_and_stt(app):
    c = TestClient(app.app)
    r = c.post("/api/tts?token=secret", json={"text": "hi *there*"})
    assert r.status_code == 200 and r.headers["x-provider"] == "azure" and r.content == b"ID3fake"
    r = c.post("/api/stt", files={"file": ("a.wav", _wav(0.5), "audio/wav")}, data={"token": "secret"})
    assert r.json()["text"] == "hello mahadeva"
