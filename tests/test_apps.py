"""Stage 9: several apps behind one container. A fake prof-snarky /api/query (Clerk-style Bearer check, footnotes,
figures, a Sources footer, the similar-question offer, 401/402), the token-to-app map, per-app origins, per-app
monthly caps (free floor only when spent), the auth hand-off over the socket and the meta event for the page."""
import asyncio
import io
import json
import threading
import time
import wave

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import uvicorn

from test_ws import brain_server  # noqa: F401  (the OpenAI-style fake for the default app)

ANSWER = ("Oh, **amiodarone**[^1]. It works beautifully right up until it destroys the thyroid (Fig. F1).\n\n"
          "![F1: thyroid](http://x/f1.jpg)\n*F1. A thyroid, outraged.*\n\n"
          "- Check TFTs at baseline[^2]\n- Then every six months\n\n**Sources:** Smith 2020; Jones 2021\n")


def _wav(seconds=1.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture(scope="module")
def snarky_server():
    fake = FastAPI()
    seen = []

    @fake.post("/api/query")
    async def query(req: Request):
        body = await req.json()
        auth = req.headers.get("authorization", "")
        seen.append({"body": body, "auth": auth})
        if auth != "Bearer clerk-jwt":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        if body["query"].startswith("broke"):
            return JSONResponse({"error": "Insufficient credits. Each query costs $0.25 (1 credit)."}, status_code=402)
        if body["query"].startswith("again"):
            return {"similar": {"id": "chat-old", "query": "amiodarone and the thyroid", "answer": "Old answer[^1] here.",
                                "citations": [{"title": "Smith", "url": "u"}], "persona": "snarky", "saved": False,
                                "createdAt": "2026-09-01T10:00:00.000Z", "similarity": 0.93}, "creditsUsed": 0}
        return {"answer": ANSWER, "citations": [{"title": "Smith 2020", "url": "u1"}, {"title": "Jones 2021", "url": "u2"}],
                "figures": [{"id": "f1"}], "creditsUsed": 1, "freeSearchUsed": False, "mode": "quick",
                "persona": body.get("persona") or "snarky", "archetype": "", "chatId": "chat-new",
                "threadId": body.get("threadId") or "chat-new", "threadReset": False, "freeFollowupsLeft": None}

    cfg = uvicorn.Config(fake, host="127.0.0.1", port=18078, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    yield seen
    server.should_exit = True
    t.join(10)


@pytest.fixture()
def app(tmp_path, monkeypatch, brain_server, snarky_server):  # noqa: F811
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOICE_TOKEN", "secret")
    monkeypatch.setenv("VOICE_TOKEN_PROF_SNARKY", "snarky-token")
    monkeypatch.setenv("PROF_SNARKY_URL", "http://127.0.0.1:18078/api/query")
    monkeypatch.setenv("PROF_SNARKY_BUDGET_USD", "0.05")
    monkeypatch.setenv("VOICE_ALLOWED_ORIGINS", "https://mahadeva.example")
    monkeypatch.setenv("BRAIN_URL", "http://127.0.0.1:18077/v1")
    monkeypatch.setenv("VOICE_PROFILE", "mahadeva")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("VOICE_FAIL_PROVIDERS", raising=False)
    import importlib
    import voice_agent.app as appmod
    appmod = importlib.reload(appmod)
    from voice_agent.providers import STTResult, TTSResult

    async def fake_transcribe(self, wav, language):
        return STTResult(text="tell me about amiodarone", seconds=1.0, usd=0.01)

    async def fake_synth(self, text, language):
        return TTSResult(audio=b"ID3fake", mime="audio/mpeg", chars=len(text), usd=0.01)

    async def local_synth(self, text, language):
        return TTSResult(audio=b"RIFFlocal", mime="audio/wav", chars=len(text), usd=0.0)

    async def local_transcribe(self, wav, language):
        return STTResult(text="local heard you", seconds=1.0, usd=0.0)
    monkeypatch.setattr(type(appmod.PROVIDERS["openai"]), "transcribe", fake_transcribe)
    monkeypatch.setattr(type(appmod.PROVIDERS["openai"]), "synthesize", fake_synth)
    monkeypatch.setattr(type(appmod.PROVIDERS["local"]), "configured", lambda self: True)
    monkeypatch.setattr(type(appmod.PROVIDERS["local"]), "synthesize", local_synth)
    monkeypatch.setattr(type(appmod.PROVIDERS["local"]), "transcribe", local_transcribe)
    return appmod


def _turn(ws, msg, auth_reply=None):
    ws.send_text(json.dumps(msg))
    events = []
    while True:
        m = json.loads(ws.receive_text())
        if m["type"] == "auth_request":
            if auth_reply is not None:
                ws.send_text(json.dumps({"type": "auth", "token": auth_reply}))
            continue
        events.append(m)
        if m["type"] == "done":
            return events


def test_registry_and_status(app):
    assert set(app.APPS.apps) == {"mahadeva", "prof-snarky"}
    assert app.APPS.resolve("secret").name == "mahadeva" and app.APPS.resolve("snarky-token").name == "prof-snarky"
    assert app.APPS.resolve("nope") is None and app.APPS.resolve(None) is None
    assert app.ALLOWED_ORIGINS == {"https://mahadeva.example", "https://www.prof-snarky.com", "https://prof-snarky.com"}
    c = TestClient(app.app)
    assert c.get("/healthz").json()["apps"] == ["mahadeva", "prof-snarky"]
    st = c.get("/api/status?token=snarky-token").json()
    assert st["app"] == "prof-snarky"
    ps = st["apps"]["prof-snarky"]
    assert ps["brain"] == "prof-snarky" and ps["needs_auth"] and ps["budget_usd"] == 0.05 and not ps["camera"]
    assert "tokens" in ps and "snarky-token" not in json.dumps(st)          # never the token itself
    assert st["apps"]["mahadeva"]["brain"] == "openai" and st["apps"]["mahadeva"]["camera"]


def test_origin_per_app(app):
    ok = app._origin_ok
    snarky, maha = app.APPS.resolve("snarky-token"), app.APPS.resolve("secret")
    assert ok("https://www.prof-snarky.com", "voice.surajmenon.ai", snarky)
    assert not ok("https://www.prof-snarky.com", "voice.surajmenon.ai", maha)       # another app's site, wrong token
    assert ok("https://mahadeva.example", "voice.surajmenon.ai", maha)
    assert not ok("https://mahadeva.example", "voice.surajmenon.ai", snarky)
    assert ok("https://voice.surajmenon.ai", "voice.surajmenon.ai", snarky)         # own host always
    c = TestClient(app.app)
    from starlette.websockets import WebSocketDisconnect
    with c.websocket_connect("/ws?token=secret", headers={"origin": "https://www.prof-snarky.com", "host": "voice.surajmenon.ai"}) as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_text()
    assert e.value.code == 4403
    with c.websocket_connect("/ws?token=snarky-token", headers={"origin": "https://www.prof-snarky.com", "host": "voice.surajmenon.ai"}) as ws:
        ready = json.loads(ws.receive_text())
        assert ready["app"] == "prof-snarky" and ready["needs_auth"] and ready["camera"] is False
        assert ready["brain"] == "prof-snarky" and ready["budget"]["usd"] == 0.05 and ready["fallback"] is None


def test_prof_snarky_turn_spoken_and_meta(app, snarky_server):
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=snarky-token") as ws:
        json.loads(ws.receive_text())
        ws.send_bytes(_wav(1.0))
        events = _turn(ws, {"type": "utterance_end", "auth": "stale-jwt", "persona": "perspicacity"}, auth_reply="clerk-jwt")
        kinds = [e["type"] for e in events]
        assert "transcript" in kinds and "meta" in kinds and "error" not in kinds, kinds
        sentences = " ".join(e["text"] for e in events if e["type"] == "sentence")
        assert "amiodarone" in sentences and "[^1]" not in sentences and "Fig" not in sentences
        assert "Sources:" not in sentences and "thyroid, outraged" not in sentences and "**" not in sentences
        assert "Check TFTs at baseline" in sentences and "2 sources are listed in the chat" in sentences
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["chat_id"] == "chat-new" and len(meta["citations"]) == 2 and meta["credits_used"] == 1
        assert meta["answer"] == ANSWER and meta["similar"] is False and meta["persona"] == "perspicacity"
        assert events[-1]["reply"].startswith("Oh, **amiodarone**")
        # the brain got the FRESH credential (auth_request answered), the persona, no history, the raw words
        last = snarky_server[-1]
        assert last["auth"] == "Bearer clerk-jwt" and last["body"]["persona"] == "perspicacity"
        assert last["body"] == {"query": "tell me about amiodarone", "mode": "quick", "force": False, "persona": "perspicacity"}
        # a stored similar answer is spoken, free, and announced
        events = _turn(ws, {"type": "text", "text": "again please"}, auth_reply="clerk-jwt")
        sentences = " ".join(e["text"] for e in events if e["type"] == "sentence")
        assert "similar on 1 September" in sentences and "Old answer here" in sentences
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["similar"] and meta["chat_id"] == "chat-old" and meta["credits_used"] == 0


def test_prof_snarky_thread_rides_along(app, snarky_server):
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=snarky-token") as ws:
        json.loads(ws.receive_text())
        events = _turn(ws, {"type": "text", "text": "and in children?", "thread_id": "11111111-1111-1111-1111-111111111111"},
                       auth_reply="clerk-jwt")
        assert snarky_server[-1]["body"]["threadId"] == "11111111-1111-1111-1111-111111111111"
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["thread_id"] == "11111111-1111-1111-1111-111111111111" and meta["thread_reset"] is False
        # an empty thread_id ("New question") clears it for the next turn
        _turn(ws, {"type": "text", "text": "a fresh question", "thread_id": ""}, auth_reply="clerk-jwt")
        assert "threadId" not in snarky_server[-1]["body"]
        # a similar replay is a new question: the page gets no thread back (it clears its own)
        events = _turn(ws, {"type": "text", "text": "again please", "thread_id": "11111111-1111-1111-1111-111111111111"},
                       auth_reply="clerk-jwt")
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["similar"] and "thread_id" not in meta


def test_prof_snarky_refusals_are_spoken(app, snarky_server):
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=snarky-token") as ws:
        json.loads(ws.receive_text())
        n = len(snarky_server)
        events = _turn(ws, {"type": "text", "text": "no auth at all"})             # page never sends a credential
        err = next(e for e in events if e["type"] == "error")
        assert err["kind"] == "auth" and "not signed in" in err["text"]
        assert len(snarky_server) == n                                             # nothing was sent to the site
        assert any(e["type"] == "audio" for e in events)                           # and the refusal was spoken
        events = _turn(ws, {"type": "text", "text": "hello", "auth": "wrong-jwt"})
        err = next(e for e in events if e["type"] == "error")
        assert err["kind"] == "auth" and "did not accept your sign-in" in err["text"]
        events = _turn(ws, {"type": "text", "text": "broke now", "auth": "clerk-jwt"})
        err = next(e for e in events if e["type"] == "error")
        assert err["kind"] == "auth" and "no credits left" in err["text"]


def test_default_app_unchanged(app, brain_server):
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=secret") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["app"] == "mahadeva" and ready["brain"] == "mahadeva" and not ready["needs_auth"] and ready["camera"]
        events = _turn(ws, {"type": "text", "text": "hello there"})
        kinds = [e["type"] for e in events]
        assert "auth_request" not in kinds and "meta" not in kinds and "error" not in kinds
        assert events[-1]["reply"].startswith("You said hello there")


def test_app_budget_falls_to_the_free_floor(app, snarky_server):
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=snarky-token") as ws:
        json.loads(ws.receive_text())
        for _ in range(3):                                                         # 3 turns x (stt 0.01 + tts 0.01 x n)
            _turn(ws, {"type": "text", "text": "tell me more", "auth": "clerk-jwt"})
    st = c.get("/api/status?token=snarky-token").json()
    assert st["apps"]["prof-snarky"]["spent_usd"] >= 0.05 and st["apps"]["prof-snarky"]["over_budget"]
    assert st["waterfall"]["apps"]["prof-snarky"]["spent_usd"] >= 0.05
    assert not st["apps"]["mahadeva"]["over_budget"]                               # the other app is untouched
    with c.websocket_connect("/ws?token=snarky-token") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["budget"]["free_only"] and ready["tts"] == ["local"] and ready["stt"] == ["local"]
        ws.send_bytes(_wav(1.0))
        events = _turn(ws, {"type": "utterance_end", "auth": "clerk-jwt"})
        assert next(e for e in events if e["type"] == "transcript")["provider"] == "local"
        assert {e["provider"] for e in events if e["type"] == "audio"} == {"local"}
    with c.websocket_connect("/ws?token=secret") as ws:                            # mahadeva still gets openai
        ready = json.loads(ws.receive_text())
        assert ready["tts"][0] == "openai"
