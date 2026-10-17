"""The Telegram voice-note fallback: lost browser -> reply forwarded; resend -> same; unconfigured -> nothing."""
import asyncio
import pytest
from voice_agent import fallback


class FakeWS:
    def __init__(self, die_after=1):
        self.sent, self.die_after = [], die_after
    async def send_text(self, s):
        if len(self.sent) >= self.die_after:
            raise RuntimeError("Cannot call send once a close message has been sent")
        self.sent.append(s)


def test_unconfigured():
    assert not fallback.TelegramVoiceNote({}).configured()
    assert fallback.TelegramVoiceNote({"telegram_bot_token": "t", "telegram_chat_id": "1"}).configured()


@pytest.mark.asyncio
async def test_lost_browser_forwards(monkeypatch):
    from voice_agent import app
    calls = []
    async def fake_send(text, clips, note=""):
        calls.append((text, len(clips), note)); return "voice"
    monkeypatch.setattr(app.FALLBACK, "send", fake_send)
    monkeypatch.setattr(app.FALLBACK, "configured", lambda: True)
    t = app.Turn(FakeWS(die_after=1), "voice-test", [])
    await t.send(type="transcript", text="hi")          # delivered
    await t.send(type="token", text="x")                # socket gone -> lost
    assert t.lost
    t.reply = "Four."; t.clips = [("audio/mpeg", b"abc")]
    assert await t.forward("note") == "voice"
    assert calls == [("Four.", 1, "note")]


@pytest.mark.asyncio
async def test_sendvoice_falls_back_to_text(monkeypatch):
    fb = fallback.TelegramVoiceNote({"telegram_bot_token": "t", "telegram_chat_id": "1", "telegram_api": "http://tg.test"})
    posts = []
    class R:
        def __init__(self, code): self.status_code, self.text = code, ""
    class C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, data=None, files=None):
            posts.append(url.rsplit("/", 1)[1]); return R(200 if url.endswith("sendMessage") else 500)
    monkeypatch.setattr(fallback.httpx, "AsyncClient", C)
    async def no_ffmpeg(clips, timeout=60.0): return b"OggS"
    monkeypatch.setattr(fallback, "_to_ogg_opus", no_ffmpeg)
    assert await fb.send("hello", [("audio/mpeg", b"x")]) == "text"
    assert posts == ["sendVoice", "sendMessage"]
