"""Stage 10, the camera leg: tagged frames, clip + still through a fake vision provider, degrade to still-only,
"could not see" when nothing arrives, the still forwarded to the brain as an image part, and the ffmpeg helpers."""
import asyncio
import base64
import io
import json
import subprocess
import wave

import pytest
from fastapi.testclient import TestClient

from test_ws import brain_server  # noqa: F401  (fixture)


def _wav(seconds=1.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(rate * seconds))
    return buf.getvalue()


def _clip(seconds=2, colour="red"):
    """A tiny solid-colour MP4 made by ffmpeg (what a phone sends, in spirit)."""
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                           f"color=c={colour}:s=160x120:r=10", "-t", str(seconds), "-pix_fmt", "yuv420p",
                           "-movflags", "+frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"],
                          capture_output=True, check=True).stdout


def _jpeg(colour="green"):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                           f"color=c={colour}:s=64x48", "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
                          capture_output=True, check=True).stdout


@pytest.fixture()
def app(tmp_path, monkeypatch, brain_server):  # noqa: F811
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOICE_TOKEN", "secret")
    monkeypatch.setenv("BRAIN_URL", "http://127.0.0.1:18077/v1")
    monkeypatch.setenv("VOICE_PROFILE", "mahadeva")
    monkeypatch.delenv("VOICE_FAIL_PROVIDERS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("GEMINI_API_KEY", "g-x")
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    import importlib
    import voice_agent.app as appmod
    appmod = importlib.reload(appmod)
    appmod.CLIP_WAIT_SECONDS = 0.6
    from voice_agent.providers import STTResult, TTSResult, VisionResult
    seen = []

    async def fake_transcribe(self, wav, language):
        return STTResult(text="what bird is that", seconds=1.0, usd=0.001)

    async def fake_synth(self, text, language):
        return TTSResult(audio=b"ID3fake", mime="audio/mpeg", chars=len(text), usd=0.001)

    async def fake_describe(self, req):
        seen.append((self.name, req))
        if self.name == "gemini" and getattr(appmod, "_gemini_fails", False):
            from voice_agent.providers import ProviderError
            raise ProviderError("gemini vision HTTP 503", status=503)
        return VisionResult(text="A small brown bird on a sill",
                            usd=0.004, tokens_in=3000, tokens_out=200,
                            used="+".join(x for x, ok in (("clip", req.clip), ("still", req.still), ("frames", req.frames)) if ok))
    for name in ("openai", "gemini"):
        p = appmod.PROVIDERS[name]
        monkeypatch.setattr(type(p), "transcribe", fake_transcribe)
        monkeypatch.setattr(type(p), "synthesize", fake_synth)
        monkeypatch.setattr(type(p), "describe", fake_describe)
    monkeypatch.setattr(type(appmod.PROVIDERS["local"]), "configured", lambda self: False)
    appmod._seen = seen
    appmod._gemini_fails = False
    return appmod


def _tag(tag, data):
    return bytes([tag]) + data


def _collect(ws):
    events = []
    while True:
        m = json.loads(ws.receive_text())
        if m["type"] == "tick":
            continue
        events.append(m)
        if m["type"] == "done":
            return events


def test_status_shows_eyes(app):
    c = TestClient(app.app)
    h = c.get("/healthz").json()
    assert h["camera"] is True and h["vision"] == ["gemini", "openai"]
    st = c.get("/api/status?token=secret").json()
    assert st["camera"]["enabled"] and st["waterfall"]["legs"]["vision"]["order"] == ["gemini", "openai"]
    assert "enableCamera" in c.get("/voice.js").text and 'id="camera"' in c.get("/").text


def test_clip_and_still_reach_gemini_and_brain(app, brain_server):  # noqa: F811
    c = TestClient(app.app)
    still, clip = _jpeg(), _clip()
    with c.websocket_connect("/ws?token=secret") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["camera"] is True and ready["vision"] == ["gemini", "openai"]
        ws.send_text(json.dumps({"type": "camera", "on": True}))
        ws.send_text(json.dumps({"type": "clip_begin", "mime": "video/mp4"}))
        ws.send_bytes(_tag(1, _wav(1.2)))
        ws.send_bytes(_tag(3, still))                                  # still first
        ws.send_text(json.dumps({"type": "utterance_end", "camera": True, "clip": "pending"}))
        for i in range(0, len(clip), 4096):                            # then the clip, in chunks
            ws.send_bytes(_tag(2, clip[i:i + 4096]))
        ws.send_text(json.dumps({"type": "clip_end"}))
        events = _collect(ws)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "transcript" and events[0]["text"] == "what bird is that"
    v = next(e for e in events if e["type"] == "vision")
    assert v["provider"] == "gemini" and v["used"] == "clip+still" and "brown bird" in v["text"] and v["usd"] > 0
    assert events[-1]["reply"].startswith("You said")
    name, req = app._seen[-1]
    assert name == "gemini" and req.question == "what bird is that" and req.still == still
    assert req.clip and req.clip_mime == "video/mp4" and req.frames is None    # Gemini got the video itself
    # the brain got words + description as text, and the still as an image part
    content = brain_server[-1]["messages"][-1]["content"]
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert "what bird is that" in content[0]["text"] and "described by gemini" in content[0]["text"]
    assert content[1]["type"] == "image_url" and content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == still
    st = c.get("/api/status?token=secret").json()["waterfall"]["providers"]["gemini"]
    assert st["usage"]["looks"] == 1 and st["spent_usd"] >= 0.004


def test_late_clip_degrades_to_still(app):
    c = TestClient(app.app)
    still = _jpeg("blue")
    with c.websocket_connect("/ws?token=secret") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "camera", "on": True}))
        ws.send_text(json.dumps({"type": "clip_begin", "mime": "video/mp4"}))
        ws.send_bytes(_tag(1, _wav(1.0)))
        ws.send_bytes(_tag(3, still))
        ws.send_text(json.dumps({"type": "utterance_end", "camera": True, "clip": "pending"}))
        events = _collect(ws)                                           # clip_end never comes
    v = next(e for e in events if e["type"] == "vision")
    assert v["provider"] == "gemini" and v["used"] == "still" and "did not arrive" in v["note"]
    assert "waiting for the clip" in [e.get("text") for e in events if e["type"] == "status"]


def test_openai_fallback_gets_frames(app):
    app._gemini_fails = True
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=secret") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "camera", "on": True}))
        ws.send_text(json.dumps({"type": "clip_begin", "mime": "video/mp4"}))
        ws.send_bytes(_tag(2, _clip(2, "red")))
        ws.send_text(json.dumps({"type": "clip_end"}))
        ws.send_text(json.dumps({"type": "text", "text": "what colour", "camera": True}))
        events = _collect(ws)
    v = next(e for e in events if e["type"] == "vision")
    assert v["provider"] == "openai" and v["used"] == "clip+frames"
    name, req = app._seen[-1]
    assert name == "openai" and req.frames and len(req.frames) >= 1 and req.frames[0][:2] == b"\xff\xd8"
    assert any("gemini failed" in (e.get("text") or "") for e in events if e["type"] == "status")


def test_nothing_arrived_says_so(app, brain_server):  # noqa: F811
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=secret") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "camera", "on": True}))
        ws.send_text(json.dumps({"type": "text", "text": "look at this", "camera": True, "clip": "none"}))
        events = _collect(ws)
    v = next(e for e in events if e["type"] == "vision")
    assert v["provider"] == "none" and "no picture" in v["note"]
    assert "could not see" in brain_server[-1]["messages"][-1]["content"]
    assert app._seen == []


def test_untagged_audio_still_works(app):
    """A 0.2.x widget (no camera message, bare audio frames) is unchanged."""
    c = TestClient(app.app)
    with c.websocket_connect("/ws?token=secret") as ws:
        ws.receive_text()
        ws.send_bytes(_wav(1.0))
        ws.send_text(json.dumps({"type": "utterance_end"}))
        events = _collect(ws)
    assert [e["type"] for e in events][0] == "transcript" and not any(e["type"] == "vision" for e in events)


@pytest.mark.asyncio
async def test_video_helpers():
    from voice_agent.video import clip_frames, normalise_clip
    raw = _clip(12, "red")
    out, mime = await normalise_clip(raw, "video/mp4", max_seconds=3, max_width=320, fps=4)
    assert mime == "video/mp4" and out[:12] != raw[:12] and 0 < len(out) < len(raw) + 4096
    frames = await clip_frames(out, "video/mp4", count=3)
    assert 1 <= len(frames) <= 3 and all(f[:2] == b"\xff\xd8" for f in frames)
    bad, bad_mime = await normalise_clip(b"not a video at all", "video/mp4")
    assert bad == b"not a video at all" and bad_mime == "video/mp4"


def test_brain_text_and_prompt():
    from voice_agent.vision import brain_text, prompt_for
    assert "what is this" in prompt_for("what is this") and "invent text" in prompt_for("")
    t = brain_text("how much is it", "A shelf label reading £4.99", "gemini")
    assert t.startswith("how much is it") and "£4.99" in t and "described by gemini" in t
    t2 = brain_text("", None, None, "no picture arrived")
    assert "could not see" in t2 and "pointed the camera" in t2
