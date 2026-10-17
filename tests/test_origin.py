"""Browser-origin allowlist: own host always, listed sites, nothing else (stage 8)."""
import os
import importlib
import sys

import pytest


@pytest.fixture
def app_mod(monkeypatch):
    monkeypatch.setenv("VOICE_ALLOWED_ORIGINS", "https://radread.co.uk, https://sitewright.uk/")
    monkeypatch.setenv("VOICE_TOKEN", "t0k")
    monkeypatch.setenv("VOICE_DATA_DIR", "/tmp/voice-test-origin")
    os.makedirs("/tmp/voice-test-origin", exist_ok=True)
    sys.modules.pop("voice_agent.app", None)
    mod = importlib.import_module("voice_agent.app")
    yield mod
    sys.modules.pop("voice_agent.app", None)


def test_origin_rules(app_mod):
    ok = app_mod._origin_ok
    assert ok(None, "voice.surajmenon.ai")                              # curl / scripts: token decides
    assert ok("https://voice.surajmenon.ai", "voice.surajmenon.ai")     # the page itself
    assert ok("https://radread.co.uk", "voice.surajmenon.ai")           # listed
    assert ok("https://sitewright.uk", "voice.surajmenon.ai")           # listed, trailing slash trimmed
    assert not ok("https://evil.example", "voice.surajmenon.ai")        # not listed
    assert not ok("http://voice.surajmenon.ai.evil.example", "voice.surajmenon.ai")
    assert app_mod.ALLOWED_ORIGINS == {"https://radread.co.uk", "https://sitewright.uk"}


def test_ws_refuses_foreign_origin(app_mod):
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=t0k", headers={"origin": "https://evil.example", "host": "voice.surajmenon.ai"}) as ws:
            ws.receive_json()
    with client.websocket_connect("/ws?token=t0k", headers={"origin": "https://radread.co.uk", "host": "voice.surajmenon.ai"}) as ws:
        msg = ws.receive_json()
        assert msg.get("type") == "ready"
