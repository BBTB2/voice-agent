import asyncio
import time
from pathlib import Path

import pytest

from voice_agent.brain import SentenceSplitter, speakable
from voice_agent.config import load_config
from voice_agent.providers import Provider, ProviderError, STTResult, TTSResult
from voice_agent.waterfall import Exhausted, Waterfall


class Fake(Provider):
    def __init__(self, name, fail=None, delay=0.0, key=True, usd=0.01):
        super().__init__({})
        self.name, self.fail, self.delay, self.key, self.usd = name, fail, delay, key, usd
        self.calls = 0

    def configured(self):
        return self.key

    async def transcribe(self, wav, language):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        return STTResult(text=f"hello from {self.name}", seconds=2.0, usd=self.usd)

    async def synthesize(self, text, language):
        self.calls += 1
        if self.fail:
            raise self.fail
        return TTSResult(audio=b"RIFF", mime="audio/wav", chars=len(text), usd=self.usd)


def make(tmp_path, providers, cfg_extra=None):
    cfg = {"waterfall": {"timeout_seconds": 0.2, "cooldown_minutes": 15, "default_monthly_budget_usd": 10},
           "providers": {p.name: {} for p in providers},
           "legs": {"stt": [p.name for p in providers], "tts": [p.name for p in providers]}}
    if cfg_extra:
        cfg["providers"].update(cfg_extra)
    return Waterfall(cfg, {p.name: p for p in providers}, Path(tmp_path))


@pytest.mark.asyncio
async def test_first_provider_wins(tmp_path):
    a, b = Fake("a"), Fake("b")
    wf = make(tmp_path, [a, b])
    name, res = await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    assert name == "a" and res.text == "hello from a" and b.calls == 0


@pytest.mark.asyncio
async def test_429_falls_through_and_cools(tmp_path):
    a, b = Fake("a", fail=ProviderError("rate", status=429)), Fake("b")
    wf = make(tmp_path, [a, b])
    seen = []

    async def cb(name, reason, nxt):
        seen.append((name, nxt))
    name, _ = await wf.run("stt", lambda p: p.transcribe(b"", "en"), cb)
    assert name == "b" and seen == [("a", "b")]
    assert wf.breaker.state("a") == "cooling"
    # second call skips a without calling it
    await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    assert a.calls == 1


@pytest.mark.asyncio
async def test_timeout_falls_through(tmp_path):
    a, b = Fake("a", delay=1.0), Fake("b")
    wf = make(tmp_path, [a, b])
    name, _ = await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    assert name == "b" and "timeout" in wf.breaker.reason["a"]


@pytest.mark.asyncio
async def test_non_retryable_parks(tmp_path):
    a, b = Fake("a", fail=ProviderError("bad key", status=401, retryable=False)), Fake("b")
    wf = make(tmp_path, [a, b])
    await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    assert wf.breaker.state("a") == "parked"
    wf.breaker.reset("a")
    assert wf.breaker.state("a") == "ok"


@pytest.mark.asyncio
async def test_budget_cap_skips_provider(tmp_path):
    a, b = Fake("a", usd=6.0), Fake("b")
    wf = make(tmp_path, [a, b], {"a": {"budget_usd": 10}})
    name, res = await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    wf.record(name, usd=res.usd, stt_seconds=res.seconds)
    name, res = await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    wf.record(name, usd=res.usd, stt_seconds=res.seconds)
    assert wf.usage.spent("a") == 12.0
    assert wf.skip_reason("a", "stt").startswith("budget")
    name, _ = await wf.run("stt", lambda p: p.transcribe(b"", "en"))
    assert name == "b"
    # persisted
    wf2 = make(tmp_path, [Fake("a"), Fake("b")])
    assert wf2.usage.spent("a") == 12.0


@pytest.mark.asyncio
async def test_no_key_skipped_and_exhausted(tmp_path):
    a, b = Fake("a", key=False), Fake("b", fail=ProviderError("boom", status=500))
    wf = make(tmp_path, [a, b])
    with pytest.raises(Exhausted) as e:
        await wf.run("tts", lambda p: p.synthesize("hi", "en"))
    assert ("a", "no key") in e.value.tried and a.calls == 0
    st = wf.status()
    assert st["providers"]["b"]["state"] == "cooling" and st["legs"]["tts"]["active"] == []


def test_config_profiles_and_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VOICE_TOKEN", "t")
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    cfg = load_config(profile="nhs-demo")
    assert cfg["legs"]["stt"] == ["azure", "local"]
    assert cfg["providers"]["openai"]["api_key"] == "sk-test"
    assert cfg["providers"]["azure"]["key"] == ""
    assert cfg["providers"]["azure"]["region"] == "swedencentral"
    assert cfg["service"]["tokens"] == ["t"] and cfg["service"]["profile"] == "nhs-demo"
    cfg = load_config(profile="mahadeva")
    assert cfg["brain"]["url"].endswith(":8077/v1") and cfg["legs"]["tts"][-1] == "local"


def test_sentence_splitter_and_speakable():
    sp = SentenceSplitter(min_chars=10)
    out = []
    for d in ["Yes. I can ", "do that now. Here is **the** plan:\n- one\n- two. See [docs](http://x).", " Done"]:
        out += sp.feed(d)
    out += sp.flush()
    assert out[0] == "Yes. I can do that now."
    assert "**" not in " ".join(out) and "http" not in " ".join(out)
    assert out[-1] == "Done"
    assert speakable("> ⚙ read_file(x)\nHello *world*") == "Hello world"


@pytest.mark.asyncio
async def test_to_wav16k_header_and_duration():
    """ffmpeg output must carry a real size header so durations (and spend) are right."""
    import io, wave, shutil
    from voice_agent.audio import to_wav16k, wav_duration_seconds, is_wav_16k_mono
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x00\x08" * 24000 * 3)
    out = await to_wav16k(buf.getvalue())
    assert is_wav_16k_mono(out)
    with wave.open(io.BytesIO(out)) as w:
        assert abs(w.getnframes() / 16000 - 3.0) < 0.05
    assert abs(wav_duration_seconds(out) - 3.0) < 0.05
