"""Google Cloud Speech adapter, offline: request shape, auth modes, response parsing, error mapping."""
import base64
import json

import httpx
import pytest

from voice_agent.audio import pcm16_to_wav
from voice_agent.config import load_config
from voice_agent.providers import ProviderError, build_providers
from voice_agent.providers.google import GoogleProvider, make_jwt_assertion, tts_price_for_voice

WAV = pcm16_to_wav(b"\x00\x00" * 16000, 16000)   # one second of silence at 16 kHz


def mock(provider, handler):
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.asyncio
async def test_stt_with_api_key_parses_transcript():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"results": [{"alternatives": [{"transcript": "hello world", "confidence": 0.9}]},
                                                     {"alternatives": [{"transcript": "again"}]}]})
    p = mock(GoogleProvider({"api_key": "k123"}), handler)
    res = await p.transcribe(WAV, "en-GB")
    assert res.text == "hello world again"
    assert "key=k123" in seen["url"] and "speech.googleapis.com/v1/speech:recognize" in seen["url"]
    cfg = seen["body"]["config"]
    assert cfg["encoding"] == "LINEAR16" and cfg["sampleRateHertz"] == 16000
    assert cfg["languageCode"] == "en-GB" and cfg["model"] == "latest_short"
    assert len(base64.b64decode(seen["body"]["audio"]["content"])) == 32000   # raw PCM, header stripped
    assert res.seconds == pytest.approx(1.0, abs=0.01)
    assert res.usd == pytest.approx(0.024 / 60, rel=0.01)


@pytest.mark.asyncio
async def test_tts_returns_mp3_and_prices_by_voice_family():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"audioContent": base64.b64encode(b"ID3fake").decode()})
    p = mock(GoogleProvider({"api_key": "k", "voice": "en-GB-Standard-A"}), handler)
    res = await p.synthesize("Hello there.", "en-GB")
    assert res.audio == b"ID3fake" and res.mime == "audio/mpeg" and res.chars == 12
    assert seen["body"]["voice"] == {"languageCode": "en-GB", "name": "en-GB-Standard-A"}
    assert seen["body"]["audioConfig"]["audioEncoding"] == "MP3"
    assert res.usd == pytest.approx(12 / 1000 * 0.004)
    assert tts_price_for_voice("en-GB-Chirp3-HD-Charon") == 0.030
    assert tts_price_for_voice("en-GB-Neural2-B") == 0.016


@pytest.mark.asyncio
async def test_service_account_mints_bearer_token_once():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM,
                                                          serialization.PrivateFormat.PKCS8,
                                                          serialization.NoEncryption()).decode()
    sa = {"type": "service_account", "client_email": "voice@proj.iam.gserviceaccount.com", "private_key": pem,
          "private_key_id": "abc", "token_uri": "https://oauth2.googleapis.com/token"}
    assertion = make_jwt_assertion(sa, now=1000)
    header, claims, sig = assertion.split(".")
    claims = json.loads(base64.urlsafe_b64decode(claims + "=="))
    assert claims["iss"] == sa["client_email"] and claims["exp"] == 4600 and "cloud-platform" in claims["scope"]

    calls = []

    def handler(req: httpx.Request):
        calls.append(str(req.url))
        if "oauth2" in str(req.url):
            assert b"jwt-bearer" in req.content
            return httpx.Response(200, json={"access_token": "tok-1", "expires_in": 3600})
        assert req.headers["authorization"] == "Bearer tok-1" and "key=" not in str(req.url)
        return httpx.Response(200, json={"audioContent": base64.b64encode(b"x").decode()})
    b64 = base64.b64encode(json.dumps(sa).encode()).decode()
    p = mock(GoogleProvider({"credentials_b64": b64}), handler)
    assert p.configured()
    await p.synthesize("one", "en-GB")
    await p.synthesize("two", "en-GB")
    assert sum("oauth2" in c for c in calls) == 1 and len(calls) == 3


def test_unconfigured_and_broken_credentials():
    assert not GoogleProvider({}).configured()
    assert not GoogleProvider({"credentials_b64": "not base64 json"}).configured()
    assert GoogleProvider({"api_key": "k"}).configured()


@pytest.mark.asyncio
async def test_error_mapping():
    def denied(req):
        return httpx.Response(403, json={"error": {"code": 403, "status": "PERMISSION_DENIED",
                                                   "message": "Cloud Speech-to-Text API has not been used in project"}})
    with pytest.raises(ProviderError) as e:
        await mock(GoogleProvider({"api_key": "k"}), denied).transcribe(WAV, "en-GB")
    assert e.value.retryable is False and "PERMISSION_DENIED" in str(e.value)

    def quota(req):
        return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}})
    with pytest.raises(ProviderError) as e:
        await mock(GoogleProvider({"api_key": "k"}), quota).synthesize("x", "en-GB")
    assert e.value.retryable is True and e.value.status == 429


def test_default_config_has_google_between_azure_and_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_SPEECH_API_KEY", "k")
    cfg = load_config(profile="mahadeva")
    assert cfg["legs"]["stt"] == ["openai", "azure", "google", "gemini", "local"]
    assert cfg["legs"]["tts"] == ["openai", "azure", "google", "gemini", "local"]
    provs = build_providers(cfg)
    assert provs["google"].configured() and provs["google"].voice == "en-GB-Neural2-B"
    assert provs["google"].stt_usd_per_minute == 0.024
    nhs = load_config(profile="nhs-demo")
    assert "google" not in nhs["legs"]["stt"]       # decision 5b: Microsoft-only speech path
