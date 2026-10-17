"""Google Cloud Speech: Speech-to-Text v1 (synchronous recognize) + Text-to-Speech v1 over REST.

Distinct from the ``gemini`` provider (AI Studio key, Gemini models). This one needs a Google Cloud
project with the two Speech APIs enabled and billing on. Two credential shapes, either works:

* ``api_key``: an API key restricted to Cloud Speech-to-Text and Cloud Text-to-Speech (simplest:
  one value in voice-keys.env as GOOGLE_SPEECH_API_KEY).
* ``credentials_b64``: a service-account JSON key, base64 in one line (GOOGLE_SPEECH_CREDENTIALS_B64).
  We mint an OAuth2 access token ourselves (RS256 JWT bearer grant, no google-auth dependency) and
  refresh it a minute before expiry.

Sync recognize takes up to about 60 s of audio, which matches ``max_utterance_seconds``. Prices
verified 17 Oct 2026 (cloud.google.com/speech-to-text/pricing, text-to-speech pricing): STT v1
standard $0.024 per minute without data logging ($0.016 with; first 60 min a month free), TTS
Neural2 $16 per million characters, Chirp 3 HD $30, Standard $4 (4 M free). The budget estimate
uses the no-logging STT rate and the Neural2 TTS rate unless ``voice`` says otherwise.
"""
from __future__ import annotations

import base64
import json
import time

import httpx

from ..audio import wav_duration_seconds
from .base import Provider, ProviderError, STTResult, TTSResult, raise_for_status

STT_URL = "https://speech.googleapis.com/v1/speech:recognize"
TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# USD per 1k characters by voice family, from the voice name (en-GB-Neural2-B, en-GB-Chirp3-HD-Charon ...)
TTS_PRICE_BY_FAMILY = {"standard": 0.004, "wavenet": 0.004, "neural2": 0.016, "chirp3-hd": 0.030,
                       "chirp": 0.030, "studio": 0.160, "polyglot": 0.016, "journey": 0.030}


def tts_price_for_voice(voice: str) -> float:
    v = voice.lower()
    for fam, price in TTS_PRICE_BY_FAMILY.items():
        if fam in v:
            return price
    return 0.016


def make_jwt_assertion(sa: dict, now: float | None = None) -> str:
    """RS256-signed JWT for the service-account bearer grant (RFC 7523), valid 1 hour."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    now = int(now if now is not None else time.time())
    header = {"alg": "RS256", "typ": "JWT", "kid": sa.get("private_key_id", "")}
    claims = {"iss": sa["client_email"], "scope": SCOPE, "aud": sa.get("token_uri", TOKEN_URL),
              "iat": now, "exp": now + 3600}
    signing_input = b64(json.dumps(header, separators=(",", ":")).encode()) + "." + \
        b64(json.dumps(claims, separators=(",", ":")).encode())
    key = serialization.load_pem_private_key(sa["private_key"].encode(), password=None)
    sig = key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return signing_input + "." + b64(sig)


class GoogleProvider(Provider):
    name = "google"
    stt_usd_per_minute = 0.024          # STT v1 standard, data logging off (0.016 with logging on)
    tts_usd_per_1k_chars = 0.016        # Neural2; recomputed from the voice name in __init__

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.api_key = (cfg.get("api_key") or "").strip()
        self.sa: dict | None = None
        raw = (cfg.get("credentials_b64") or "").strip()
        if raw:
            try:
                self.sa = json.loads(base64.b64decode(raw))
            except (ValueError, json.JSONDecodeError):
                self.sa = {"_broken": True}
        self.stt_model = cfg.get("stt_model", "latest_short")
        self.data_logging = bool(cfg.get("data_logging", False))
        self.voice = cfg.get("voice", "en-GB-Neural2-B")
        self.speaking_rate = float(cfg.get("speaking_rate", 1.0))
        self.output = cfg.get("output_format", "MP3")       # MP3 | LINEAR16 | OGG_OPUS
        self.tts_usd_per_1k_chars = float(cfg.get("tts_usd_per_1k_chars") or tts_price_for_voice(self.voice))
        if self.data_logging:
            self.stt_usd_per_minute = 0.016
        self._token: str | None = None
        self._token_exp = 0.0
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    def configured(self) -> bool:
        return bool(self.api_key) or (self.sa is not None and "client_email" in self.sa)

    # -- auth ---------------------------------------------------------------------------------
    async def _auth(self) -> tuple[dict, dict]:
        """(query params, headers) for the next call."""
        if self.sa and "client_email" in self.sa:
            if not self._token or time.time() > self._token_exp - 60:
                try:
                    assertion = make_jwt_assertion(self.sa)
                except Exception as e:  # noqa: BLE001  bad PEM = configuration error
                    raise ProviderError(f"google auth: bad service-account key ({e})", retryable=False) from e
                try:
                    r = await self._client.post(self.sa.get("token_uri", TOKEN_URL), data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion})
                except httpx.HTTPError as e:
                    raise ProviderError(f"google auth: {e}") from e
                raise_for_status(r, "google auth")
                j = r.json()
                self._token = j["access_token"]
                self._token_exp = time.time() + float(j.get("expires_in", 3600))
            return {}, {"Authorization": f"Bearer {self._token}"}
        return {"key": self.api_key}, {}

    # -- STT ----------------------------------------------------------------------------------
    async def transcribe(self, wav16k: bytes, language: str) -> STTResult:
        from ..audio import wav_pcm16
        pcm, rate = wav_pcm16(wav16k)
        body = {
            "config": {"encoding": "LINEAR16", "sampleRateHertz": rate, "languageCode": language or "en-GB",
                       "model": self.stt_model, "enableAutomaticPunctuation": True},
            "audio": {"content": base64.b64encode(pcm).decode()},
        }
        params, headers = await self._auth()
        try:
            r = await self._client.post(STT_URL, params=params, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"google stt: {e}") from e
        _raise_google(r, "google stt")
        text = " ".join(
            (alt.get("transcript") or "").strip()
            for res in r.json().get("results", [])
            for alt in res.get("alternatives", [])[:1]
        ).strip()
        secs = wav_duration_seconds(wav16k)
        return STTResult(text=text, seconds=secs, usd=self.stt_cost(secs))

    # -- TTS ----------------------------------------------------------------------------------
    async def synthesize(self, text: str, language: str) -> TTSResult:
        lang = language or "en-GB"
        if not self.voice.lower().startswith(lang.lower()[:2]):
            lang = "-".join(self.voice.split("-")[:2])   # the voice decides its own language
        body = {
            "input": {"text": text},
            "voice": {"languageCode": lang, "name": self.voice},
            "audioConfig": {"audioEncoding": self.output, "speakingRate": self.speaking_rate,
                            "sampleRateHertz": 24000},
        }
        params, headers = await self._auth()
        try:
            r = await self._client.post(TTS_URL, params=params, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"google tts: {e}") from e
        _raise_google(r, "google tts")
        content = r.json().get("audioContent")
        if not content:
            raise ProviderError("google tts: no audioContent in response", retryable=False)
        audio = base64.b64decode(content)
        mime = {"MP3": "audio/mpeg", "OGG_OPUS": "audio/ogg"}.get(self.output, "audio/wav")
        return TTSResult(audio=audio, mime=mime, chars=len(text), usd=self.tts_cost(len(text)))

    async def close(self) -> None:
        await self._client.aclose()


def _raise_google(resp: httpx.Response, who: str) -> None:
    """Google errors carry a status word; PERMISSION_DENIED / UNAUTHENTICATED / INVALID_ARGUMENT are config
    errors that a cooldown will not fix, RESOURCE_EXHAUSTED (quota) and UNAVAILABLE are retryable."""
    if resp.status_code < 400:
        return
    status = ""
    try:
        status = resp.json().get("error", {}).get("status", "")
    except ValueError:
        pass
    if status in ("PERMISSION_DENIED", "UNAUTHENTICATED", "INVALID_ARGUMENT", "NOT_FOUND"):
        raise ProviderError(f"{who} HTTP {resp.status_code} {status}: {resp.text[:300]}",
                            status=resp.status_code, retryable=False)
    raise_for_status(resp, who)
