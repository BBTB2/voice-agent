"""Google leg for stage 6: the Gemini API (AI Studio key). Audio understanding for STT, Gemini TTS for speech.

Cloud Speech-to-Text / Text-to-Speech adapters come when Rock's Cloud billing clears review.
Gemini TTS returns raw PCM16 at 24 kHz; we wrap it in a WAV header.
"""
from __future__ import annotations

import base64

import httpx

from ..audio import pcm16_to_wav, wav_duration_seconds
from .base import Provider, ProviderError, STTResult, TTSResult, raise_for_status


class GeminiProvider(Provider):
    name = "gemini"
    stt_usd_per_minute = 0.002          # flash audio input ~32 tokens/s at $1/M tokens, plus output
    tts_usd_per_1k_chars = 0.012        # flash TTS preview, estimate

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.key = (cfg.get("api_key") or "").strip()
        self.base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.stt_model = cfg.get("stt_model", "gemini-3.8-flash")
        self.tts_model = cfg.get("tts_model", "gemini-2.5-flash-preview-tts")
        self.voice = cfg.get("voice", "Kore")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0),
                                         headers={"x-goog-api-key": self.key})

    def configured(self) -> bool:
        return bool(self.key)

    async def transcribe(self, wav16k: bytes, language: str) -> STTResult:
        body = {
            "contents": [{"parts": [
                {"text": f"Transcribe this {language or 'English'} speech verbatim. Reply with the transcript only, "
                         "no quotes, no commentary. If there is no speech reply with an empty string."},
                {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav16k).decode()}},
            ]}],
            "generationConfig": {"temperature": 0.0},
        }
        try:
            r = await self._client.post(f"{self.base}/models/{self.stt_model}:generateContent", json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"gemini stt: {e}") from e
        raise_for_status(r, "gemini stt")
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        secs = wav_duration_seconds(wav16k)
        return STTResult(text=text, seconds=secs, usd=self.stt_cost(secs))

    async def synthesize(self, text: str, language: str) -> TTSResult:
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}},
            },
        }
        try:
            r = await self._client.post(f"{self.base}/models/{self.tts_model}:generateContent", json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"gemini tts: {e}") from e
        raise_for_status(r, "gemini tts")
        try:
            part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("gemini tts: no audio in response", retryable=False)
        pcm = base64.b64decode(part["data"])
        mime = part.get("mimeType", "audio/L16;rate=24000")
        rate = 24000
        if "rate=" in mime:
            try:
                rate = int(mime.split("rate=")[1].split(";")[0])
            except ValueError:
                pass
        return TTSResult(audio=pcm16_to_wav(pcm, rate), mime="audio/wav", chars=len(text), usd=self.tts_cost(len(text)))

    async def close(self) -> None:
        await self._client.aclose()
