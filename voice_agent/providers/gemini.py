"""Google leg for stage 6: the Gemini API (AI Studio key). Audio understanding for STT, Gemini TTS for speech.

Cloud Speech-to-Text / Text-to-Speech adapters come when Rock's Cloud billing clears review.
Gemini TTS returns raw PCM16 at 24 kHz; we wrap it in a WAV header.
"""
from __future__ import annotations

import base64

import httpx

from ..audio import pcm16_to_wav, wav_duration_seconds
from ..vision import prompt_for
from .base import Provider, ProviderError, STTResult, TTSResult, VisionRequest, VisionResult, raise_for_status


class GeminiProvider(Provider):
    name = "gemini"
    supports_vision = True
    stt_usd_per_minute = 0.002          # flash audio input ~32 tokens/s at $1/M tokens, plus output
    tts_usd_per_1k_chars = 0.012        # flash TTS preview, estimate
    vision_usd_per_m_in = 0.75          # flash text/image/video input, list price Oct 2026 (3.7 flash rate)
    vision_usd_per_m_out = 3.75

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.key = (cfg.get("api_key") or "").strip()
        self.base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.stt_model = cfg.get("stt_model", "gemini-3.8-flash")
        self.tts_model = cfg.get("tts_model", "gemini-2.5-flash-preview-tts")
        self.voice = cfg.get("voice", "Kore")
        self.vision_model = cfg.get("vision_model") or self.stt_model
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

    async def describe(self, req: VisionRequest) -> VisionResult:
        """Clip and still in one request: Gemini reads video natively (about one frame a second)."""
        parts: list[dict] = [{"text": prompt_for(req.question)}]
        used = []
        if req.clip:
            parts.append({"text": "The clip from the hold:"})
            parts.append({"inline_data": {"mime_type": req.clip_mime or "video/mp4",
                                          "data": base64.b64encode(req.clip).decode()}})
            used.append("clip")
        if req.still:
            parts.append({"text": "The still taken at release (sharpest frame):"})
            parts.append({"inline_data": {"mime_type": req.still_mime or "image/jpeg",
                                          "data": base64.b64encode(req.still).decode()}})
            used.append("still")
        if not used:
            raise ProviderError("gemini vision: nothing to look at", retryable=False)
        body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}}
        try:
            r = await self._client.post(f"{self.base}/models/{self.vision_model}:generateContent", json=body,
                                        timeout=httpx.Timeout(90.0, connect=5.0))
        except httpx.HTTPError as e:
            raise ProviderError(f"gemini vision: {e}") from e
        raise_for_status(r, "gemini vision")
        j = r.json()
        try:
            text = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"]).strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        if not text:
            raise ProviderError("gemini vision: empty description", retryable=True)
        usage = j.get("usageMetadata") or {}
        tin = int(usage.get("promptTokenCount") or 0)
        tout = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        if not tin:   # no usage block: estimate from list rates (~300 tokens per clip second, ~1k per still)
            tin = 400 + (300 * 8 if req.clip else 0) + (1000 if req.still else 0)
            tout = tout or 250
        usd = round(tin / 1e6 * self.vision_usd_per_m_in + tout / 1e6 * self.vision_usd_per_m_out, 6)
        return VisionResult(text=text, usd=usd, tokens_in=tin, tokens_out=tout, used="+".join(used))

    async def close(self) -> None:
        await self._client.aclose()
