"""OpenAI cascade: gpt-4o-mini-transcribe (STT) + gpt-4o-mini-tts (TTS). About $0.01 per conversation minute."""
from __future__ import annotations

import httpx

from .base import Provider, ProviderError, STTResult, TTSResult, raise_for_status


class OpenAIProvider(Provider):
    name = "openai"
    stt_usd_per_minute = 0.003          # gpt-4o-mini-transcribe list price
    tts_usd_per_1k_chars = 0.0167       # gpt-4o-mini-tts ~ $0.015 per audio minute (~450 chars per 30 s)

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.key = (cfg.get("api_key") or "").strip()
        self.base = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.stt_model = cfg.get("stt_model", "gpt-4o-mini-transcribe")
        self.tts_model = cfg.get("tts_model", "gpt-4o-mini-tts")
        self.voice = cfg.get("voice", "coral")
        self.instructions = cfg.get("tts_instructions", "")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0),
                                         headers={"Authorization": f"Bearer {self.key}"})

    def configured(self) -> bool:
        return bool(self.key)

    async def transcribe(self, wav16k: bytes, language: str) -> STTResult:
        from ..audio import wav_duration_seconds
        data = {"model": self.stt_model, "response_format": "json"}
        if language:
            data["language"] = language.split("-")[0]
        try:
            r = await self._client.post(f"{self.base}/audio/transcriptions", data=data,
                                        files={"file": ("audio.wav", wav16k, "audio/wav")})
        except httpx.HTTPError as e:
            raise ProviderError(f"openai stt: {e}") from e
        raise_for_status(r, "openai stt")
        text = (r.json().get("text") or "").strip()
        secs = wav_duration_seconds(wav16k)
        return STTResult(text=text, seconds=secs, usd=self.stt_cost(secs))

    async def synthesize(self, text: str, language: str) -> TTSResult:
        body = {"model": self.tts_model, "voice": self.voice, "input": text, "response_format": "mp3"}
        if self.instructions:
            body["instructions"] = self.instructions
        try:
            r = await self._client.post(f"{self.base}/audio/speech", json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"openai tts: {e}") from e
        raise_for_status(r, "openai tts")
        return TTSResult(audio=r.content, mime="audio/mpeg", chars=len(text), usd=self.tts_cost(len(text)))

    async def close(self) -> None:
        await self._client.aclose()
