"""Azure AI Speech cascade via the REST endpoints (no SDK): short-audio STT + neural TTS.

Works with a multi-service (AIServices) resource key and its region. EU/UK regions for the NHS
profile: swedencentral (Rock's resource), uksouth, westeurope. About $0.015 per conversation minute.
"""
from __future__ import annotations

import html

import httpx

from .base import Provider, ProviderError, STTResult, TTSResult, raise_for_status


class AzureProvider(Provider):
    name = "azure"
    stt_usd_per_minute = 1.0 / 60.0     # $1 per audio hour, standard STT
    tts_usd_per_1k_chars = 0.015        # $15 per 1M characters, neural voices

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.key = (cfg.get("key") or "").strip()
        self.region = (cfg.get("region") or "").strip()
        self.voice = cfg.get("voice", "en-GB-RyanNeural")
        self.output = cfg.get("output_format", "audio-24khz-48kbitrate-mono-mp3")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0),
                                         headers={"Ocp-Apim-Subscription-Key": self.key})

    def configured(self) -> bool:
        return bool(self.key and self.region)

    async def transcribe(self, wav16k: bytes, language: str) -> STTResult:
        from ..audio import wav_duration_seconds
        url = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        try:
            r = await self._client.post(url, params={"language": language or "en-GB", "format": "simple"},
                                        content=wav16k,
                                        headers={"Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                                                 "Accept": "application/json"})
        except httpx.HTTPError as e:
            raise ProviderError(f"azure stt: {e}") from e
        raise_for_status(r, "azure stt")
        j = r.json()
        status = j.get("RecognitionStatus", "")
        if status not in ("Success", "NoMatch", "InitialSilenceTimeout"):
            raise ProviderError(f"azure stt: {status}", retryable=status not in ("BabbleTimeout",))
        text = (j.get("DisplayText") or "").strip()
        secs = wav_duration_seconds(wav16k)
        return STTResult(text=text, seconds=secs, usd=self.stt_cost(secs))

    async def synthesize(self, text: str, language: str) -> TTSResult:
        lang = language or "en-GB"
        ssml = (f"<speak version='1.0' xml:lang='{lang}'>"
                f"<voice name='{self.voice}'>{html.escape(text)}</voice></speak>")
        url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        try:
            r = await self._client.post(url, content=ssml.encode(),
                                        headers={"Content-Type": "application/ssml+xml",
                                                 "X-Microsoft-OutputFormat": self.output,
                                                 "User-Agent": "voice-agent"})
        except httpx.HTTPError as e:
            raise ProviderError(f"azure tts: {e}") from e
        raise_for_status(r, "azure tts")
        mime = "audio/mpeg" if "mp3" in self.output else "audio/wav"
        return TTSResult(audio=r.content, mime=mime, chars=len(text), usd=self.tts_cost(len(text)))

    async def close(self) -> None:
        await self._client.aclose()
