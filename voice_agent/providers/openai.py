"""OpenAI cascade: gpt-4o-mini-transcribe (STT) + gpt-4o-mini-tts (TTS). About $0.01 per conversation minute."""
from __future__ import annotations

import httpx

from ..vision import prompt_for
from .base import Provider, ProviderError, STTResult, TTSResult, VisionRequest, VisionResult, raise_for_status


class OpenAIProvider(Provider):
    name = "openai"
    supports_vision = True
    vision_stills_only = True           # frames are pulled from the clip for it
    stt_usd_per_minute = 0.003          # gpt-4o-mini-transcribe list price
    tts_usd_per_1k_chars = 0.0167       # gpt-4o-mini-tts ~ $0.015 per audio minute (~450 chars per 30 s)
    vision_usd_per_m_in = 0.15          # gpt-4o-mini list price; images ~1k tokens each at high detail
    vision_usd_per_m_out = 0.60

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.key = (cfg.get("api_key") or "").strip()
        self.base = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.stt_model = cfg.get("stt_model", "gpt-4o-mini-transcribe")
        self.tts_model = cfg.get("tts_model", "gpt-4o-mini-tts")
        self.voice = cfg.get("voice", "coral")
        self.instructions = cfg.get("tts_instructions", "")
        self.vision_model = cfg.get("vision_model", "gpt-4o-mini")
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

    async def describe(self, req: VisionRequest) -> VisionResult:
        """Stills only (the fallback leg): the release still plus a few frames pulled from the clip."""
        import base64
        content: list[dict] = [{"type": "text", "text": prompt_for(req.question)}]
        used = []
        for i, frame in enumerate(req.frames or []):
            content.append({"type": "text", "text": f"Frame {i + 1} from the clip:"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame).decode(), "detail": "low"}})
            used.append("frames") if "frames" not in used else None
        if req.still:
            content.append({"type": "text", "text": "The still taken at release (sharpest frame):"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{req.still_mime or 'image/jpeg'};base64," + base64.b64encode(req.still).decode(), "detail": "high"}})
            used.append("still")
        if not used:
            raise ProviderError("openai vision: nothing to look at", retryable=False)
        body = {"model": self.vision_model, "max_tokens": 600, "temperature": 0.2,
                "messages": [{"role": "user", "content": content}]}
        try:
            r = await self._client.post(f"{self.base}/chat/completions", json=body, timeout=httpx.Timeout(90.0, connect=5.0))
        except httpx.HTTPError as e:
            raise ProviderError(f"openai vision: {e}") from e
        raise_for_status(r, "openai vision")
        j = r.json()
        try:
            text = (j["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        if not text:
            raise ProviderError("openai vision: empty description", retryable=True)
        usage = j.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens") or 1500), int(usage.get("completion_tokens") or 250)
        usd = round(tin / 1e6 * self.vision_usd_per_m_in + tout / 1e6 * self.vision_usd_per_m_out, 6)
        return VisionResult(text=text, usd=usd, tokens_in=tin, tokens_out=tout, used="+".join(used))

    async def close(self) -> None:
        await self._client.aclose()
