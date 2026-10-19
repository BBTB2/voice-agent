"""Provider adapter interface. Each adapter does STT and/or TTS and reports its own cost estimate."""
from __future__ import annotations

from dataclasses import dataclass


class ProviderError(Exception):
    """Any failure the waterfall should treat as 'try the next provider'.

    ``status`` carries the HTTP status when there was one; ``retryable`` False means a
    configuration error (bad key, bad model) that a cooldown will not fix, so the provider is
    parked until the next month tick / restart instead of 15 minutes.
    """

    def __init__(self, message: str, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class STTResult:
    text: str
    seconds: float          # audio seconds billed
    usd: float              # estimated cost


@dataclass
class TTSResult:
    audio: bytes
    mime: str               # audio/mpeg or audio/wav
    chars: int
    usd: float


@dataclass
class VisionRequest:
    """What the camera saw during one hold-to-talk, plus the words. Stills and clips are raw bytes."""
    question: str
    still: bytes | None = None          # one full-resolution frame taken at release (JPEG)
    still_mime: str = "image/jpeg"
    clip: bytes | None = None           # the hold, trimmed (video/mp4 after normalise_clip)
    clip_mime: str = "video/mp4"
    frames: list[bytes] | None = None   # JPEG frames pulled from the clip, for stills-only models
    language: str = "en-GB"


@dataclass
class VisionResult:
    text: str               # the model's description, or its honest "cannot see" when the picture is too poor
    usd: float
    tokens_in: int = 0
    tokens_out: int = 0
    used: str = ""          # "clip+still", "still", "frames"


class Provider:
    name = "base"
    supports_stt = True
    supports_tts = True
    supports_vision = False     # describe() implemented (the camera leg, stage 10)
    # list prices used ONLY for the budget estimate (USD); verified 16 Oct 2026, see voice/README.md
    stt_usd_per_minute = 0.0
    tts_usd_per_1k_chars = 0.0

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def configured(self) -> bool:
        return True

    async def transcribe(self, wav16k: bytes, language: str) -> STTResult:
        raise NotImplementedError

    async def synthesize(self, text: str, language: str) -> TTSResult:
        raise NotImplementedError

    async def describe(self, req: VisionRequest) -> VisionResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def stt_cost(self, seconds: float) -> float:
        return round(seconds / 60.0 * self.stt_usd_per_minute, 6)

    def tts_cost(self, chars: int) -> float:
        return round(chars / 1000.0 * self.tts_usd_per_1k_chars, 6)


def raise_for_status(resp, who: str) -> None:
    """Map an httpx response to ProviderError with the right retryability."""
    if resp.status_code < 400:
        return
    body = resp.text[:300].replace("\n", " ")
    retryable = resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500
    raise ProviderError(f"{who} HTTP {resp.status_code}: {body}", status=resp.status_code, retryable=retryable)
