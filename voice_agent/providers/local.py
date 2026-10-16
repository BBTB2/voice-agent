"""The free floor: faster-whisper (CPU, int8) for STT and Piper for TTS. Zero marginal cost, never on quota."""
from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from pathlib import Path

from .base import Provider, ProviderError, STTResult, TTSResult

LOG = logging.getLogger("voice.local")


class LocalProvider(Provider):
    name = "local"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.models_dir = Path(cfg.get("models_dir") or os.environ.get("VOICE_MODELS_DIR", "/models"))
        self.whisper_model = cfg.get("whisper_model", "small")
        self.compute_type = cfg.get("compute_type", "int8")
        self.threads = int(cfg.get("threads", 4))
        self.piper_voice = cfg.get("piper_voice", "en_GB-alan-medium")
        self._whisper = None
        self._piper = None
        self._lock = asyncio.Lock()

    def configured(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def _piper_path(self) -> Path:
        return self.models_dir / "piper" / f"{self.piper_voice}.onnx"

    def _load_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel  # heavy import, on demand
            self._whisper = WhisperModel(self.whisper_model, device="cpu", compute_type=self.compute_type,
                                         cpu_threads=self.threads,
                                         download_root=str(self.models_dir / "whisper"), local_files_only=False)
        return self._whisper

    def _load_piper(self):
        if self._piper is None:
            from piper import PiperVoice
            p = self._piper_path()
            if not p.exists():
                raise ProviderError(f"piper voice missing: {p}", retryable=False)
            self._piper = PiperVoice.load(str(p))
        return self._piper

    def _transcribe_sync(self, wav16k: bytes, language: str) -> str:
        import numpy as np
        from ..audio import wav_pcm16
        model = self._load_whisper()
        pcm, _rate = wav_pcm16(wav16k)          # already 16 kHz mono PCM16; skip PyAV decoding entirely
        audio = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = model.transcribe(audio, language=(language or "en").split("-")[0],
                                           beam_size=1, vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def _synthesize_sync(self, text: str) -> bytes:
        voice = self._load_piper()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize_wav(text, w)
        return buf.getvalue()

    async def transcribe(self, wav16k: bytes, language: str) -> STTResult:
        from ..audio import wav_duration_seconds
        async with self._lock:  # one CPU job at a time
            try:
                text = await asyncio.get_running_loop().run_in_executor(None, self._transcribe_sync, wav16k, language)
            except ProviderError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ProviderError(f"local stt: {e}") from e
        return STTResult(text=text, seconds=wav_duration_seconds(wav16k), usd=0.0)

    async def synthesize(self, text: str, language: str) -> TTSResult:
        async with self._lock:
            try:
                audio = await asyncio.get_running_loop().run_in_executor(None, self._synthesize_sync, text)
            except ProviderError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ProviderError(f"local tts: {e}") from e
        return TTSResult(audio=audio, mime="audio/wav", chars=len(text), usd=0.0)
