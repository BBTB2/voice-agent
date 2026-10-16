"""Audio helpers: everything the STT adapters see is 16 kHz mono PCM16 WAV."""
from __future__ import annotations

import asyncio
import io
import struct
import wave


class AudioError(Exception):
    pass


def pcm16_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_duration_seconds(wav: bytes) -> float:
    """Duration from the data length, not the header: a WAV streamed out of ffmpeg carries a bogus size."""
    try:
        with wave.open(io.BytesIO(wav)) as w:
            bytes_per_second = w.getframerate() * w.getnchannels() * w.getsampwidth()
        return max(0.0, (len(wav) - 44) / float(bytes_per_second or 1))
    except (wave.Error, EOFError, struct.error):
        return 0.0


def wav_pcm16(wav: bytes) -> tuple[bytes, int]:
    """(pcm bytes, sample rate) of a PCM16 WAV; tolerant of a bogus header size."""
    with wave.open(io.BytesIO(wav)) as w:
        rate = w.getframerate()
    return wav[44:], rate


def is_wav_16k_mono(data: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
    except (wave.Error, EOFError, struct.error):
        return False


async def to_wav16k(data: bytes, timeout: float = 20.0) -> bytes:
    """Transcode any browser container (webm/opus, mp4/aac, ogg, wav) to 16 kHz mono PCM16 WAV via ffmpeg.

    ffmpeg writes raw PCM to the pipe and we add the header ourselves: a WAV written to a pipe gets
    placeholder size fields, which made every downstream duration (and cost estimate) nonsense.
    """
    if not data:
        raise AudioError("empty audio")
    if is_wav_16k_mono(data):
        return data
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", "16000", "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(data), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise AudioError("ffmpeg timed out")
    if proc.returncode != 0 or not out:
        raise AudioError(f"ffmpeg failed: {err.decode(errors='replace')[-300:]}")
    return pcm16_to_wav(out, 16000)
