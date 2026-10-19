"""Camera helpers: trim and shrink a browser clip, pull frames out of it, all through ffmpeg.

Nothing here keeps a file: everything is written to a private temp directory for the ffmpeg call and
deleted before the function returns (the camera leg stores nothing beyond the turn).
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

LOG = logging.getLogger("voice.video")

_EXT = {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov", "video/x-matroska": "mkv"}


class VideoError(Exception):
    pass


def _ext(mime: str | None) -> str:
    return _EXT.get((mime or "").split(";")[0].strip().lower(), "bin")


async def _run(args: list[str], timeout: float) -> tuple[int, bytes]:
    proc = await asyncio.create_subprocess_exec("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise VideoError("ffmpeg timed out")
    return proc.returncode or 0, err


async def normalise_clip(data: bytes, mime: str | None, max_seconds: float = 8.0, max_width: int = 960,
                         fps: int = 4, timeout: float = 40.0) -> tuple[bytes, str]:
    """The last ``max_seconds`` of a browser recording as a small H.264 MP4 without sound.

    Vision models sample about one frame a second, so 4 fps at 960 px wide loses nothing they would use
    and keeps a 5 s clip near 300 KB whatever the phone recorded. Falls back to the original bytes when
    ffmpeg cannot read the container (the model gets the phone's own MP4 or WebM then).
    """
    if not data:
        raise VideoError("empty clip")
    with tempfile.TemporaryDirectory(prefix="clip-") as d:
        src = os.path.join(d, f"in.{_ext(mime)}")
        dst = os.path.join(d, "out.mp4")
        with open(src, "wb") as f:
            f.write(data)
        args = ["-sseof", f"-{max_seconds:g}", "-i", src, "-an",
                "-vf", f"scale='min({max_width},iw)':-2", "-r", str(fps),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", dst]
        rc, err = await _run(args, timeout)
        if rc != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            # -sseof needs a seekable, well-formed file; a fragmented recording may refuse it. Try without the trim.
            rc2, err2 = await _run(args[2:], timeout)
            if rc2 != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
                LOG.warning("clip not normalised (%s); sending the original", (err2 or err).decode(errors="replace")[-200:])
                return data, (mime or "video/mp4").split(";")[0]
        with open(dst, "rb") as f:
            return f.read(), "video/mp4"


async def clip_frames(data: bytes, mime: str | None, count: int = 3, max_width: int = 1280,
                      timeout: float = 30.0) -> list[bytes]:
    """``count`` JPEG frames spread over the clip, for vision models that take stills only."""
    if not data:
        return []
    with tempfile.TemporaryDirectory(prefix="frames-") as d:
        src = os.path.join(d, f"in.{_ext(mime)}")
        with open(src, "wb") as f:
            f.write(data)
        secs = await clip_seconds(src, timeout)
        step = max(secs / max(count, 1), 0.4) if secs else 1.0
        rc, err = await _run(["-i", src, "-vf", f"fps=1/{step:.3f},scale='min({max_width},iw)':-2",
                              "-frames:v", str(count), "-q:v", "4", os.path.join(d, "f%d.jpg")], timeout)
        if rc != 0:
            LOG.warning("frame extraction failed: %s", err.decode(errors="replace")[-200:])
        out: list[bytes] = []
        for i in range(1, count + 1):
            p = os.path.join(d, f"f{i}.jpg")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    out.append(f.read())
        return out


async def clip_seconds(path: str, timeout: float = 15.0) -> float:
    proc = await asyncio.create_subprocess_exec("ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                "-of", "default=nw=1:nk=1", path,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        return float(out.decode().strip() or 0)
    except (asyncio.TimeoutError, ValueError):
        proc.kill() if proc.returncode is None else None
        return 0.0
