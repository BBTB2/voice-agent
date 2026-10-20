"""The waterfall (decision 4a): ordered providers per leg; fall through on 429, 5xx, timeout (8 s),
any adapter error, or a monthly budget cap per provider; 15-minute cooldown before a failed
provider is tried again. Usage persists in <data_dir>/usage.json per calendar month.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from .providers import Provider, ProviderError

LOG = logging.getLogger("voice.waterfall")


def month_key(now: float | None = None) -> str:
    return datetime.fromtimestamp(now or time.time(), tz=timezone.utc).strftime("%Y-%m")


class Usage:
    """Per-month, per-provider spend estimate. Small JSON file, rewritten on every change."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict[str, dict[str, float]]] = {}
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def row(self, provider: str, now: float | None = None) -> dict[str, float]:
        return self.data.setdefault(month_key(now), {}).setdefault(
            provider, {"usd": 0.0, "calls": 0, "stt_seconds": 0.0, "tts_chars": 0, "failures": 0})

    def add(self, provider: str, *, usd: float = 0.0, stt_seconds: float = 0.0, tts_chars: int = 0,
            failure: bool = False, looks: int = 0) -> None:
        r = self.row(provider)
        r["usd"] = round(r["usd"] + usd, 6)
        r["calls"] += 0 if failure else 1
        r["stt_seconds"] = round(r["stt_seconds"] + stt_seconds, 2)
        r["tts_chars"] += tts_chars
        r["failures"] += 1 if failure else 0
        if looks:
            r["looks"] = int(r.get("looks", 0)) + looks     # camera descriptions (stage 10)
        self._save()

    def spent(self, provider: str) -> float:
        return float(self.data.get(month_key(), {}).get(provider, {}).get("usd", 0.0))

    # Stage 9: the same spend counted once more per app (row "app:<name>"), so a site has its own monthly cap.
    def add_app(self, app: str, usd: float) -> None:
        r = self.row(f"app:{app}")
        r["usd"] = round(r["usd"] + usd, 6)
        r["calls"] += 1
        self._save()

    def app_spent(self, app: str) -> float:
        return self.spent(f"app:{app}")

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            os.replace(tmp, self.path)
        except OSError as e:
            LOG.warning("usage not saved: %s", e)


class Breaker:
    """Per-provider circuit: closed (ok), cooling (until ts), or parked (config error, until restart)."""

    def __init__(self, cooldown_seconds: float):
        self.cooldown = cooldown_seconds
        self.until: dict[str, float] = {}
        self.reason: dict[str, str] = {}
        self.parked: set[str] = set()

    def available(self, name: str, now: float | None = None) -> bool:
        if name in self.parked:
            return False
        return (now or time.time()) >= self.until.get(name, 0.0)

    def trip(self, name: str, reason: str, retryable: bool = True) -> None:
        self.reason[name] = reason[:200]
        if retryable:
            self.until[name] = time.time() + self.cooldown
        else:
            self.parked.add(name)

    def reset(self, name: str) -> None:
        self.until.pop(name, None)
        self.reason.pop(name, None)
        self.parked.discard(name)

    def state(self, name: str) -> str:
        if name in self.parked:
            return "parked"
        if not self.available(name):
            return "cooling"
        return "ok"


class Exhausted(Exception):
    """Every provider in the leg failed or was unavailable."""

    def __init__(self, leg: str, tried: list[tuple[str, str]]):
        self.leg = leg
        self.tried = tried
        super().__init__(f"{leg}: no provider left; " + "; ".join(f"{n}: {r}" for n, r in tried))


class Waterfall:
    def __init__(self, cfg: dict, providers: dict[str, Provider], data_dir: Path):
        wf = cfg.get("waterfall") or {}
        self.timeout = float(wf.get("timeout_seconds", 8))
        self.local_timeout = float(wf.get("local_timeout_seconds", 60))
        # A leg may need longer than the 8 s speech budget: a clip upload plus a video read takes 10-30 s.
        self.leg_timeouts = {k[:-16]: float(v) for k, v in wf.items() if k.endswith("_timeout_seconds") and k not in ("local_timeout_seconds",)}
        self.default_budget = float(wf.get("default_monthly_budget_usd", 10))
        self.breaker = Breaker(float(wf.get("cooldown_minutes", 15)) * 60)
        self.usage = Usage(data_dir / "usage.json")
        self.providers = providers
        self.legs: dict[str, list[str]] = {k: list(v or []) for k, v in (cfg.get("legs") or {}).items()}
        self.pcfg: dict[str, dict] = cfg.get("providers") or {}
        self.forced_failures = {s for s in os.environ.get("VOICE_FAIL_PROVIDERS", "").split(",") if s}
        self.last_used: dict[str, str] = {}

    # -- policy -------------------------------------------------------------------------------
    def budget(self, name: str) -> float | None:
        if name == "local":
            return None
        b = (self.pcfg.get(name) or {}).get("budget_usd", self.default_budget)
        return None if b in (None, "", "none") else float(b)

    def over_budget(self, name: str) -> bool:
        b = self.budget(name)
        return b is not None and self.usage.spent(name) >= b

    def skip_reason(self, name: str, leg: str, free_only: bool = False) -> str | None:
        p = self.providers.get(name)
        if p is None:
            return "not built"
        if not p.configured():
            return "no key"
        if free_only and self.budget(name) is not None:
            return "app budget reached"
        if not getattr(p, f"supports_{leg}", True):
            return f"no {leg}"
        if self.over_budget(name):
            return f"budget ${self.budget(name):.2f} reached"
        if not self.breaker.available(name):
            return f"{self.breaker.state(name)}: {self.breaker.reason.get(name, '')}"
        return None

    def candidates(self, leg: str, free_only: bool = False) -> list[str]:
        return [n for n in self.legs.get(leg, []) if self.skip_reason(n, leg, free_only) is None]

    def app_over_budget(self, app: str | None, budget: float | None) -> bool:
        """An app past its monthly cap gets the free floor only (stage 9)."""
        return bool(app) and budget is not None and self.usage.app_spent(app) >= budget

    # -- run ----------------------------------------------------------------------------------
    async def run(self, leg: str, call: Callable[[Provider], Awaitable[Any]],
                  on_fallthrough: Callable[[str, str, str | None], Awaitable[None]] | None = None,
                  free_only: bool = False) -> tuple[str, Any]:
        """Try each provider of the leg in order; returns (provider_name, result). ``free_only`` (an app over its
        monthly cap) skips every provider that has a budget, i.e. everything but the local floor."""
        tried: list[tuple[str, str]] = []
        order = self.legs.get(leg, [])
        for name in order:
            why = self.skip_reason(name, leg, free_only)
            if why:
                tried.append((name, why))
                continue
            p = self.providers[name]
            timeout = self.local_timeout if name == "local" else self.leg_timeouts.get(leg, self.timeout)
            t0 = time.monotonic()
            try:
                if name in self.forced_failures:
                    raise ProviderError("forced failure (VOICE_FAIL_PROVIDERS)", status=503)
                result = await asyncio.wait_for(call(p), timeout)
            except asyncio.TimeoutError:
                reason = f"timeout after {timeout:.0f} s"
                self._fail(name, reason, True)
            except ProviderError as e:
                reason = str(e)
                self._fail(name, reason, e.retryable)
            except Exception as e:  # noqa: BLE001  anything else is still 'try the next one'
                reason = f"{type(e).__name__}: {e}"
                self._fail(name, reason, True)
            else:
                self.breaker.reset(name)
                self.last_used[leg] = name
                LOG.info("%s via %s in %.0f ms", leg, name, (time.monotonic() - t0) * 1000)
                return name, result
            tried.append((name, reason))
            nxt = next((n for n in order[order.index(name) + 1:] if self.skip_reason(n, leg, free_only) is None), None)
            LOG.warning("%s: %s failed (%s); next: %s", leg, name, reason, nxt)
            if on_fallthrough:
                await on_fallthrough(name, reason, nxt)
        raise Exhausted(leg, tried)

    def _fail(self, name: str, reason: str, retryable: bool) -> None:
        self.breaker.trip(name, reason, retryable)
        self.usage.add(name, failure=True)

    def record(self, name: str, *, usd: float, stt_seconds: float = 0.0, tts_chars: int = 0, looks: int = 0,
               app: str | None = None) -> None:
        self.usage.add(name, usd=usd, stt_seconds=stt_seconds, tts_chars=tts_chars, looks=looks)
        if app and usd:
            self.usage.add_app(app, usd)

    # -- reporting ----------------------------------------------------------------------------
    def status(self) -> dict:
        out: dict[str, Any] = {"month": month_key(), "legs": {}, "providers": {}}
        for leg, order in self.legs.items():
            out["legs"][leg] = {"order": order, "active": self.candidates(leg), "last_used": self.last_used.get(leg)}
        out["apps"] = {k[4:]: {"spent_usd": v.get("usd", 0.0), "calls": v.get("calls", 0)}
                       for k, v in self.usage.data.get(month_key(), {}).items() if k.startswith("app:")}
        for name, p in self.providers.items():
            out["providers"][name] = {
                "configured": p.configured(),
                "state": self.breaker.state(name),
                "reason": self.breaker.reason.get(name),
                "spent_usd": self.usage.spent(name),
                "budget_usd": self.budget(name),
                "usage": self.usage.row(name),
            }
        return out
