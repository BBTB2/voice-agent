"""Apps (stage 9): one voice container serves several sites. The token a page connects with picks its app, and the
app says which brain answers, which browser origins may embed the widget, whether the camera leg and the Telegram
fallback apply, and how much paid speech it may spend a month (past that: the free local floor, never a refusal).

Config::

    apps:
      prof-snarky:
        tokens: ["${VOICE_TOKEN_PROF_SNARKY}"]   # empty token = app not loaded
        allowed_origins: "https://www.prof-snarky.com"
        budget_usd: 10
        camera: false
        fallback: false
        brain: {kind: prof-snarky, url: https://www.prof-snarky.com/api/query}

The default app (name = the profile) is what the service always had: ``service.tokens``, ``brain``,
``service.allowed_origins``, ``camera.enabled`` and the Telegram fallback when configured.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field

from .brain import BaseBrain, make_brain

LOG = logging.getLogger("voice.apps")


def _origins(value) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = str(value or "").split(",")
    return {o.strip().rstrip("/").lower() for o in items if o.strip()}


@dataclass
class App:
    name: str
    brain: BaseBrain
    tokens: set[str] = field(default_factory=set)
    allowed_origins: set[str] = field(default_factory=set)
    budget_usd: float | None = None      # paid speech per month for this app; None = only the per-provider caps
    camera: bool = False
    fallback: bool = False
    default: bool = False

    def accepts(self, token: str | None) -> bool:
        return any(secrets.compare_digest(token or "", t) for t in self.tokens)

    def public(self) -> dict:
        return {"name": self.name, "brain": self.brain.kind, "brain_label": self.brain.label or None,
                "brain_configured": self.brain.configured(), "needs_auth": self.brain.needs_auth,
                "origins": sorted(self.allowed_origins), "budget_usd": self.budget_usd,
                "camera": self.camera, "fallback": self.fallback, "tokens": len(self.tokens)}


class AppRegistry:
    def __init__(self, cfg: dict, fallback_configured: bool = False):
        service = cfg.get("service") or {}
        brain_cfg = dict(cfg.get("brain") or {})
        name = str(service.get("profile") or "default")
        self.default = App(name=name, brain=make_brain(brain_cfg),
                           tokens={t for t in (service.get("tokens") or []) if t},
                           allowed_origins=_origins(service.get("allowed_origins")),
                           budget_usd=_budget(service.get("budget_usd")),
                           camera=bool((cfg.get("camera") or {}).get("enabled", True)),
                           fallback=fallback_configured, default=True)
        self.apps: dict[str, App] = {name: self.default}
        for app_name, acfg in (cfg.get("apps") or {}).items():
            acfg = acfg or {}
            tokens = {t for t in (acfg.get("tokens") or []) if t}
            if not tokens:
                LOG.info("app %s has no token; not loaded", app_name)
                continue
            if app_name in self.apps:
                raise ValueError(f"app {app_name!r} clashes with the default app")
            for other in self.apps.values():
                if tokens & other.tokens:
                    raise ValueError(f"app {app_name!r} shares a token with {other.name!r}")
            bcfg = acfg.get("brain") or {}
            if str(bcfg.get("kind") or "openai") == "openai":
                merged = dict(brain_cfg); merged.update(bcfg); bcfg = merged     # an OpenAI-style app inherits the defaults
            self.apps[app_name] = App(name=app_name, brain=make_brain(bcfg), tokens=tokens,
                                      allowed_origins=_origins(acfg.get("allowed_origins")),
                                      budget_usd=_budget(acfg.get("budget_usd", 10)),
                                      camera=bool(acfg.get("camera", False)),
                                      fallback=bool(acfg.get("fallback", False)) and fallback_configured)
        self.open = not any(a.tokens for a in self.apps.values())

    def resolve(self, token: str | None) -> App | None:
        """The app this token belongs to; the default app when the service runs open (no tokens anywhere)."""
        if self.open:
            return self.default
        for app in self.apps.values():
            if app.accepts(token):
                return app
        return None

    def all_origins(self) -> set[str]:
        out: set[str] = set()
        for app in self.apps.values():
            out |= app.allowed_origins
        return out

    def brains(self) -> list[BaseBrain]:
        return [a.brain for a in self.apps.values()]

    def public(self) -> dict:
        return {a.name: a.public() for a in self.apps.values()}


def _budget(v) -> float | None:
    if v in (None, "", "none", "null"):
        return None
    return float(v)
