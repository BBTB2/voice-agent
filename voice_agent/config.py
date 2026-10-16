"""YAML config with ${ENV} expansion and profile overlays."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
HERE = Path(__file__).resolve().parent.parent  # voice/


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None, profile: str | None = None) -> dict:
    """default.yaml, then the profile overlay (VOICE_PROFILE), then an explicit file (VOICE_CONFIG)."""
    cfg_dir = Path(os.environ.get("VOICE_CONFIG_DIR", HERE / "config"))
    cfg = yaml.safe_load((cfg_dir / "default.yaml").read_text()) or {}
    profile = profile or os.environ.get("VOICE_PROFILE", "")
    if profile:
        p = cfg_dir / "profiles" / f"{profile}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"profile {profile!r} not found at {p}")
        cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
    path = path or os.environ.get("VOICE_CONFIG", "")
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    cfg = _expand(cfg)
    cfg.setdefault("service", {})["profile"] = profile or "default"
    return cfg
