from .base import Provider, ProviderError, STTResult, TTSResult

REGISTRY: dict[str, type[Provider]] = {}


def _register() -> None:
    from .azure import AzureProvider
    from .gemini import GeminiProvider
    from .local import LocalProvider
    from .openai import OpenAIProvider
    for cls in (OpenAIProvider, AzureProvider, GeminiProvider, LocalProvider):
        REGISTRY[cls.name] = cls


_register()


def build_providers(cfg: dict) -> dict[str, Provider]:
    """Instantiate every provider named in config; unconfigured ones are kept so /status can say why."""
    out: dict[str, Provider] = {}
    for name, pcfg in (cfg.get("providers") or {}).items():
        cls = REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"unknown provider {name!r}; known: {sorted(REGISTRY)}")
        out[name] = cls(pcfg or {})
    return out


__all__ = ["Provider", "ProviderError", "STTResult", "TTSResult", "REGISTRY", "build_providers"]
