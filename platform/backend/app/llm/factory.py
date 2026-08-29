import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from app.config import settings
from .base import LLMProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .local_provider import LocalProvider

# Default model names per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "ollama": "llama3",
    "local": "local-nlp",
}

NLP_MODEL_DIR = Path(__file__).parents[2] / "data" / "nlp_models"
REQUIRED_NLP_ARTIFACTS = (
    "intent_model.pkl",
    "emotion_model.pkl",
    "sub_intent_model.pkl",
    "confidence_model.pkl",
    "urgency_model.pkl",
    "politeness_model.pkl",
    "learning_style_model.pkl",
)


@dataclass(frozen=True)
class ProviderResolution:
    provider: LLMProvider
    configured_provider: str
    configured_model: str
    effective_provider: str
    effective_model: str
    status: str
    reason: str | None = None
    runtime_warnings: tuple[str, ...] = ()


def get_local_runtime_warnings() -> tuple[str, ...]:
    """Inspect local model prerequisites without loading models or using the network."""
    warnings: list[str] = []
    metadata_path = NLP_MODEL_DIR / "training_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trained_version = str(metadata.get("environment", {}).get("scikit_learn", ""))
        runtime_version = version("scikit-learn")
        if trained_version and trained_version.split(".")[:2] != runtime_version.split(".")[:2]:
            warnings.append("sklearn_model_version_mismatch")
    except (OSError, json.JSONDecodeError, PackageNotFoundError):
        warnings.append("model_metadata_unavailable")

    if any(not (NLP_MODEL_DIR / name).is_file() for name in REQUIRED_NLP_ARTIFACTS):
        warnings.append("model_artifacts_missing")

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_path = Path(HF_HUB_CACHE) / (
            "models--sentence-transformers--paraphrase-multilingual-MiniLM-L6-v2"
        ) / "snapshots"
        if not cache_path.is_dir() or not any(cache_path.iterdir()):
            warnings.append("semantic_model_uncached")
    except (ImportError, OSError):
        warnings.append("semantic_model_unavailable")

    return tuple(warnings)


def create_llm_provider(provider: str | None = None, model: str | None = None) -> LLMProvider:
    """Create an LLM provider. If provider/model not specified, use global settings."""
    p = provider or settings.llm_provider
    m = model or (settings.model_name if not provider else DEFAULT_MODELS.get(p, ""))
    match p:
        case "anthropic":
            return AnthropicProvider(api_key=settings.anthropic_api_key, model=m)
        case "openai":
            return OpenAIProvider(api_key=settings.openai_api_key, model=m)
        case "ollama":
            return OllamaProvider(base_url=settings.ollama_base_url, model=m)
        case "local":
            return LocalProvider()
        case _:
            raise ValueError(f"Unknown LLM provider: {p}")


def resolve_llm_provider(
    provider: str | None = None,
    model: str | None = None,
) -> ProviderResolution:
    """Resolve configured and effective providers without claiming network reachability."""
    configured_provider = provider or settings.llm_provider
    configured_model = model or DEFAULT_MODELS.get(configured_provider, settings.model_name)
    reason = None
    effective_provider = configured_provider
    runtime_warnings: tuple[str, ...] = ()

    if configured_provider not in DEFAULT_MODELS:
        effective_provider = "local"
        reason = "unknown_provider"
    elif configured_provider == "openai" and not settings.openai_api_key:
        effective_provider = "local"
        reason = "missing_api_key"
    elif configured_provider == "anthropic" and not settings.anthropic_api_key:
        effective_provider = "local"
        reason = "missing_api_key"

    if effective_provider == "local":
        resolved = LocalProvider()
        effective_model = resolved.model_name
        runtime_warnings = get_local_runtime_warnings()
        if reason is None and runtime_warnings:
            reason = "runtime_degraded"
        status = "degraded" if reason or runtime_warnings else "ready"
    else:
        resolved = create_llm_provider(effective_provider, configured_model)
        effective_model = getattr(resolved, "model", configured_model)
        status = "configured"

    return ProviderResolution(
        provider=resolved,
        configured_provider=configured_provider,
        configured_model=configured_model,
        effective_provider=effective_provider,
        effective_model=effective_model,
        status=status,
        reason=reason,
        runtime_warnings=runtime_warnings,
    )


def list_available_providers() -> list[dict]:
    """Return which providers are configured (have API keys or are local)."""
    # Local is always first — always available, zero cost
    providers = [
        {"id": "local", "name": "本地 NLP (免 API)", "models": ["local-nlp"]},
    ]
    if settings.anthropic_api_key:
        providers.append({"id": "anthropic", "name": "Claude (Anthropic)", "models": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]})
    if settings.openai_api_key:
        providers.append({"id": "openai", "name": "GPT (OpenAI)", "models": ["gpt-4o", "gpt-4o-mini"]})
    providers.append({"id": "ollama", "name": "本地大模型 (Ollama)", "models": ["llama3", "mistral", "gemma2"]})
    return providers
