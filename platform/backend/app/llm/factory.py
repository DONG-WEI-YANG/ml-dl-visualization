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
