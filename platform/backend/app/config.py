import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "claude-sonnet-4-20250514"
    jwt_secret: str = "change-me-in-production-use-a-long-random-string"
    jwt_expire_minutes: int = 480
    default_admin_password: str = "admin123"
    cors_origins: str = "http://localhost:5173"

    model_config = {"env_file": ".env"}


settings = Settings()

if settings.jwt_secret == "change-me-in-production-use-a-long-random-string":
    logger.warning("JWT_SECRET is using default value. Set JWT_SECRET in .env for production.")
