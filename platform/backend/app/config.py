import logging
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_env: Literal['development', 'test', 'production'] = 'production'
    database_path: Path = Path(__file__).resolve().parent.parent / 'data' / 'app.db'
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "local"
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "local-nlp"
    jwt_secret: str = "change-me-in-production-use-a-long-random-string"
    jwt_expire_minutes: int = Field(default=480, ge=1, le=1440)
    default_admin_password: str = "admin123"
    cors_origins: str = "http://localhost:5173,https://dong-wei-yang.github.io,https://kevin19830331-ml-dl-viz-api.hf.space"

    model_config = {"env_file": ".env"}

    @model_validator(mode='after')
    def validate_production_security(self):
        if self.app_env == 'production' and (
            len(self.jwt_secret) < 32
            or self.jwt_secret == 'change-me-in-production-use-a-long-random-string'
        ):
            raise ValueError('Production requires JWT_SECRET with at least 32 characters; default secrets are forbidden')
        return self


settings = Settings()

if settings.jwt_secret == "change-me-in-production-use-a-long-random-string":
    logger.warning("JWT_SECRET is using default value. Set JWT_SECRET in .env for production.")
