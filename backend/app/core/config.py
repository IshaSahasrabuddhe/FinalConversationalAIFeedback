from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Feedback Collector"
    api_v1_prefix: str = "/api/v1"
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24
    algorithm: str = "HS256"
    database_url: str = "sqlite:///./feedback_collector.db"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    issue_category_confidence_threshold: float = 0.7
    frontend_origin: str = "http://localhost:5173"
    admin_email: str = "admin@system.com"
    admin_password: str = "Admin@123"
    admin_static_token: str = "admin-static-token"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("frontend_origin", mode="before")
    @classmethod
    def normalize_frontend_origin(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def cors_origins(self) -> List[str]:
        return [self.frontend_origin]


@lru_cache
def get_settings() -> Settings:
    return Settings()
