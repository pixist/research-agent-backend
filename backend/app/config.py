"""Application settings, loaded from the environment."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM / embedding provider (OpenAI-compatible).
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"

    @property
    def use_fake_provider(self) -> bool:
        """Fall back to a deterministic offline provider when no key is set."""
        return not self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
