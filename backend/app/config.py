"""Application settings, loaded from the environment."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Chat provider (OpenAI-compatible). Point base_url at OpenRouter to use it.
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"

    # Embedding provider. Falls back to the chat provider when unset.
    embed_api_key: str | None = None
    embed_base_url: str | None = None
    embed_model: str = "text-embedding-3-small"

    # Retrieval.
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 4

    # External search.
    search_max_results: int = 5

    # Uploads are embedded on a background worker pool.
    ingest_workers: int = 2
    max_upload_bytes: int = 5 * 1024 * 1024

    # CORS: the provided vite frontend runs on 5173.
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def use_fake_provider(self) -> bool:
        """Fall back to a deterministic offline provider when no chat key is set."""
        return not self.openai_api_key

    @property
    def effective_embed_api_key(self) -> str | None:
        return self.embed_api_key or self.openai_api_key

    @property
    def effective_embed_base_url(self) -> str:
        return self.embed_base_url or self.openai_base_url

    @property
    def use_fake_embeddings(self) -> bool:
        """Embeddings fall back to the hash independently of chat."""
        return not self.effective_embed_api_key


def openrouter_headers(base_url: str) -> dict[str, str]:
    """OpenRouter uses these for attribution/routing. Harmless on other providers."""
    if "openrouter.ai" in base_url:
        return {"HTTP-Referer": "http://localhost:8787", "X-Title": "research-agent-backend"}
    return {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
