"""Embedding client. Wraps the OpenAI embeddings API with a deterministic
offline fallback so the service runs without a key."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import Settings, openrouter_headers

_FAKE_DIM = 256


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        if not settings.use_fake_embeddings:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.effective_embed_api_key,
                base_url=settings.effective_embed_base_url,
                default_headers=openrouter_headers(settings.effective_embed_base_url),
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None:
            return [_fake_embed(text) for text in texts]
        response = await self._client.embeddings.create(
            model=self._settings.embed_model, input=texts
        )
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


def _fake_embed(text: str) -> list[float]:
    """A cheap but stable embedding: hash tokens into a fixed vector.

    Not semantically meaningful, but consistent for the same input, which is all
    the tests and the offline demo need.
    """
    vector = np.zeros(_FAKE_DIM, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % _FAKE_DIM
        vector[idx] += 1.0
    return vector.tolist()
