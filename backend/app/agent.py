"""The research agent: retrieve uploaded context, then stream a grounded answer."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from .config import Settings
from .embeddings import EmbeddingClient
from .llm import ChatClient
from .store import Retrieved, VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research agent. Answer the user's request in clear GitHub-flavoured "
    "markdown. Ground every claim in the provided context and cite sources inline "
    "as [n]. If the context is thin, say so and answer from general knowledge."
)


class ResearchAgent:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embeddings: EmbeddingClient,
        chat: ChatClient,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embeddings
        self._chat = chat

    async def run(self, request: str) -> AsyncIterator[str]:
        """Stream the answer to ``request`` as markdown."""
        try:
            local = await self._retrieve(request)
        except Exception:
            logger.exception("retrieval failed")
            yield "\n\n> Could not gather sources for this request.\n"
            return

        messages = _build_messages(request, local)
        async for token in self._chat.stream(messages):
            yield token
        yield _sources_section(local)

    async def _retrieve(self, request: str) -> list[Retrieved]:
        query_vec = await self._embeddings.embed_one(request)
        return await asyncio.to_thread(
            self._store.search, query_vec, self._settings.retrieval_top_k
        )


def _build_messages(request: str, local: list[Retrieved]) -> list[dict[str, str]]:
    if local:
        context = json.dumps(
            [{"source": r.chunk.source, "text": r.chunk.text} for r in local], indent=2
        )
    else:
        context = "(no sources retrieved)"
    user = f"Context:\n{context}\n\nQuestion: {request}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _sources_section(local: list[Retrieved]) -> str:
    if not local:
        return ""
    lines = ["\n\n---\n\n### Sources\n"]
    for n, item in enumerate(local, start=1):
        lines.append(f"{n}. uploaded — {item.chunk.source}")
    return "\n".join(lines) + "\n"
