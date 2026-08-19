"""The research agent: gather sources, then stream a grounded markdown answer."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from .config import Settings
from .embeddings import EmbeddingClient
from .llm import ChatClient
from .search import SearchResult, WebSearch
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
        search: WebSearch,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embeddings
        self._chat = chat
        self._search = search

    async def run(self, request: str) -> AsyncIterator[str]:
        """Stream the answer to ``request`` as markdown."""
        try:
            local, web = await self._gather(request)
        except Exception:
            logger.exception("gathering sources failed")
            yield "\n\n> Could not gather sources for this request.\n"
            return

        messages = _build_messages(request, local, web)
        try:
            async for token in self._chat.stream(messages):
                yield token
        except Exception:
            logger.exception("generation failed")
            yield "\n\n> The model stopped generating before finishing.\n"
            return

        yield _sources_section(local, web)

    async def _gather(
        self, request: str
    ) -> tuple[list[Retrieved], list[SearchResult]]:
        """Retrieve uploaded context and external results concurrently."""
        query_vec = await self._embeddings.embed_one(request)
        local_task = asyncio.to_thread(
            self._store.search, query_vec, self._settings.retrieval_top_k
        )
        web_task = self._search.search(request)
        local, web = await asyncio.gather(local_task, web_task)
        return local, web


def _build_messages(
    request: str, local: list[Retrieved], web: list[SearchResult]
) -> list[dict[str, str]]:
    blocks: list[str] = []
    n = 1
    for item in local:
        blocks.append(f"[{n}] (uploaded: {item.chunk.source}) {item.chunk.text}")
        n += 1
    for hit in web:
        blocks.append(f"[{n}] (web: {hit.url}) {hit.title} — {hit.snippet}")
        n += 1
    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    user = f"Context:\n{context}\n\nQuestion: {request}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _sources_section(local: list[Retrieved], web: list[SearchResult]) -> str:
    lines = ["\n\n---\n\n### Sources\n"]
    n = 1
    for item in local:
        lines.append(f"{n}. uploaded — {item.chunk.source}")
        n += 1
    for hit in web:
        lines.append(f"{n}. [{hit.title}]({hit.url})")
        n += 1
    if n == 1:
        return ""
    return "\n".join(lines) + "\n"
