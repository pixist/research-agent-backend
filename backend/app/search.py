"""Web search tool so the agent can pull in external sources."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearch:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(self, query: str) -> list[SearchResult]:
        if self._settings.use_fake_provider:
            return _fake_results(query)
        # ddgs is synchronous; run it off the event loop.
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> list[SearchResult]:
        from ddgs import DDGS

        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=self._settings.search_max_results):
                results.append(
                    SearchResult(
                        title=hit.get("title", ""),
                        url=hit.get("href", ""),
                        snippet=hit.get("body", ""),
                    )
                )
        return results


def _fake_results(query: str) -> list[SearchResult]:
    return [
        SearchResult(
            title=f"Overview of {query[:60]}",
            url="https://example.com/overview",
            snippet=f"A placeholder external source about {query[:80]}.",
        )
    ]
