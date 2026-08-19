"""Background ingestion: uploaded files are embedded off the request path.

Uploads return immediately; a worker pulls each file off a queue, chunks it,
embeds the chunks (the slow part) and writes them to the store.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .chunking import chunk_text
from .config import Settings
from .embeddings import EmbeddingClient
from .store import Chunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestJob:
    name: str
    data: bytes


class IngestQueue:
    def __init__(
        self, settings: Settings, embeddings: EmbeddingClient, store: VectorStore
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._store = store
        self._queue: asyncio.Queue[IngestJob] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        # one worker keeps ingest order simple for the demo
        self._worker = asyncio.create_task(self._run())

    async def enqueue(self, name: str, data: bytes) -> None:
        await self._queue.put(IngestJob(name=name, data=data))

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process(job)
            except Exception:
                logger.exception("failed to ingest %s", job.name)
            finally:
                self._queue.task_done()

    async def _process(self, job: IngestJob) -> None:
        pieces = chunk_text(
            job.data.decode("utf-8", errors="replace"),
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        if not pieces:
            return
        vectors = await self._embeddings.embed(pieces)
        self._store.add([Chunk(text=p, source=job.name) for p in pieces], vectors)
        logger.info("ingested %s (%d chunks)", job.name, len(pieces))
