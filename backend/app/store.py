"""A small in-memory vector store backed by numpy cosine similarity.

Good enough for a tech test: no external database, everything lives in the
process. A lock guards the arrays because the ingest worker writes to the store
while research requests read from it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[np.ndarray] = field(default_factory=list)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must line up")
        with self._lock:
            for chunk, vector in zip(chunks, embeddings):
                self._chunks.append(chunk)
                self._vectors.append(_normalize(np.asarray(vector, dtype=np.float32)))

    def search(self, query: list[float], k: int) -> list[Retrieved]:
        with self._lock:
            if not self._vectors:  # nothing indexed yet
                return []
            matrix = np.vstack(self._vectors)
            chunks = list(self._chunks)
        q = _normalize(np.asarray(query, dtype=np.float32))
        scores = matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [Retrieved(chunk=chunks[i], score=float(scores[i])) for i in top]

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
