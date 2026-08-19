"""FastAPI application wiring the research agent to the provided frontend."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from .agent import ResearchAgent
from .chunking import chunk_text
from .config import get_settings
from .embeddings import EmbeddingClient
from .llm import ChatClient
from .schemas import UploadedFile, UploadResponse
from .store import Chunk, VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = VectorStore()
    embeddings = EmbeddingClient(settings)
    chat = ChatClient(settings)

    app.state.settings = settings
    app.state.store = store
    app.state.embeddings = embeddings
    app.state.agent = ResearchAgent(settings, store, embeddings, chat)
    yield


app = FastAPI(title="Research Agent Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,  # the vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def research(request: Request):
    body = await request.json()
    question = (body or {}).get("request", "")
    if not isinstance(question, str) or not question.strip():
        return PlainTextResponse("A non-empty 'request' is required.", status_code=400)

    agent: ResearchAgent = request.app.state.agent
    return StreamingResponse(
        agent.run(question.strip()), media_type="text/markdown; charset=utf-8"
    )


@app.post("/api/sources")
async def sources(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    embeddings = request.app.state.embeddings

    uploaded: list[UploadedFile] = []
    for file in files:
        data = await file.read()
        name = file.filename or "upload.txt"
        pieces = chunk_text(
            data.decode("utf-8", errors="replace"),
            settings.chunk_size,
            settings.chunk_overlap,
        )
        if pieces:
            vectors = await embeddings.embed(pieces)
            store.add([Chunk(text=p, source=name) for p in pieces], vectors)
        uploaded.append(
            UploadedFile(
                name=name, size=len(data), type=file.content_type or "text/plain"
            )
        )
    return UploadResponse(uploaded=uploaded)
