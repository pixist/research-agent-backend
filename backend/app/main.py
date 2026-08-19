"""FastAPI application wiring the research agent to the provided frontend."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from .agent import ResearchAgent
from .config import get_settings
from .embeddings import EmbeddingClient
from .ingest import IngestQueue
from .llm import ChatClient
from .schemas import UploadedFile, UploadResponse
from .search import WebSearch
from .store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = VectorStore()
    embeddings = EmbeddingClient(settings)
    chat = ChatClient(settings)
    search = WebSearch(settings)
    ingest = IngestQueue(settings, embeddings, store)
    ingest.start()

    app.state.settings = settings
    app.state.agent = ResearchAgent(settings, store, embeddings, chat, search)
    app.state.ingest = ingest
    if settings.use_fake_provider:
        logger.warning("no OPENAI_API_KEY set — using the offline demo provider")
    try:
        yield
    finally:
        await ingest.stop()


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
    ingest: IngestQueue = request.app.state.ingest

    uploaded: list[UploadedFile] = []
    for file in files:
        data = await file.read()
        name = file.filename or "upload.txt"
        if len(data) > settings.max_upload_bytes:
            return PlainTextResponse(f"{name} is too large.", status_code=413)
        await ingest.enqueue(name, data)
        uploaded.append(
            UploadedFile(
                name=name, size=len(data), type=file.content_type or "text/plain"
            )
        )
    return UploadResponse(uploaded=uploaded)
