# Research Agent Backend

A small FastAPI service behind the provided research-agent frontend. It takes a question, pulls context from your uploaded files and a web search, and streams a markdown answer back to the browser.

## Setup

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
cd backend
uv sync
cp .env.example .env   # optional — add an OPENAI_API_KEY for real answers
uv run uvicorn app.main:app --port 8787 --reload
```

Without an `OPENAI_API_KEY` it uses a deterministic offline provider, so you can click through the UI with no key. The frontend already points at `http://localhost:8787` — just run `npm run dev` next to it.

## API

- `POST /api/research` — body `{ "request": "..." }`. Streams markdown as `text/markdown`. The frontend concatenates the chunks, so it's plain text, not SSE.
- `POST /api/sources` — multipart upload under the repeated `files` field. Returns `{ "uploaded": [{ "name", "size", "type" }] }`. Files embed on a background queue, so the request returns right away.

## Architecture

```
main.py      FastAPI app, CORS, routes, lifespan
agent.py     gather sources (concurrently) -> stream a grounded answer
store.py     in-memory numpy vector store (cosine similarity)
ingest.py    background upload queue + embedding workers
embeddings.py / llm.py   OpenAI-compatible clients with an offline fallback
search.py    DuckDuckGo web search for external sources
chunking.py  overlapping text splitter
```

Design decisions:

- **In-memory store.** No database — a numpy matrix is enough for a single-process demo and keeps setup to one command. A real vector DB would mean rewriting `store.py` behind the same two methods.
- **Background ingestion.** Embedding is the slow part of an upload, so files are queued and embedded on a worker pool. The endpoint returns as soon as the bytes are read.
- **Concurrent gathering.** Local retrieval and web search run together with `asyncio.gather`.
- **Offline provider.** Lets the whole thing, and the tests, run with no key.

### Things I'd do with more time

- Persist the store and dedupe re-uploaded files.
- A real planning/tool-use loop (several search rounds) instead of one pass.
- Stream tool progress to the UI as status lines.
- Per-session sources instead of one global store.

## Example

Upload a `.txt` and ask about it:

```
> Summarise the uploaded notes on photosynthesis and compare with current sources.
```

The answer streams in as markdown and ends with a **Sources** list mixing your uploads and web results.

## Evaluating the agent

Two things I'd measure:

- **Retrieval** — a small set of (question, relevant-chunk) pairs, scored with recall@k / MRR. Checks whether the store finds the right context before the LLM runs.
- **Answers** — an LLM-as-judge rubric over a fixed question set: faithfulness (are claims backed by the cited sources?), relevance, completeness. Faithfulness is the one that catches hallucination, so I'd start there.

Both are cheap to run in CI, so you can watch a retrieval or prompt change move the number.

## Tests

```sh
cd backend
uv run pytest
```

## Docker

```sh
cd backend
docker build -t research-agent-backend .
docker run -p 8787:8787 --env-file .env research-agent-backend
```
