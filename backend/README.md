# Research Agent Backend

A small FastAPI service behind the provided research-agent frontend. It takes a question, pulls context from your uploaded files and a web search, and streams a markdown answer back to the browser.

## Layout

The provided frontend and this backend sit side by side:

```
frontend/   the provided React app (Vite)
backend/    this service
```

## Setup

Backend needs Python 3.12+ and [uv](https://docs.astral.sh/uv/); the frontend needs Node.

### Backend

```sh
cd backend
uv sync
cp .env.example .env   # optional — add an OPENAI_API_KEY for real answers
uv run uvicorn app.main:app --port 8787 --reload
```

Without an `OPENAI_API_KEY` it uses a deterministic offline provider, so you can click through the UI with no key.

### Frontend

In another terminal, next to the backend:

```sh
cd frontend
npm install
npm run dev
```

It defaults to the backend on `http://localhost:8787`; set `VITE_API_BASE_URL` to point elsewhere.

### Using OpenRouter

The chat client is OpenAI-compatible, so you can point it at OpenRouter and pick any model it serves:

```sh
OPENAI_API_KEY=sk-or-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
CHAT_MODEL=openai/gpt-4o-mini   # or any model id on OpenRouter
```

Important detail: OpenRouter has no embeddings endpoint, so uploads/retrieval need a provider that can embed. Give the embedding client its own key — otherwise it falls back to the offline hash and retrieval won't be meaningful:

```sh
EMBED_API_KEY=sk-...                     # e.g. an OpenAI key
EMBED_BASE_URL=https://api.openai.com/v1
```

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

## Examples

### With an uploaded source

Upload a `.txt` and ask about it:

```
> Summarise the uploaded notes on photosynthesis and compare with current sources.
```

The answer streams in as markdown and ends with a **Sources** list mixing your uploads and web results.

### With web search

A real run (chat on OpenRouter, no upload), trimmed:

```
> Best hostels in lake bled

Here are some of the best hostels in Lake Bled, Slovenia:

### 1. Hostel Bled
- Dorms from EUR 16.00. Vibrant atmosphere, friendly staff, near the lake,
  with kitchen and common areas. [2]

### 2. Alp Penzion
- Brokepackr score 9.5/10. Praised for cleanliness and comfort. [5]

### 3. Bled Hostel
- Dormitory and private rooms, close to the attractions, affordable.

...(trimmed)

---
### Sources
1. The 5 Best Hostels in Lake Bled - The Broke Backpacker
2. Best Hostels in Bled from EUR 16.00 | 2026 - Hostelworld
3. The 7 Best Hostels In Bled (Updated 2026)
4. The 6 BEST Backpacker Hostels in Lake Bled (2025)
5. 5 Best Hostels in Lake Bled (2026) - Brokepackr
```

The **Sources** block is the web results the agent actually used, cited inline as `[n]`.

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
