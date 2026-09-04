# Qwen3 RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant built on **Qwen3** (served locally via **Ollama**), with a FAISS vector store, a Streamlit UI, and a production-ready FastAPI service — packaged for one-command deployment with Docker.

Originally built as a CLI/Streamlit prototype, this version adds a REST API, containerization, CI, and a set of grounding/config bug fixes found during review.

---

## Features

- **Retrieval-Augmented Generation** — answers are grounded strictly in retrieved document context, not model memory
- **Local LLM inference** — runs Qwen3 through Ollama, no external API calls or API keys required
- **FAISS vector store** — fast local similarity search over embedded document chunks
- **Hallucination / grounding evaluation** — a dedicated evaluation chain scores responses for faithfulness to source context, with automatic regeneration if a response is under-grounded
- **Two interfaces**
  - `app.py` — Streamlit UI for interactive chat
  - `api/main.py` — FastAPI REST service for programmatic access
- **Streaming responses** — Server-Sent Events (SSE) endpoint for token-by-token output
- **Dockerized** — Ollama, the API, and the Streamlit UI run together via `docker-compose`
- **CI** — GitHub Actions runs the full pytest suite on Python 3.10 and 3.11, plus a Docker build check, on every push

---

## Architecture

```
                     ┌─────────────────┐
                     │   Documents      │
                     │  (data/raw/)     │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  Ingestion        │
                     │  loader → splitter│
                     │  → embedder        │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  FAISS Vector     │
                     │  Store             │
                     └────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│  Streamlit UI    │   │  FastAPI Service │   │  Evaluation /     │
│  (app.py)        │   │  (api/main.py)   │   │  Regeneration      │
└───────┬────────┘   └────────┬────────┘   │  chains             │
        │                     │             └────────┬────────┘
        └──────────┬──────────┘                       │
                    │                                  │
           ┌────────▼──────────┐                       │
           │  Query Service      │◄──────────────────────┘
           │  → RAG Chain         │
           │  → Qwen3 (Ollama)    │
           └──────────────────────┘
```

**Query flow:** a user query is embedded and used to retrieve the top-k relevant chunks from FAISS → the RAG chain builds a strictly-grounded prompt from those chunks → Qwen3 (via Ollama) generates a response → the evaluation chain checks the response for grounding, triggering the regeneration chain (with a conservative fallback) if the response strays from the retrieved context.

---

## Project structure

```
Qwen3-RAG-Assistant/
├── api/                  # FastAPI service (main.py: /query, /query/stream, /health, /history)
├── app.py                # Streamlit UI
├── chains/                # RAG, evaluation, and regeneration chains
├── config/                # Settings, constants, and prompt templates
├── data/                  # raw / processed / metadata document storage
├── ingestion/              # Document loading, splitting, embedding
├── llm/                    # Qwen3 (Ollama) client and streaming callbacks
├── scripts/                # ingest.py, evaluate.py, check_db.py — CLI utilities
├── services/                # Logging and query orchestration services
├── tests/                   # pytest suite (API, retrieval, prompts, grounding)
├── utils/                   # Context, validation, and helper utilities
├── vectorstore/              # FAISS store + retriever
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Getting started

### Prerequisites

- Python 3.10 or 3.11
- [Ollama](https://ollama.com) installed locally, with the Qwen3 model pulled:
  ```bash
  ollama pull qwen3
  ```
- (Optional) Docker + Docker Compose, if you'd rather run everything in containers

### Local setup

```bash
git clone https://github.com/Kushagra-Kapoor-04/Qwen3-RAG-Assistant.git
cd Qwen3-RAG-Assistant

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env          # adjust settings as needed
```

Ingest your documents (place source files in `data/raw/` first):

```bash
python scripts/ingest.py
```

Run the Streamlit UI:

```bash
streamlit run app.py
```

Or run the API:

```bash
uvicorn api.main:app --reload
```

### Docker setup

```bash
docker-compose up --build
```

This starts Ollama, the FastAPI service, and the Streamlit UI together. See `docker-compose.yml` for exposed ports and volume mounts.

---

## API reference

Base URL (local): `http://localhost:8000`

| Endpoint          | Method | Description                                      |
|--------------------|--------|---------------------------------------------------|
| `/health`           | GET    | Health check                                       |
| `/query`            | POST   | Submit a query, get a full grounded response        |
| `/query/stream`     | POST   | Same as above, streamed via Server-Sent Events (SSE) |
| `/history`          | GET    | Retrieve recent query history                        |

**Example — `/query`:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about deployment?"}'
```

```json
{
  "answer": "...",
  "sources": ["..."],
  "grounded": true
}
```

**Example — `/query/stream`:**

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the ingestion pipeline."}'
```

Streams tokens as they're generated via SSE.

---

## Configuration

Key settings live in `config/settings.py` and `.env` (see `.env.example` for the full list):

| Variable            | Default | Description                                  |
|----------------------|---------|------------------------------------------------|
| `CHUNK_SIZE`          | `500`   | Document chunk size for splitting                |
| `CHUNK_OVERLAP`       | `50`    | Overlap between chunks                            |
| `OLLAMA_MODEL`        | `qwen3` | Model name served by Ollama                       |
| `OLLAMA_BASE_URL`     | —       | Ollama server URL                                  |
| `VECTORSTORE_PATH`    | —       | Local path for the FAISS index                       |
| `MAX_REGEN_ATTEMPTS`  | —       | Max regeneration attempts before conservative fallback |

---

## Testing

```bash
pytest
```

The suite covers the API (mocked, no live Ollama/FAISS required), retrieval, prompt construction, and grounding/evaluation behavior. CI runs this automatically on Python 3.10 and 3.11 for every push, plus a Docker build check.

---

## Bug fixes in this release

- Fixed a broken f-string in the regeneration fallback path (`chains/regeneration_chain.py`) that crashed the assistant whenever it hit max regeneration attempts
- Aligned `chunk_size`/`chunk_overlap` defaults in `config/settings.py` with `config/constants.py` (previously inconsistent)
- Corrected the RAG prompt wording to explicitly enforce strict grounding

---

## License

See [LICENSE](./LICENSE).