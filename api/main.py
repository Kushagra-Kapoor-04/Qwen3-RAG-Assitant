"""
Qwen3 RAG Assistant — REST API

Exposes the existing QueryService over HTTP with FastAPI so the assistant
can be consumed by any client (curl, frontend, other services), not just
the Streamlit UI. Includes streaming (SSE) and standard JSON endpoints.

Run:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings
from services.query_service import QueryService, create_query_service
from services.logging_service import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Qwen3 RAG Assistant API",
    description=(
        "Production REST API for a Retrieval-Augmented Generation assistant "
        "with strict grounding, hallucination detection, and automatic "
        "answer regeneration."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    global _query_service
    if _query_service is None:
        _query_service = create_query_service()
    return _query_service


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Documents to retrieve")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold")
    enable_evaluation: bool = Field(True, description="Run hallucination check")
    enable_regeneration: bool = Field(True, description="Regenerate if ungrounded")


class SourceInfo(BaseModel):
    source: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    is_grounded: bool
    was_regenerated: bool
    regeneration_attempts: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    documents_indexed: bool
    ollama_model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Liveness/readiness probe. Checks that a FAISS index is loaded."""
    service = get_query_service()
    ready = service.is_ready()
    return HealthResponse(
        status="ok" if ready else "not_ready",
        documents_indexed=ready,
        ollama_model=settings.ollama_model,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question against the ingested knowledge base.

    Returns a fully-formed, non-streaming answer with grounding metadata.
    """
    service = get_query_service()

    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed. Run `python scripts/ingest.py` first.",
        )

    service.enable_evaluation = request.enable_evaluation
    service.enable_regeneration = request.enable_regeneration

    try:
        result = service.query(
            request.question,
            top_k=request.top_k,
            threshold=request.threshold,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        is_grounded=result.is_grounded,
        was_regenerated=result.was_regenerated,
        regeneration_attempts=result.regeneration_attempts,
        processing_time_ms=result.processing_time_ms,
    )


@app.post("/query/stream", tags=["RAG"])
def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    Stream an answer token-by-token as Server-Sent Events (SSE).

    Event types:
      - `token`: a chunk of generated text
      - `done`: final metadata (sources, groundedness, timing)
    """
    service = get_query_service()

    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed. Run `python scripts/ingest.py` first.",
        )

    service.enable_evaluation = request.enable_evaluation
    service.enable_regeneration = request.enable_regeneration

    def event_generator():
        import json

        for item in service.stream_query(
            request.question, top_k=request.top_k, threshold=request.threshold
        ):
            if isinstance(item, str):
                payload = json.dumps({"token": item})
                yield f"event: token\ndata: {payload}\n\n"
            else:
                payload = json.dumps(
                    {
                        "sources": item.sources,
                        "is_grounded": item.is_grounded,
                        "was_regenerated": item.was_regenerated,
                        "regeneration_attempts": item.regeneration_attempts,
                        "processing_time_ms": item.processing_time_ms,
                    }
                )
                yield f"event: done\ndata: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history", tags=["RAG"])
def get_history(limit: int = 20):
    """Return the most recent queries processed by this server instance."""
    service = get_query_service()
    history = service.get_query_history()
    return {"count": len(history), "queries": history[-limit:]}


@app.delete("/history", tags=["RAG"])
def clear_history():
    """Clear in-memory query history."""
    service = get_query_service()
    service.clear_history()
    return {"status": "cleared"}
