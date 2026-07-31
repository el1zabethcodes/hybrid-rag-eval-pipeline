"""Pydantic schemas for the FastAPI surface of the Hybrid RAG Evaluation Pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.models import Document, EvaluationReport, QAPair, ScoredChunk


class QueryRequest(BaseModel):
    """Request payload for the `/query` endpoint."""

    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=0)


class QueryResponse(BaseModel):
    """Response payload for the non-streaming `/query` endpoint."""

    answer: str
    context_chunks: list[ScoredChunk]
    reranked_scores: list[float]
    latency_ms: float


class IngestRequest(BaseModel):
    """Request payload for the `/ingest` endpoint."""

    documents: list[Document] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Response payload for the `/ingest` endpoint."""

    chunks_ingested: int
    collection_name: str


class EvaluateRequest(BaseModel):
    """Request payload for the `/evaluate` endpoint."""

    qa_dataset: list[QAPair] = Field(default_factory=list)


class EvaluateResponse(BaseModel):
    """Response payload for the `/evaluate` endpoint."""

    report: EvaluationReport


class HealthResponse(BaseModel):
    """Response payload for the `/health` endpoint."""

    ok: bool
    qdrant_ok: bool
    llm_ok: bool
    details: dict[str, Any] = Field(default_factory=dict)
