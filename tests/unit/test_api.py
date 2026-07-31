from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.testclient import TestClient

from src.api.app import ApiComponents, create_app
from src.config.settings import AppSettings
from src.exceptions import EvaluationError, IngestionError, PipelineError
from src.models import (
    Chunk,
    EvaluationReport,
    GenerationResult,
    IngestionResult,
    PipelineResult,
    ScoredChunk,
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        metadata={},
        chunk_index=0,
    )


class DummyRetriever:
    async def retrieve(self, query: str) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=_chunk("c1", f"ctx:{query}"), score=0.9)]


class DummyReranker:
    async def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return candidates


class DummyGenerator:
    async def generate(self, query: str, context: list[str]) -> GenerationResult:
        return GenerationResult(
            answer=f"answer:{query}",
            model_id="dummy",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def stream(self, query: str, context: list[str]) -> AsyncGenerator[str, None]:
        yield f"token:{query}"


class DummyPipeline:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.write_lock = asyncio.Lock()

    async def query(self, query: str, top_k: int | None = None) -> PipelineResult:
        if self.fail_on == "query":
            raise PipelineError("query failed", step_name="generator")
        scored = [ScoredChunk(chunk=_chunk("c1", "ctx"), score=0.9)]
        return PipelineResult(
            answer=f"answer:{query}",
            context_chunks=scored,
            reranked_scores=[0.9],
            latency_ms=1.0,
        )

    async def stream(self, query: str, top_k: int | None = None) -> AsyncGenerator[str, None]:
        if self.fail_on == "stream":
            raise PipelineError("stream failed", step_name="generator")
        yield "token1"
        yield "token2"


class DummyIngestionService:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def ingest(self, documents: list[Any]) -> IngestionResult:
        if self.fail:
            raise IngestionError("ingest failed")
        return IngestionResult(chunks_ingested=len(documents), collection_name="test")


class DummyEvaluator:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def evaluate(self, qa_dataset: list[Any]) -> EvaluationReport:
        if self.fail:
            raise EvaluationError("evaluation failed")
        return EvaluationReport(per_question=[], aggregate={}, errors=[], timestamp="now")


class DummyQdrantClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def get_collections(self) -> Any:
        if self.fail:
            raise RuntimeError("qdrant unavailable")
        return []


def _create_test_client(
    fail_on: str | None = None,
    ingest_fail: bool = False,
    eval_fail: bool = False,
    qdrant_fail: bool = False,
) -> TestClient:
    settings = AppSettings()
    pipeline = DummyPipeline(fail_on=fail_on)
    components = ApiComponents(
        qdrant_client=DummyQdrantClient(fail=qdrant_fail),
        retriever=DummyRetriever(),  # type: ignore[arg-type]
        reranker=DummyReranker(),  # type: ignore[arg-type]
        generator=DummyGenerator(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        ingestion_service=DummyIngestionService(fail=ingest_fail),  # type: ignore[arg-type]
        evaluator=DummyEvaluator(fail=eval_fail),  # type: ignore[arg-type]
    )
    app = create_app(settings, components)
    return TestClient(app)


def test_health_endpoint_ok() -> None:
    client = _create_test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["qdrant_ok"] is True
    assert data["llm_ok"] is True


def test_health_endpoint_fail() -> None:
    client = _create_test_client(qdrant_fail=True)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["qdrant_ok"] is False


def test_query_endpoint_success() -> None:
    client = _create_test_client()
    response = client.post("/query", json={"query": "test query", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "answer:test query"
    assert data["latency_ms"] == 1.0


def test_query_endpoint_failure() -> None:
    client = _create_test_client(fail_on="query")
    response = client.post("/query", json={"query": "test query"})
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["step"] == "generator"


def test_query_stream_endpoint_success() -> None:
    client = _create_test_client()
    response = client.post("/query/stream", json={"query": "test query"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.text == "token1token2"


def test_query_stream_endpoint_failure() -> None:
    client = _create_test_client(fail_on="stream")
    response = client.post("/query/stream", json={"query": "test query"})
    assert response.status_code == 200
    assert "ERROR" in response.text


def test_ingest_endpoint_success() -> None:
    client = _create_test_client()
    response = client.post(
        "/ingest",
        json={"documents": [{"id": "d1", "text": "hello"}, {"id": "d2", "text": "world"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_ingested"] == 2
    assert data["collection_name"] == "test"


def test_ingest_endpoint_failure() -> None:
    client = _create_test_client(ingest_fail=True)
    response = client.post(
        "/ingest",
        json={"documents": [{"id": "d1", "text": "hello"}]},
    )
    assert response.status_code == 500
    assert "ingest failed" in response.json()["detail"]


def test_evaluate_endpoint_success() -> None:
    client = _create_test_client()
    response = client.post(
        "/evaluate",
        json={"qa_dataset": [{"question": "q", "ground_truth": "gt"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["timestamp"] == "now"


def test_evaluate_endpoint_failure() -> None:
    client = _create_test_client(eval_fail=True)
    response = client.post(
        "/evaluate",
        json={"qa_dataset": [{"question": "q", "ground_truth": "gt"}]},
    )
    assert response.status_code == 500
    assert "evaluation failed" in response.json()["detail"]
