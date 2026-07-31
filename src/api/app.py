"""FastAPI application factory for the Hybrid RAG Evaluation Pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routers.evaluate import router as evaluate_router
from src.api.routers.health import router as health_router
from src.api.routers.ingest import router as ingest_router
from src.api.routers.query import router as query_router
from src.api.schemas import HealthResponse  # noqa: F401
from src.config.settings import AppSettings
from src.evaluation.evaluator import RagasEvaluator
from src.exceptions import PipelineError
from src.generation.generator import LLMGenerator
from src.ingestion.ingestor import IngestionService
from src.pipeline.pipeline import Pipeline
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.dense_embedder import DenseEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_encoder import SparseEncoder


@dataclass(frozen=True)
class ApiComponents:
    qdrant_client: Any
    retriever: HybridRetriever
    reranker: CrossEncoderReranker
    generator: LLMGenerator
    pipeline: Pipeline
    ingestion_service: IngestionService
    evaluator: RagasEvaluator


def create_app(
    settings: AppSettings, components: ApiComponents | None = None
) -> FastAPI:
    """Create a configured FastAPI application.

    Args:
        settings: Application settings.
        components: Optional pre-built components for dependency injection in tests.

    Returns:
        A FastAPI app instance with routers registered and dependencies stored in
        ``app.state``.
    """
    app = FastAPI(title="Hybrid RAG Evaluation Pipeline", version="0.1.0")

    @app.exception_handler(PipelineError)
    async def _pipeline_error_handler(
        request: Request, exc: PipelineError
    ) -> JSONResponse:
        _ = request
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "step": exc.step_name},
        )

    if components is None:
        components = _build_components(settings)

    app.state.settings = settings
    app.state.pipeline = components.pipeline
    app.state.retriever = components.retriever
    app.state.reranker = components.reranker
    app.state.generator = components.generator
    app.state.ingestion_service = components.ingestion_service
    app.state.evaluator = components.evaluator
    app.state.qdrant_client = components.qdrant_client
    app.state.write_lock = components.pipeline.write_lock

    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(evaluate_router)
    app.include_router(health_router)

    return app


def _build_components(settings: AppSettings) -> ApiComponents:
    write_lock = asyncio.Lock()
    dense_embedder = DenseEmbedder(settings.embedder)
    sparse_encoder = SparseEncoder(settings.sparse_encoder)
    reranker = CrossEncoderReranker(settings.reranker)
    generator = LLMGenerator(settings.llm)

    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "QdrantClient is required to build the API in production. "
            "Install 'qdrant-client' or inject components in tests."
        ) from exc

    qdrant_client = QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)
    retriever = HybridRetriever(
        dense_embedder=dense_embedder,
        sparse_encoder=sparse_encoder,
        qdrant_client=qdrant_client,
        retrieval_settings=settings.retrieval,
        qdrant_settings=settings.qdrant,
    )

    pipeline = Pipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        write_lock=write_lock,
    )
    ingestion_service = IngestionService(
        dense_embedder=dense_embedder,
        sparse_encoder=sparse_encoder,
        qdrant_client=qdrant_client,
        ingestion_settings=settings.ingestion,
        embedder_settings=settings.embedder,
        qdrant_settings=settings.qdrant,
    )
    evaluator = RagasEvaluator(pipeline=pipeline, settings=settings.evaluation)

    return ApiComponents(
        qdrant_client=qdrant_client,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        pipeline=pipeline,
        ingestion_service=ingestion_service,
        evaluator=evaluator,
    )
