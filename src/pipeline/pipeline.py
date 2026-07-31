"""Pipeline orchestrator for the Hybrid RAG Evaluation Pipeline.

This module implements :class:`~src.pipeline.pipeline.Pipeline`, which runs
retrieval, reranking, and answer generation in sequence and wraps step errors
into :class:`~src.exceptions.PipelineError`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Protocol

from src.exceptions import PipelineError
from src.models import GenerationResult, PipelineResult, ScoredChunk


class Retriever(Protocol):
    async def retrieve(self, query: str) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]: ...


class Generator(Protocol):
    async def generate(self, query: str, context: list[str]) -> GenerationResult: ...

    def stream(self, query: str, context: list[str]) -> AsyncGenerator[str, None]: ...


class Pipeline:
    """Orchestrates Retriever → Reranker → Generator steps for a single query.

    The pipeline uses an ``asyncio.Lock`` to guard against concurrent writes
    (document ingestion) during active queries within a single process.

    Args:
        retriever: Configured hybrid retriever.
        reranker: Configured cross-encoder reranker.
        generator: Configured LLM generator.
        write_lock: Optional shared lock. If omitted, a private lock is created.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._write_lock = write_lock or asyncio.Lock()

    @property
    def write_lock(self) -> asyncio.Lock:
        """Lock shared with writers (e.g. ingestion) to avoid concurrent writes."""
        return self._write_lock

    async def query(self, query: str, top_k: int | None = None) -> PipelineResult:
        """Run the full pipeline for a single query.

        Args:
            query: User query text.
            top_k: Optional cap on the number of context chunks passed to the LLM.
                When provided, this is applied after reranking.

        Returns:
            A :class:`~src.models.PipelineResult` containing the answer, context
            chunks, reranked scores, and end-to-end latency.

        Raises:
            PipelineError: If any pipeline step fails. The ``step_name`` is one of
                ``"retriever"``, ``"reranker"``, ``"generator"``.
        """
        start = time.perf_counter()
        async with self._write_lock:
            candidates: list[ScoredChunk]
            try:
                candidates = await self._retriever.retrieve(query)
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(
                    message=f"Retrieval step failed: {exc}",
                    step_name="retriever",
                    cause=exc,
                ) from exc

            ranked: list[ScoredChunk]
            try:
                ranked = await self._reranker.rerank(query, candidates)
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(
                    message=f"Reranking step failed: {exc}",
                    step_name="reranker",
                    cause=exc,
                ) from exc

            if top_k is not None:
                k = max(0, int(top_k))
                ranked = ranked[:k]

            try:
                generation = await self._generator.generate(
                    query=query,
                    context=[sc.chunk.text for sc in ranked],
                )
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(
                    message=f"Generation step failed: {exc}",
                    step_name="generator",
                    cause=exc,
                ) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        return PipelineResult(
            answer=generation.answer,
            context_chunks=ranked,
            reranked_scores=[sc.score for sc in ranked],
            latency_ms=latency_ms,
        )

    async def stream(
        self, query: str, top_k: int | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream the full pipeline for a single query.

        Args:
            query: User query text.
            top_k: Optional cap on the number of context chunks passed to the LLM.
                When provided, this is applied after reranking.

        Yields:
            Token/chunk strings as produced by the LLM backend.

        Raises:
            PipelineError: If any pipeline step fails. The ``step_name`` is one of
                ``"retriever"``, ``"reranker"``, ``"generator"``.
        """
        async with self._write_lock:
            candidates: list[ScoredChunk]
            try:
                candidates = await self._retriever.retrieve(query)
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(
                    message=f"Retrieval step failed: {exc}",
                    step_name="retriever",
                    cause=exc,
                ) from exc

            ranked: list[ScoredChunk]
            try:
                ranked = await self._reranker.rerank(query, candidates)
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(
                    message=f"Reranking step failed: {exc}",
                    step_name="reranker",
                    cause=exc,
                ) from exc

            if top_k is not None:
                k = max(0, int(top_k))
                ranked = ranked[:k]

            try:
                async for token in self._generator.stream(
                    query=query,
                    context=[sc.chunk.text for sc in ranked],
                ):
                    yield token
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(
                    message=f"Generation step failed: {exc}",
                    step_name="generator",
                    cause=exc,
                ) from exc
