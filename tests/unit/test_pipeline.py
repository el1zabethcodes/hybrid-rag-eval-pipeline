from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from src.exceptions import PipelineError
from src.models import Chunk, GenerationResult, ScoredChunk
from src.pipeline.pipeline import Pipeline


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        metadata={},
        chunk_index=0,
    )


@dataclass
class DummyRetriever:
    started: asyncio.Event
    block: asyncio.Event

    async def retrieve(self, query: str) -> list[ScoredChunk]:
        _ = query
        self.started.set()
        await self.block.wait()
        return [ScoredChunk(chunk=_chunk("c1", "ctx"), score=0.1)]


@dataclass
class DummyReranker:
    async def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        _ = query
        return [ScoredChunk(chunk=candidates[0].chunk, score=0.9)]


@dataclass
class DummyGenerator:
    async def generate(self, query: str, context: list[str]) -> GenerationResult:
        _ = (query, context)
        return GenerationResult(
            answer="answer",
            model_id="dummy",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def stream(self, query: str, context: list[str]) -> AsyncGenerator[str, None]:
        _ = (query, context)
        yield "token1"
        yield "token2"


@pytest.mark.asyncio
async def test_pipeline_happy_path() -> None:
    lock = asyncio.Lock()
    retriever = DummyRetriever(started=asyncio.Event(), block=asyncio.Event())
    pipeline = Pipeline(
        retriever=retriever,
        reranker=DummyReranker(),
        generator=DummyGenerator(),
        write_lock=lock,
    )

    retriever.block.set()
    result = await pipeline.query("q")

    assert result.answer == "answer"
    assert [c.chunk.id for c in result.context_chunks] == ["c1"]
    assert result.reranked_scores == [0.9]
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_pipeline_stream_happy_path() -> None:
    retriever = DummyRetriever(started=asyncio.Event(), block=asyncio.Event())
    pipeline = Pipeline(
        retriever=retriever,
        reranker=DummyReranker(),
        generator=DummyGenerator(),
    )

    retriever.block.set()
    tokens = []
    async for token in pipeline.stream("q"):
        tokens.append(token)

    assert tokens == ["token1", "token2"]


@pytest.mark.asyncio
async def test_pipeline_wraps_step_errors() -> None:
    class FailingRetriever:
        async def retrieve(self, query: str) -> list[ScoredChunk]:
            raise RuntimeError(query)

    pipeline = Pipeline(
        retriever=FailingRetriever(),
        reranker=DummyReranker(),
        generator=DummyGenerator(),
    )

    with pytest.raises(PipelineError) as excinfo:
        await pipeline.query("q")

    assert excinfo.value.step_name == "retriever"
    assert isinstance(excinfo.value.cause, RuntimeError)


@pytest.mark.asyncio
async def test_pipeline_holds_write_lock_during_query() -> None:
    lock = asyncio.Lock()
    started = asyncio.Event()
    block = asyncio.Event()
    retriever = DummyRetriever(started=started, block=block)
    pipeline = Pipeline(
        retriever=retriever,
        reranker=DummyReranker(),
        generator=DummyGenerator(),
        write_lock=lock,
    )

    task = asyncio.create_task(pipeline.query("q"))
    await started.wait()

    assert lock.locked() is True

    acquire_attempt = asyncio.create_task(lock.acquire())
    await asyncio.sleep(0)
    assert acquire_attempt.done() is False

    block.set()
    await task
    await acquire_attempt
    lock.release()
