from __future__ import annotations

from typing import List, Tuple

import pytest

from src.config.settings import RerankerSettings
from src.models import Chunk, ScoredChunk
from src.reranking.reranker import CrossEncoderReranker


class DummyCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[float]:
        _ = (batch_size, show_progress_bar)
        return [float(len(text)) for _, text in pairs]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_1",
        text=text,
        metadata={},
        chunk_index=0,
    )


def test_rerank_sync_sorts_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.reranking.reranker.CrossEncoder", DummyCrossEncoder)
    settings = RerankerSettings(model_name="dummy", rerank_top_k=2)
    reranker = CrossEncoderReranker(settings)

    candidates = [
        ScoredChunk(chunk=_chunk("c1", "aaa"), score=0.1),
        ScoredChunk(chunk=_chunk("c2", "bbbbbbbb"), score=0.2),
        ScoredChunk(chunk=_chunk("c3", "ccccc"), score=0.3),
    ]

    reranked = reranker.rerank_sync("query", candidates)

    assert len(reranked) == 2
    assert [c.chunk.id for c in reranked] == ["c2", "c3"]
    assert reranked[0].score >= reranked[1].score


def test_rerank_sync_tie_breaks_by_chunk_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.reranking.reranker.CrossEncoder", DummyCrossEncoder)
    settings = RerankerSettings(model_name="dummy", rerank_top_k=10)
    reranker = CrossEncoderReranker(settings)

    a = ScoredChunk(chunk=_chunk("a", "same"), score=0.0)
    b = ScoredChunk(chunk=_chunk("b", "same"), score=0.0)

    order1 = reranker.rerank_sync("query", [b, a])
    order2 = reranker.rerank_sync("query", [a, b])

    assert [c.chunk.id for c in order1] == ["a", "b"]
    assert [c.chunk.id for c in order2] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_async_matches_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.reranking.reranker.CrossEncoder", DummyCrossEncoder)
    settings = RerankerSettings(model_name="dummy", rerank_top_k=3)
    reranker = CrossEncoderReranker(settings)

    candidates = [
        ScoredChunk(chunk=_chunk("c1", "aaa"), score=0.1),
        ScoredChunk(chunk=_chunk("c2", "bbbb"), score=0.2),
    ]

    sync_result = reranker.rerank_sync("query", candidates)
    async_result = await reranker.rerank("query", candidates)

    assert sync_result == async_result

