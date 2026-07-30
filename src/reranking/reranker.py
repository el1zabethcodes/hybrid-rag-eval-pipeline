"""Cross-encoder reranking component for the Hybrid RAG Evaluation Pipeline.

This module provides :class:`~src.reranking.reranker.CrossEncoderReranker`,
which wraps a ``sentence_transformers.CrossEncoder`` model to rerank candidate
chunks returned by the retriever. The synchronous scoring call is dispatched
to a thread pool for the async API to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

from sentence_transformers import CrossEncoder

from src.config.settings import RerankerSettings
from src.exceptions import RerankerError
from src.models import ScoredChunk


class CrossEncoderReranker:
    """Rerank retrieval candidates using a cross-encoder model.

    The reranker scores each candidate chunk by running a cross-encoder on the
    pair ``(query, chunk_text)``. Candidates are then sorted by descending
    cross-encoder score and truncated to ``RerankerSettings.rerank_top_k``.

    Args:
        settings: Reranker configuration.
        executor: Optional thread-pool executor used for the async method.
            When ``None`` (the default), a private executor is created.
        batch_size: Batch size passed to ``CrossEncoder.predict``.

    Raises:
        RerankerError: If the model fails to load.
    """

    def __init__(
        self,
        settings: RerankerSettings,
        executor: Optional[ThreadPoolExecutor] = None,
        batch_size: int = 32,
    ) -> None:
        self._settings = settings
        self._batch_size = batch_size
        try:
            self._model = CrossEncoder(settings.model_name)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Failed to load cross-encoder model '{settings.model_name}': {exc}"
            ) from exc
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="reranker",
        )

    def rerank_sync(self, query: str, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        """Rerank candidates synchronously.

        Args:
            query: User query.
            candidates: Retrieval candidates to rerank.

        Returns:
            A reranked and truncated list of candidates.

        Raises:
            RerankerError: If the underlying model inference fails.
        """
        if not candidates or self._settings.rerank_top_k <= 0:
            return []

        pairs: List[Tuple[str, str]] = [(query, c.chunk.text) for c in candidates]
        try:
            scores_raw = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Cross-encoder reranking failed: {exc}") from exc

        scores: Sequence[float]
        if hasattr(scores_raw, "tolist"):
            scores = scores_raw.tolist()
        else:
            scores = list(scores_raw)

        reranked = [
            ScoredChunk(chunk=candidates[i].chunk, score=float(scores[i]))
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda sc: (-sc.score, sc.chunk.id))
        return reranked[: min(self._settings.rerank_top_k, len(reranked))]

    async def rerank(self, query: str, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        """Rerank candidates asynchronously without blocking the event loop.

        Args:
            query: User query.
            candidates: Retrieval candidates to rerank.

        Returns:
            A reranked and truncated list of candidates.

        Raises:
            RerankerError: If the underlying model inference fails.
        """
        loop = asyncio.get_event_loop()
        fn = functools.partial(self.rerank_sync, query, candidates)
        result: List[ScoredChunk] = await loop.run_in_executor(self._executor, fn)
        return result

