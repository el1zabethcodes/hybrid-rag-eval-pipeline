"""Reciprocal Rank Fusion (RRF) merger for hybrid retrieval results.

This module exposes a single pure function, :func:`rrf_merge`, which combines
dense and sparse retrieval result lists into a single, deduplicated ranking
using the RRF algorithm described in:

    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank
    fusion outperforms condorcet and individual rank learning methods. SIGIR.

The function is stateless and has no I/O side-effects, making it trivially
testable and safe to call from any concurrency context.
"""

from __future__ import annotations

from src.models import ScoredChunk


def rrf_merge(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    rrf_k: int,
    top_k: int,
) -> list[ScoredChunk]:
    """Merge dense and sparse retrieval results using Reciprocal Rank Fusion.

    Each input list is treated as an independent ranking.  For every chunk
    that appears in a list, a per-list contribution of ``1 / (rrf_k + rank)``
    is added to that chunk's cumulative RRF score, where ``rank`` is
    1-indexed.  Chunks that appear in both lists receive the sum of both
    contributions (deduplication by ``chunk.id``).  The merged list is then
    sorted in descending score order and truncated to ``top_k`` entries.

    This function is **pure**: it does not perform I/O, mutate its arguments,
    or rely on any external state.

    Args:
        dense_results: Ordered list of :class:`~src.models.ScoredChunk` objects
            returned by the dense (vector) search, sorted by descending
            similarity score.  The original scores are ignored — only the
            list order (rank) is used.
        sparse_results: Ordered list of :class:`~src.models.ScoredChunk`
            objects returned by the sparse (BM25) search, sorted by descending
            BM25 score.  The original scores are ignored — only the list order
            (rank) is used.
        rrf_k: RRF smoothing constant (typically 60).  A larger value
            reduces the influence of highly-ranked documents relative to
            lower-ranked ones.  Must be a positive integer.
        top_k: Maximum number of results to return.  If the merged set
            contains fewer than ``top_k`` unique chunks, all chunks are
            returned.

    Returns:
        A list of :class:`~src.models.ScoredChunk` objects where each
        ``score`` field holds the computed RRF score for that chunk.  The
        list is sorted by descending RRF score and contains at most ``top_k``
        entries.  Every chunk ID that appeared in either input list appears
        exactly once in the output.

    Example:
        >>> from src.models import Chunk, ScoredChunk
        >>> def make(chunk_id: str, score: float = 0.0) -> ScoredChunk:
        ...     chunk = Chunk(id=chunk_id, document_id="doc", text="t", chunk_index=0)
        ...     return ScoredChunk(chunk=chunk, score=score)
        >>> dense = [make("a"), make("b"), make("c")]
        >>> sparse = [make("b"), make("d"), make("a")]
        >>> results = rrf_merge(dense, sparse, rrf_k=60, top_k=4)
        >>> [r.chunk.id for r in results]
        ['b', 'a', 'c', 'd']
    """
    # Map from chunk.id → accumulated RRF score
    scores: dict[str, float] = {}
    # Preserve one ScoredChunk object per unique chunk id (first-seen wins for
    # the Chunk payload; only the score field is replaced with the RRF score)
    chunks: dict[str, ScoredChunk] = {}

    for result_list in (dense_results, sparse_results):
        for rank, scored_chunk in enumerate(result_list, start=1):
            chunk_id = scored_chunk.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if chunk_id not in chunks:
                chunks[chunk_id] = scored_chunk

    # Build merged list: replace original score with computed RRF score
    merged: list[ScoredChunk] = [
        ScoredChunk(chunk=chunks[chunk_id], score=rrf_score)
        for chunk_id, rrf_score in scores.items()
    ]

    # Sort descending by RRF score, then truncate to top_k
    merged.sort(key=lambda sc: sc.score, reverse=True)
    return merged[:top_k]
