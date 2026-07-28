"""Hybrid retriever for the Hybrid RAG Evaluation Pipeline.

This module implements :class:`HybridRetriever`, which encodes a query with
both a dense embedder and a sparse BM25 encoder concurrently, issues two
Qdrant vector searches in parallel, converts the raw Qdrant results into
typed :class:`~src.models.ScoredChunk` objects, and merges both ranked lists
using Reciprocal Rank Fusion (:func:`~src.retrieval.rrf.rrf_merge`).

The Qdrant client is injected via the constructor so that tests can substitute
a mock without touching any live service.

Example::

    from qdrant_client import QdrantClient
    from src.config.settings import QdrantSettings, RetrievalSettings
    from src.retrieval.dense_embedder import DenseEmbedder
    from src.retrieval.sparse_encoder import SparseEncoder
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        dense_embedder=DenseEmbedder(embedder_settings),
        sparse_encoder=SparseEncoder(sparse_settings),
        qdrant_client=QdrantClient(url=qdrant_settings.url),
        retrieval_settings=RetrievalSettings(),
        qdrant_settings=QdrantSettings(),
    )
    results = await retriever.retrieve("What is RAG?")
"""

from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector

from src.config.settings import QdrantSettings, RetrievalSettings
from src.exceptions import RetrievalError
from src.models import Chunk, ScoredChunk
from src.retrieval.dense_embedder import DenseEmbedder
from src.retrieval.rrf import rrf_merge
from src.retrieval.sparse_encoder import SparseEncoder


class HybridRetriever:
    """Orchestrates hybrid dense+sparse retrieval and RRF fusion.

    Encodes the query with :class:`~src.retrieval.dense_embedder.DenseEmbedder`
    and :class:`~src.retrieval.sparse_encoder.SparseEncoder` concurrently via
    ``asyncio.gather``, then issues two Qdrant searches (one per vector type)
    also concurrently.  The two ranked result lists are merged with
    :func:`~src.retrieval.rrf.rrf_merge` and the top-``retrieval_top_k``
    candidates are returned sorted by descending RRF score.

    Args:
        dense_embedder: Encoder that converts query text into a dense vector.
        sparse_encoder: Encoder that converts query text into a BM25 sparse
            vector.
        qdrant_client: Pre-configured :class:`~qdrant_client.QdrantClient`
            instance.  Injected to support test mocking without live services.
        retrieval_settings: Settings controlling ``dense_top_k``,
            ``sparse_top_k``, ``retrieval_top_k``, and ``rrf_k``.
        qdrant_settings: Settings carrying the ``collection_name`` used for
            every Qdrant query.

    Example::

        retriever = HybridRetriever(
            dense_embedder=embedder,
            sparse_encoder=encoder,
            qdrant_client=client,
            retrieval_settings=RetrievalSettings(),
            qdrant_settings=QdrantSettings(),
        )
        chunks = await retriever.retrieve("Explain hybrid search")
    """

    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_encoder: SparseEncoder,
        qdrant_client: QdrantClient,
        retrieval_settings: RetrievalSettings,
        qdrant_settings: QdrantSettings,
    ) -> None:
        self._dense_embedder = dense_embedder
        self._sparse_encoder = sparse_encoder
        self._client = qdrant_client
        self._retrieval = retrieval_settings
        self._qdrant = qdrant_settings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> list[ScoredChunk]:
        """Retrieve the top candidates for *query* using hybrid search.

        Encoding and Qdrant searches are executed concurrently where possible:

        1. Dense and sparse query encodings are run via ``asyncio.gather``.
        2. Dense Qdrant search and sparse Qdrant search are issued via a
           second ``asyncio.gather``.
        3. Both result lists are merged with :func:`~src.retrieval.rrf.rrf_merge`
           and the top-``retrieval_top_k`` results are returned.

        Args:
            query: Natural-language query string to retrieve chunks for.

        Returns:
            A list of :class:`~src.models.ScoredChunk` objects sorted by
            descending RRF score, containing at most ``retrieval_top_k``
            entries.  If the merged result set has fewer entries than
            ``retrieval_top_k``, all available entries are returned without
            padding.

        Raises:
            RetrievalError: If any Qdrant search call raises an exception.
                The error message identifies whether the failure originated in
                the dense or sparse search path.

        Example::

            results = await retriever.retrieve("What is Reciprocal Rank Fusion?")
            assert all(isinstance(sc, ScoredChunk) for sc in results)
            assert len(results) <= retrieval_settings.retrieval_top_k
        """
        # Step 1: encode query concurrently with both encoders
        dense_vecs, sparse_vecs = await asyncio.gather(
            self._dense_embedder.aencode([query]),
            self._sparse_encoder.aencode([query]),
        )
        dense_vec: list[float] = dense_vecs[0]
        sparse_vec: SparseVector = sparse_vecs[0]

        # Step 2: issue both Qdrant searches concurrently
        dense_results_raw, sparse_results_raw = await asyncio.gather(
            self._search_dense(dense_vec),
            self._search_sparse(sparse_vec),
        )

        # Step 3: convert raw Qdrant hits to ScoredChunk objects
        dense_scored = self._hits_to_scored_chunks(dense_results_raw)
        sparse_scored = self._hits_to_scored_chunks(sparse_results_raw)

        # Step 4: merge and return top-k via RRF
        return rrf_merge(
            dense_results=dense_scored,
            sparse_results=sparse_scored,
            rrf_k=self._retrieval.rrf_k,
            top_k=self._retrieval.retrieval_top_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search_dense(self, dense_vec: list[float]) -> list[Any]:
        """Issue a dense vector search against Qdrant.

        Args:
            dense_vec: The query dense vector (list of floats).

        Returns:
            List of raw Qdrant ``ScoredPoint`` objects.

        Raises:
            RetrievalError: On any Qdrant client exception.
        """
        try:
            return self._client.search(
                collection_name=self._qdrant.collection_name,
                query_vector=("dense", dense_vec),
                limit=self._retrieval.dense_top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Dense Qdrant search failed: {exc}"
            ) from exc

    async def _search_sparse(self, sparse_vec: SparseVector) -> list[Any]:
        """Issue a sparse (BM25) vector search against Qdrant.

        Args:
            sparse_vec: The query sparse vector with ``indices`` and ``values``.

        Returns:
            List of raw Qdrant ``ScoredPoint`` objects.

        Raises:
            RetrievalError: On any Qdrant client exception.
        """
        try:
            return self._client.search(
                collection_name=self._qdrant.collection_name,
                query_vector=("sparse", sparse_vec),
                limit=self._retrieval.sparse_top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Sparse Qdrant search failed: {exc}"
            ) from exc

    @staticmethod
    def _hits_to_scored_chunks(hits: list[Any]) -> list[ScoredChunk]:
        """Convert a list of raw Qdrant ``ScoredPoint`` objects to
        :class:`~src.models.ScoredChunk` instances.

        Each Qdrant point payload is expected to contain the fields
        ``chunk_id``, ``document_id``, ``text``, ``chunk_index``, and
        ``metadata`` as stored during ingestion.

        Args:
            hits: List of ``ScoredPoint`` objects returned by Qdrant's
                ``search`` method.

        Returns:
            A list of :class:`~src.models.ScoredChunk` objects in the same
            order as *hits*, preserving the original Qdrant relevance scores.
        """
        scored_chunks: list[ScoredChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                metadata=payload.get("metadata", {}),
                chunk_index=payload.get("chunk_index", 0),
            )
            scored_chunks.append(ScoredChunk(chunk=chunk, score=hit.score))
        return scored_chunks
