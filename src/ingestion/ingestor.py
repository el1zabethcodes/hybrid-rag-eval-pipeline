"""Document ingestion service for the Hybrid RAG Evaluation Pipeline.

This module implements :class:`IngestionService`, which accepts raw
:class:`~src.models.Document` objects, splits them into overlapping token
chunks, encodes each chunk with both a dense and a sparse encoder in parallel,
and uploads the resulting vectors and metadata to a Qdrant collection.

The Qdrant collection is created automatically (with the correct dual named-vector
configuration) if it does not already exist.  All Qdrant write failures are raised
as :exc:`~src.exceptions.IngestionError` — but only *after* all chunking and
encoding has completed, so callers know that no partial data has been persisted.

Example::

    from src.config.settings import AppSettings
    from src.retrieval.dense_embedder import DenseEmbedder
    from src.retrieval.sparse_encoder import SparseEncoder
    from qdrant_client import QdrantClient

    settings = AppSettings()
    embedder = DenseEmbedder(settings.embedder)
    encoder = SparseEncoder(settings.sparse_encoder)
    client = QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)

    service = IngestionService(
        dense_embedder=embedder,
        sparse_encoder=encoder,
        qdrant_client=client,
        ingestion_settings=settings.ingestion,
        embedder_settings=settings.embedder,
        qdrant_settings=settings.qdrant,
    )

    documents = [Document(id="doc1", text="Hello world")]
    result = await service.ingest(documents)
    print(result.chunks_ingested)  # 1
"""

from __future__ import annotations

import asyncio
import uuid
from typing import List

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from src.config.settings import EmbedderSettings, IngestionSettings, QdrantSettings
from src.exceptions import IngestionError
from src.models import Chunk, Document, IngestionResult
from src.retrieval.dense_embedder import DenseEmbedder
from src.retrieval.sparse_encoder import SparseEncoder


class IngestionService:
    """Chunks, encodes, and stores documents in a Qdrant hybrid-vector collection.

    The service performs the following steps when :meth:`ingest` is called:

    1. Each :class:`~src.models.Document` is split into overlapping token-based
       chunks using a sliding-window algorithm (``chunk_size`` / ``chunk_overlap``
       from :class:`~src.config.settings.IngestionSettings`).
    2. Dense and sparse vectors for all chunks are computed in parallel via
       :func:`asyncio.gather`.
    3. If the target Qdrant collection does not yet exist it is created with the
       correct dual named-vector configuration (``"dense"`` + ``"sparse"``).
    4. Chunks are uploaded to Qdrant in batches of ``ingestion_batch_size``.
       Any Qdrant connectivity or write error at this stage raises
       :exc:`~src.exceptions.IngestionError`.

    Args:
        dense_embedder: Configured :class:`~src.retrieval.dense_embedder.DenseEmbedder`
            instance used to produce dense vectors.
        sparse_encoder: Configured :class:`~src.retrieval.sparse_encoder.SparseEncoder`
            instance used to produce BM25 sparse vectors.
        qdrant_client: An initialised :class:`~qdrant_client.QdrantClient` connected
            to the target Qdrant service.
        ingestion_settings: :class:`~src.config.settings.IngestionSettings` carrying
            ``chunk_size``, ``chunk_overlap``, and ``ingestion_batch_size``.
        embedder_settings: :class:`~src.config.settings.EmbedderSettings` carrying
            ``vector_dim`` used when creating the Qdrant collection.
        qdrant_settings: :class:`~src.config.settings.QdrantSettings` carrying
            ``collection_name`` to identify the target collection.

    Example::

        service = IngestionService(
            dense_embedder=embedder,
            sparse_encoder=encoder,
            qdrant_client=client,
            ingestion_settings=IngestionSettings(),
            embedder_settings=EmbedderSettings(),
            qdrant_settings=QdrantSettings(),
        )
        result = await service.ingest([Document(id="doc1", text="Hello world")])
        assert result.chunks_ingested >= 1
    """

    # tiktoken encoding used for all tokenisation; cl100k_base is broadly compatible
    # with most OpenAI and sentence-transformer vocabularies and is the same encoder
    # used by the LLM generator for context-token counting.
    _TIKTOKEN_ENCODING = "cl100k_base"

    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_encoder: SparseEncoder,
        qdrant_client: QdrantClient,
        ingestion_settings: IngestionSettings,
        embedder_settings: EmbedderSettings,
        qdrant_settings: QdrantSettings,
    ) -> None:
        self._dense_embedder = dense_embedder
        self._sparse_encoder = sparse_encoder
        self._qdrant_client = qdrant_client
        self._ingestion_settings = ingestion_settings
        self._embedder_settings = embedder_settings
        self._qdrant_settings = qdrant_settings
        self._tokenizer = tiktoken.get_encoding(self._TIKTOKEN_ENCODING)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def ingest(self, documents: List[Document]) -> IngestionResult:
        """Chunk, encode, and store *documents* in the Qdrant collection.

        Steps performed:

        1. Every document is tokenised and split into sliding-window chunks.
        2. Dense and sparse vectors for all chunks are produced concurrently.
        3. The Qdrant collection is created if it does not already exist.
        4. Chunks are upserted in sequential batches of ``ingestion_batch_size``.

        Args:
            documents: List of :class:`~src.models.Document` objects to ingest.
                An empty list is a no-op that returns
                ``IngestionResult(chunks_ingested=0, ...)``.

        Returns:
            :class:`~src.models.IngestionResult` with the total number of chunks
            stored and the name of the target collection.

        Raises:
            :exc:`~src.exceptions.IngestionError`: If Qdrant is unreachable or the
                upsert operation fails.  This is raised *only* after all encoding
                steps have completed successfully.

        Example::

            result = await service.ingest([Document(id="d1", text="Hello world.")])
            print(result.chunks_ingested)
            print(result.collection_name)
        """
        # Step 1 — chunk all documents
        chunks: List[Chunk] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))

        if not chunks:
            return IngestionResult(
                chunks_ingested=0,
                collection_name=self._qdrant_settings.collection_name,
            )

        chunk_texts = [c.text for c in chunks]

        # Step 2 — encode dense and sparse vectors in parallel
        dense_vectors, sparse_vectors = await asyncio.gather(
            self._dense_embedder.aencode(chunk_texts),
            self._sparse_encoder.aencode(chunk_texts),
        )

        # Step 3 — ensure the collection exists (Qdrant write, but collection-level)
        self._ensure_collection()

        # Step 4 — upload in batches (raises IngestionError on failure)
        batch_size = self._ingestion_settings.ingestion_batch_size
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = batch_start + batch_size
            batch_chunks = chunks[batch_start:batch_end]
            batch_dense = dense_vectors[batch_start:batch_end]
            batch_sparse = sparse_vectors[batch_start:batch_end]
            self._upload_batch(batch_chunks, batch_dense, batch_sparse)

        return IngestionResult(
            chunks_ingested=len(chunks),
            collection_name=self._qdrant_settings.collection_name,
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single document into overlapping token-window chunks.

        The text is first tokenised using the ``cl100k_base`` tiktoken encoding.
        A sliding window of ``chunk_size`` tokens advances by
        ``(chunk_size - chunk_overlap)`` tokens on each step, so consecutive
        chunks share exactly ``chunk_overlap`` tokens at their boundary (except
        for the final chunk which may be shorter than ``chunk_size``).

        Args:
            document: The :class:`~src.models.Document` to split.

        Returns:
            Ordered list of :class:`~src.models.Chunk` objects derived from
            *document*.  Returns an empty list if *document.text* is empty.

        Example::

            doc = Document(id="d1", text="The quick brown fox jumped.")
            chunks = service._chunk_document(doc)
            assert all(c.document_id == "d1" for c in chunks)
        """
        text = document.text
        if not text:
            return []

        chunk_size = self._ingestion_settings.chunk_size
        chunk_overlap = self._ingestion_settings.chunk_overlap
        stride = chunk_size - chunk_overlap

        # Clamp overlap to avoid infinite loop or empty strides
        if stride <= 0:
            stride = 1

        token_ids = self._tokenizer.encode(text)
        if not token_ids:
            return []

        chunks: List[Chunk] = []
        chunk_index = 0
        start = 0

        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            window_ids = token_ids[start:end]
            chunk_text = self._tokenizer.decode(window_ids)

            chunk_id = f"{document.id}_{chunk_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata=dict(document.metadata),
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

            # If we reached the end, stop
            if end == len(token_ids):
                break

            start += stride

        return chunks

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist.

        The collection is configured with two named vector spaces:

        * ``"dense"``: cosine-distance dense vectors of size
          ``EmbedderSettings.vector_dim``.
        * ``"sparse"``: BM25 sparse vectors with IDF modifier.

        This method is idempotent — if the collection already exists no action
        is taken.

        Raises:
            :exc:`~src.exceptions.IngestionError`: If the collection-existence
                check or creation call fails.
        """
        collection_name = self._qdrant_settings.collection_name
        try:
            existing = {
                c.name
                for c in self._qdrant_client.get_collections().collections
            }
        except Exception as exc:
            raise IngestionError(
                f"Failed to retrieve Qdrant collections while checking for "
                f"'{collection_name}': {exc}"
            ) from exc

        if collection_name in existing:
            return

        try:
            self._qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self._embedder_settings.vector_dim,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=Modifier.IDF,
                    ),
                },
            )
        except Exception as exc:
            raise IngestionError(
                f"Failed to create Qdrant collection '{collection_name}': {exc}"
            ) from exc

    def _upload_batch(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[SparseVector],
    ) -> None:
        """Upsert a single batch of chunks into Qdrant.

        Each Qdrant point contains:

        * Named dense vector ``"dense"`` — the full-dimension float list.
        * Named sparse vector ``"sparse"`` — indices and values from BM25.
        * Payload fields ``chunk_id``, ``document_id``, ``text``,
          ``chunk_index``, and ``metadata``.

        Args:
            chunks: :class:`~src.models.Chunk` objects in this batch.
            dense_vectors: Dense vectors parallel to *chunks*.
            sparse_vectors: Sparse vectors parallel to *chunks*.

        Raises:
            :exc:`~src.exceptions.IngestionError`: If the upsert call to Qdrant
                fails for any reason (network error, server-side error, etc.).
        """
        collection_name = self._qdrant_settings.collection_name
        points: List[PointStruct] = []

        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            point = PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id)),
                vector={
                    "dense": dense_vec,
                    "sparse": rest.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                },
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)

        try:
            self._qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise IngestionError(
                f"Failed to upsert batch of {len(points)} chunks into Qdrant "
                f"collection '{collection_name}': {exc}"
            ) from exc
