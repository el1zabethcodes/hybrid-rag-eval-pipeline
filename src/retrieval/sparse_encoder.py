"""Sparse encoding component for the Hybrid RAG Evaluation Pipeline.

This module wraps ``fastembed.SparseTextEmbedding`` with the ``"Qdrant/bm25"``
model to produce BM25 sparse vectors for document chunks and queries.  The
model tokenises text and computes term weights on the fly using its bundled
vocabulary and IDF statistics — no separate corpus-level index or training
step is required.

Both a synchronous and an asynchronous encoding interface are provided; the
async variant delegates to a thread-pool executor so it never blocks the
event loop.

Example::

    from src.config.settings import SparseEncoderSettings
    from src.retrieval.sparse_encoder import SparseEncoder

    settings = SparseEncoderSettings()
    encoder = SparseEncoder(settings)

    # Synchronous
    sparse_vectors = encoder.encode(["Hello world", "Another text"])

    # Asynchronous (inside an async context)
    sparse_vectors = await encoder.aencode(["Hello world", "Another text"])
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector

from src.config.settings import SparseEncoderSettings


class SparseEncoder:
    """Wraps a ``fastembed.SparseTextEmbedding`` BM25 model to encode texts into
    sparse vectors.

    Each ``SparseVector`` contains parallel ``indices`` (vocabulary term IDs)
    and ``values`` (BM25 term weights) arrays.  The underlying model artefact
    is downloaded once on first use via ``fastembed`` and cached locally; no
    additional installation step is needed beyond ``pip install fastembed``.

    Args:
        settings: A :class:`~src.config.settings.SparseEncoderSettings`
            instance that specifies the ``fastembed`` model name.
        executor: Optional :class:`~concurrent.futures.ThreadPoolExecutor` to
            use for the async path.  When ``None`` (the default) a private
            executor is created and managed by this instance.

    Example::

        settings = SparseEncoderSettings(model_name="Qdrant/bm25")
        encoder = SparseEncoder(settings)
        vectors = encoder.encode(["The quick brown fox"])
        assert len(vectors) == 1
        assert len(vectors[0].indices) > 0
        assert len(vectors[0].values) == len(vectors[0].indices)
    """

    def __init__(
        self,
        settings: SparseEncoderSettings,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self._settings = settings
        self._model = SparseTextEmbedding(model_name=settings.model_name)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sparse_encoder",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> List[SparseVector]:
        """Encode a list of texts into BM25 sparse vectors synchronously.

        Each text is tokenised and its term weights are computed on the fly
        by the ``fastembed`` BM25 model.  The result is a list of
        :class:`~qdrant_client.http.models.SparseVector` objects, one per
        input text.

        Args:
            texts: List of strings to encode.  An empty list returns an
                empty result without invoking the model.

        Returns:
            A list of :class:`~qdrant_client.http.models.SparseVector`
            objects, one per input text.  Each ``SparseVector`` has
            ``indices`` (vocabulary term IDs) and ``values`` (BM25 term
            weights) of equal length.

        Example::

            vectors = encoder.encode(["The quick brown fox"])
            assert len(vectors) == 1
            assert len(vectors[0].indices) > 0
        """
        if not texts:
            return []

        embeddings = self._model.embed(texts)
        result: List[SparseVector] = []
        for emb in embeddings:
            result.append(
                SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
            )
        return result

    async def aencode(self, texts: List[str]) -> List[SparseVector]:
        """Encode a list of texts into BM25 sparse vectors asynchronously.

        The underlying model inference is dispatched to the thread-pool
        executor so that the calling event loop is not blocked during
        tokenisation and weight computation.

        Args:
            texts: List of strings to encode.  An empty list returns an
                empty result without invoking the model.

        Returns:
            A list of :class:`~qdrant_client.http.models.SparseVector`
            objects, one per input text.  Each ``SparseVector`` has
            ``indices`` (vocabulary term IDs) and ``values`` (BM25 term
            weights) of equal length.

        Example::

            import asyncio
            vectors = asyncio.run(encoder.aencode(["The quick brown fox"]))
            assert len(vectors[0].indices) > 0
        """
        loop = asyncio.get_event_loop()
        fn = functools.partial(self.encode, texts)
        result: List[SparseVector] = await loop.run_in_executor(self._executor, fn)
        return result
