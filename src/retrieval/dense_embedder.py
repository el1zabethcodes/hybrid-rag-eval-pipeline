"""Dense embedding component for the Hybrid RAG Evaluation Pipeline.

This module wraps ``sentence_transformers.SentenceTransformer`` to produce
fixed-dimension dense vectors for document chunks and queries. Both a
synchronous and an asynchronous encoding interface are provided; the async
variant delegates to a thread-pool executor so it never blocks the event
loop.

Example::

    from src.config.settings import EmbedderSettings
    from src.retrieval.dense_embedder import DenseEmbedder

    settings = EmbedderSettings()
    embedder = DenseEmbedder(settings)

    # Synchronous
    vectors = embedder.encode(["Hello world", "Another text"])

    # Asynchronous (inside an async context)
    vectors = await embedder.aencode(["Hello world", "Another text"])
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer

from src.config.settings import EmbedderSettings


class DenseEmbedder:
    """Wraps a ``SentenceTransformer`` model to encode texts into dense vectors.

    The output dimension of every call is validated against
    ``EmbedderSettings.vector_dim`` so mismatches between the configured
    dimension and the actual model output are caught immediately rather than
    silently propagating incorrect vectors into the pipeline.

    Args:
        settings: An :class:`~src.config.settings.EmbedderSettings` instance
            that specifies the model name, expected vector dimension, and
            batch size.
        executor: Optional :class:`~concurrent.futures.ThreadPoolExecutor` to
            use for the async path.  When ``None`` (the default) a private
            executor is created and managed by this instance.

    Raises:
        ValueError: If any encoded vector has a length that does not equal
            ``settings.vector_dim``.

    Example::

        settings = EmbedderSettings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            vector_dim=384,
            batch_size=64,
        )
        embedder = DenseEmbedder(settings)
        vectors = embedder.encode(["Hello, world!"])
        assert len(vectors[0]) == 384
    """

    def __init__(
        self,
        settings: EmbedderSettings,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._settings = settings
        self._model = SentenceTransformer(settings.model_name)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dense_embedder",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into dense vectors synchronously.

        Texts are encoded in a single batched forward pass using
        ``batch_size`` from the configured :class:`~src.config.settings.EmbedderSettings`.
        Each returned vector has exactly ``vector_dim`` dimensions.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            A list of dense vectors, one per input text.  Each inner list
            has exactly ``EmbedderSettings.vector_dim`` elements.

        Raises:
            ValueError: If any output vector length does not match
                ``EmbedderSettings.vector_dim``.

        Example::

            vectors = embedder.encode(["The quick brown fox"])
            assert len(vectors) == 1
            assert len(vectors[0]) == embedder._settings.vector_dim
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = [emb.tolist() for emb in embeddings]
        self._validate_dimensions(result)
        return result

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into dense vectors asynchronously.

        The underlying model inference is dispatched to the thread-pool
        executor so that the calling event loop is not blocked during the
        forward pass. Output validation is performed before returning.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            A list of dense vectors, one per input text.  Each inner list
            has exactly ``EmbedderSettings.vector_dim`` elements.

        Raises:
            ValueError: If any output vector length does not match
                ``EmbedderSettings.vector_dim``.

        Example::

            import asyncio
            vectors = asyncio.run(embedder.aencode(["The quick brown fox"]))
            assert len(vectors[0]) == embedder._settings.vector_dim
        """
        loop = asyncio.get_event_loop()
        fn = functools.partial(self.encode, texts)
        result: list[list[float]] = await loop.run_in_executor(self._executor, fn)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        """Validate that every vector in *vectors* has the expected length.

        Args:
            vectors: List of encoded vectors to validate.

        Raises:
            ValueError: If any vector's length differs from
                ``EmbedderSettings.vector_dim``.
        """
        expected = self._settings.vector_dim
        for i, vec in enumerate(vectors):
            actual = len(vec)
            if actual != expected:
                raise ValueError(
                    f"Dense embedding vector at index {i} has length {actual} "
                    f"but EmbedderSettings.vector_dim is {expected}. "
                    f"Ensure the configured model_name "
                    f"'{self._settings.model_name}' produces {expected}-dimensional "
                    "vectors."
                )
