"""Typed exception hierarchy for the Hybrid RAG Evaluation Pipeline.

All pipeline-specific exceptions inherit from ``HybridRAGError`` so callers
can catch the entire family with a single ``except HybridRAGError`` clause
while still being able to distinguish specific failure modes when needed.
"""

from __future__ import annotations


class HybridRAGError(Exception):
    """Base class for all Hybrid RAG pipeline exceptions.

    Args:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover
        return self.message or super().__str__()


class ConfigurationError(HybridRAGError):
    """Raised by ``AppSettings`` when a required configuration key is absent
    or a value fails type validation.

    The error message MUST contain the exact name of the missing/invalid key
    so that operators can diagnose the problem without reading source code.

    Args:
        message: Description of the configuration problem, including the key name.

    Example::

        raise ConfigurationError("Missing required configuration key: 'qdrant__api_key'")
    """


class IngestionError(HybridRAGError):
    """Raised by ``IngestionService`` when the Qdrant write step fails.

    This error is raised *only* after all chunking and embedding steps have
    completed successfully, so callers know that no data has been persisted
    and can safely retry the storage step.

    Args:
        message: Description of the Qdrant connectivity or write failure.
    """


class RetrievalError(HybridRAGError):
    """Raised by ``HybridRetriever`` when a Qdrant query fails.

    Args:
        message: Description of the query failure, including which vector type
            (dense or sparse) triggered the error when determinable.
    """


class RerankerError(HybridRAGError):
    """Raised by ``CrossEncoderReranker`` when the cross-encoder model forward
    pass fails.

    Args:
        message: Description of the model or inference failure.
    """


class GenerationError(HybridRAGError):
    """Raised by ``LLMGenerator`` when the configured LLM backend returns an
    error response.

    The error message MUST include the upstream HTTP status code and the raw
    error message returned by the backend so callers can distinguish transient
    failures (e.g. 429, 503) from permanent ones (e.g. 401).

    Args:
        message: Human-readable description including upstream status and message.
        status_code: HTTP status code returned by the LLM backend, if available.
    """

    def __init__(self, message: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class StreamingError(GenerationError):
    """Raised by ``LLMGenerator`` when an error occurs *during* a streaming
    response — i.e. after the response stream has already been opened and
    partial tokens may have been yielded.

    When this exception is raised the generator MUST discard all buffered
    partial output and the API router MUST send a final SSE error frame before
    closing the stream.

    Args:
        message: Description of the mid-stream failure.
        status_code: HTTP status code returned by the backend, if available.
    """


class PipelineError(HybridRAGError):
    """Raised by ``Pipeline`` to wrap any exception that originates in one of
    the three pipeline steps (retriever, reranker, generator).

    Using a dedicated wrapper type lets the FastAPI global exception handler
    return a structured HTTP 500 response that identifies *which* step failed
    without leaking internal implementation details.

    Args:
        message: Human-readable description of the pipeline failure.
        step_name: Name of the failing step.  MUST be one of
            ``"retriever"``, ``"reranker"``, or ``"generator"``.
        cause: The original exception raised by the failing step.

    Example::

        try:
            candidates = await self.retriever.retrieve(query)
        except RetrievalError as exc:
            raise PipelineError(
                message=f"Retrieval step failed: {exc}",
                step_name="retriever",
                cause=exc,
            ) from exc
    """

    def __init__(
        self,
        message: str = "",
        step_name: str = "",
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.step_name = step_name
        self.cause = cause


class EvaluationError(HybridRAGError):
    """Raised by ``RagasEvaluator`` when a *fatal* evaluation failure occurs
    that prevents the evaluation run from completing.

    Non-fatal per-question errors are recorded in the ``EvaluationReport``
    rather than surfaced as exceptions; this exception is reserved for
    scenarios such as a missing judge LLM API key or an unrecoverable Ragas
    library error.

    Args:
        message: Description of the fatal evaluation failure.
    """
