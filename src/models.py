"""Shared data-transfer objects (DTOs) for the Hybrid RAG Evaluation Pipeline.

Pydantic v2 ``BaseModel`` types are used for externally-facing structures that
benefit from runtime validation and serialisation.  Pure Python ``dataclass``
types are used for internal pipeline results where construction is always
controlled and validation overhead is unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic v2 BaseModel DTOs
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """A unit of raw text submitted for ingestion.

    Args:
        id: Unique identifier for the document (caller-supplied).
        text: Full text content of the document.
        metadata: Arbitrary key-value metadata attached to the document (e.g.
            source URL, author, timestamp).  Defaults to an empty dict.
    """

    id: str
    text: str
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    """A fixed-size or semantically-bounded segment of a :class:`Document`.

    Chunks are the retrieval unit stored in Qdrant.  Each chunk knows its
    parent document so that source attribution is always available.

    Args:
        id: Unique identifier derived from the parent document, formatted as
            ``"{document_id}_{chunk_index}"``.
        document_id: Identifier of the :class:`Document` this chunk came from.
        text: The chunk text content.
        metadata: Metadata inherited or derived from the parent document.
            Defaults to an empty dict.
        chunk_index: Zero-based position of this chunk within the parent
            document's chunk sequence.
    """

    id: str  # "{document_id}_{chunk_index}"
    document_id: str
    text: str
    metadata: dict[str, Any] = {}
    chunk_index: int


class ScoredChunk(BaseModel):
    """A :class:`Chunk` paired with a relevance score.

    The interpretation of ``score`` depends on the pipeline stage:

    * **Pre-rerank**: RRF (Reciprocal Rank Fusion) score produced by
      :func:`~src.retrieval.rrf.rrf_merge`.
    * **Post-rerank**: Cross-encoder score assigned by
      :class:`~src.reranking.reranker.CrossEncoderReranker`.

    Args:
        chunk: The retrieved chunk.
        score: Relevance score (higher is better).
    """

    chunk: Chunk
    score: float


class QAPair(BaseModel):
    """A single question–answer pair used by the Ragas evaluator.

    Args:
        question: The natural-language question to ask the pipeline.
        ground_truth: The reference answer used for evaluation metrics.
        metadata: Optional metadata for the QA pair (e.g. category, difficulty).
            Defaults to an empty dict.
    """

    question: str
    ground_truth: str
    metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Python dataclass DTOs
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Result of a single LLM generation call.

    Returned by :meth:`~src.generation.generator.LLMGenerator.generate`.

    Attributes:
        answer: The generated answer text.
        model_id: Identifier of the LLM model that produced the answer
            (e.g. ``"gpt-4o-mini"``).
        prompt_tokens: Number of tokens in the prompt sent to the LLM.
        completion_tokens: Number of tokens in the generated answer.
        total_tokens: Sum of ``prompt_tokens`` and ``completion_tokens``.
    """

    answer: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class PipelineResult:
    """End-to-end result of a single query through the pipeline.

    Returned by :meth:`~src.pipeline.pipeline.Pipeline.query`.

    Attributes:
        answer: The final generated answer.
        context_chunks: Ordered list of :class:`ScoredChunk` objects that were
            passed as context to the LLM (post-rerank order).
        reranked_scores: Cross-encoder scores parallel to ``context_chunks``
            (``reranked_scores[i]`` corresponds to ``context_chunks[i]``).
        latency_ms: Wall-clock time in milliseconds from query receipt to
            result completion.
    """

    answer: str
    context_chunks: list[ScoredChunk]
    reranked_scores: list[float]
    latency_ms: float


@dataclass
class IngestionResult:
    """Result of a document ingestion run.

    Returned by :meth:`~src.ingestion.ingestor.IngestionService.ingest`.

    Attributes:
        chunks_ingested: Total number of :class:`Chunk` objects stored in
            Qdrant during this ingestion run.
        collection_name: Name of the Qdrant collection that received the chunks.
    """

    chunks_ingested: int
    collection_name: str


@dataclass
class QuestionResult:
    """Per-question evaluation output produced by the Ragas evaluator.

    Attributes:
        question: The original question from the QA dataset.
        answer: The answer generated by the pipeline.
        contexts: List of context texts (chunk texts) used to produce the answer.
        ground_truth: The reference answer from the QA dataset.
        scores: Metric name → score mapping (e.g.
            ``{"faithfulness": 0.9, "answer_relevancy": 0.85,
            "context_precision": 0.75}``).
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    scores: dict[str, float]


@dataclass
class QuestionError:
    """Record of a pipeline failure for a specific question during evaluation.

    When :meth:`~src.evaluation.evaluator.RagasEvaluator.evaluate` encounters
    a per-question error it records a :class:`QuestionError` in the report and
    continues processing the remaining questions.

    Attributes:
        question: The question that caused the failure.
        error_message: Human-readable description of the failure.
    """

    question: str
    error_message: str


@dataclass
class EvaluationReport:
    """Aggregated results from a Ragas evaluation run.

    Returned by :meth:`~src.evaluation.evaluator.RagasEvaluator.evaluate` and
    persisted as a JSON file to ``EvaluationSettings.eval_output_dir``.

    Attributes:
        per_question: Per-question scored results for all successfully evaluated
            questions.
        aggregate: Mean score per metric across all scored questions (metric
            name → mean value).
        errors: Records for any questions that produced a pipeline error.
        timestamp: ISO-8601 timestamp string marking when the evaluation run
            completed.
    """

    per_question: list[QuestionResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    errors: list[QuestionError] = field(default_factory=list)
    timestamp: str = ""
