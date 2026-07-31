"""Ragas-based evaluation runner for the Hybrid RAG Evaluation Pipeline."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.settings import EvaluationSettings
from src.exceptions import EvaluationError
from src.models import EvaluationReport, QAPair, QuestionError, QuestionResult
from src.pipeline.pipeline import Pipeline

RagasScoresFn = Callable[[list[dict[str, Any]]], tuple[list[dict[str, float]], dict[str, float]]]


class RagasEvaluator:
    """Evaluate pipeline outputs using Ragas metrics and persist JSON reports.

    Args:
        pipeline: Configured :class:`~src.pipeline.pipeline.Pipeline` instance.
        settings: Evaluation configuration.
        ragas_scores_fn: Optional scoring function used for unit tests. When not
            provided, scores are computed via the ``ragas`` library.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        settings: EvaluationSettings,
        ragas_scores_fn: RagasScoresFn | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._settings = settings
        self._ragas_scores_fn = ragas_scores_fn

    async def evaluate(self, qa_dataset: list[QAPair]) -> EvaluationReport:
        """Run Ragas evaluation on a QA dataset.

        Args:
            qa_dataset: List of question/ground-truth pairs.

        Returns:
            An :class:`~src.models.EvaluationReport` and persists it as JSON.

        Raises:
            EvaluationError: On fatal configuration or evaluation failures.
        """
        timestamp_iso, filename_stamp = self._timestamps()
        sampled = self._sample_dataset(qa_dataset)

        rows: list[dict[str, Any]] = []
        errors: list[QuestionError] = []

        for qa in sampled:
            try:
                result = await self._pipeline.query(qa.question)
                rows.append(
                    {
                        "question": qa.question,
                        "answer": result.answer,
                        "contexts": [c.chunk.text for c in result.context_chunks],
                        "ground_truth": qa.ground_truth,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(QuestionError(question=qa.question, error_message=str(exc)))

        per_question: list[QuestionResult] = []
        aggregate: dict[str, float] = {}

        if rows:
            per_row_scores, aggregate = await self._score_rows(rows)
            if len(per_row_scores) != len(rows):
                raise EvaluationError(
                    "Ragas scoring produced a score count that does not match the number of rows."
                )
            for row, scores in zip(rows, per_row_scores, strict=True):
                per_question.append(
                    QuestionResult(
                        question=str(row["question"]),
                        answer=str(row["answer"]),
                        contexts=list(row["contexts"]),
                        ground_truth=str(row.get("ground_truth", "")),
                        scores=scores,
                    )
                )

        report = EvaluationReport(
            per_question=per_question,
            aggregate=aggregate,
            errors=errors,
            timestamp=timestamp_iso,
        )
        self._persist_report(report, filename_stamp)
        return report

    def _sample_dataset(self, qa_dataset: list[QAPair]) -> list[QAPair]:
        sample_size = self._settings.eval_sample_size
        if sample_size is None or sample_size >= len(qa_dataset):
            return list(qa_dataset)
        rng = random.Random(self._settings.random_seed)
        return rng.sample(list(qa_dataset), k=sample_size)

    async def _score_rows(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        if self._ragas_scores_fn is not None:
            return self._ragas_scores_fn(rows)

        self._validate_judge_settings()
        try:
            from datasets import Dataset  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(
                "Ragas evaluation requires the 'datasets' package to be installed."
            ) from exc

        try:
            from ragas import evaluate as ragas_evaluate
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(
                "Ragas evaluation requires the 'ragas' package to be installed."
            ) from exc

        metric_instances = self._build_ragas_metrics()
        dataset = Dataset.from_list(rows)
        try:
            ragas_result = ragas_evaluate(dataset=dataset, metrics=metric_instances)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Ragas evaluation failed: {exc}") from exc

        per_row_scores = self._extract_per_row_scores(ragas_result, len(rows))
        aggregate = self._extract_aggregate_scores(ragas_result, per_row_scores)
        return per_row_scores, aggregate

    def _validate_judge_settings(self) -> None:
        provider = self._settings.judge_llm_provider
        api_key = self._settings.judge_llm_api_key or os.getenv("RAGAS_LLM_API_KEY")
        if provider == "openai" and not api_key:
            raise EvaluationError(
                "Missing judge LLM API key. Set evaluation.judge_llm_api_key or RAGAS_LLM_API_KEY."
            )
        if provider in {"ollama", "litellm"} and not self._settings.judge_llm_base_url:
            raise EvaluationError(
                f"Judge provider '{provider}' requires evaluation.judge_llm_base_url to be set."
            )
        if api_key:
            os.environ.setdefault("RAGAS_LLM_API_KEY", api_key)
            os.environ.setdefault("OPENAI_API_KEY", api_key)

    def _build_ragas_metrics(self) -> list[Any]:
        try:
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                Faithfulness,
            )
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(
                "Failed to import required Ragas metrics "
                "(Faithfulness, AnswerRelevancy, ContextPrecision)."
            ) from exc

        metrics: list[Any] = []
        for metric_cls in (Faithfulness, AnswerRelevancy, ContextPrecision):
            try:
                metrics.append(metric_cls())
            except TypeError:
                metrics.append(metric_cls)
        return metrics

    @staticmethod
    def _extract_per_row_scores(result: Any, expected_len: int) -> list[dict[str, float]]:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            per_row: list[dict[str, float]] = []
            exclude = {"question", "answer", "contexts", "ground_truth"}
            cols = [c for c in df.columns if c not in exclude]
            for _, row in df.iterrows():
                per_row.append({c: float(row[c]) for c in cols})
            return per_row[:expected_len]

        if hasattr(result, "scores"):
            scores = getattr(result, "scores")
            if isinstance(scores, Sequence):
                per_row_scores: list[dict[str, float]] = []
                for item in scores:
                    if isinstance(item, dict):
                        per_row_scores.append({k: float(v) for k, v in item.items()})
                return per_row_scores[:expected_len]

        if isinstance(result, dict) and "scores" in result and isinstance(result["scores"], list):
            per_row_scores = []
            for item in result["scores"]:
                if isinstance(item, dict):
                    per_row_scores.append({k: float(v) for k, v in item.items()})
            return per_row_scores[:expected_len]

        raise EvaluationError("Unable to extract per-row scores from Ragas result.")

    @staticmethod
    def _extract_aggregate_scores(
        result: Any, per_row_scores: list[dict[str, float]]
    ) -> dict[str, float]:
        if hasattr(result, "scores") and isinstance(getattr(result, "scores"), dict):
            scores_dict = getattr(result, "scores")
            return {k: float(v) for k, v in scores_dict.items()}

        if (
            isinstance(result, dict)
            and "aggregate" in result
            and isinstance(result["aggregate"], dict)
        ):
            return {k: float(v) for k, v in result["aggregate"].items()}

        if not per_row_scores:
            return {}

        keys = sorted({k for row in per_row_scores for k in row.keys()})
        aggregate: dict[str, float] = {}
        for key in keys:
            values = [row[key] for row in per_row_scores if key in row]
            if values:
                aggregate[key] = sum(values) / float(len(values))
        return aggregate

    def _persist_report(self, report: EvaluationReport, filename_stamp: str) -> None:
        output_dir = Path(self._settings.eval_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"report_{filename_stamp}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _timestamps() -> tuple[str, str]:
        now = datetime.now(UTC)
        return now.replace(microsecond=0).isoformat(), now.strftime("%Y%m%dT%H%M%SZ")
