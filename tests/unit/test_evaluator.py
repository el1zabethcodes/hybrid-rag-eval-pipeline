from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.config.settings import EvaluationSettings
from src.evaluation.evaluator import RagasEvaluator
from src.models import Chunk, PipelineResult, QAPair, ScoredChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        metadata={},
        chunk_index=0,
    )


@dataclass
class DummyPipeline:
    fail_on: set[str]

    async def query(self, query: str, top_k: int | None = None) -> PipelineResult:
        _ = top_k
        if query in self.fail_on:
            raise RuntimeError("pipeline failed")
        scored = [ScoredChunk(chunk=_chunk("c1", "ctx"), score=0.9)]
        return PipelineResult(
            answer=f"answer:{query}",
            context_chunks=scored,
            reranked_scores=[0.9],
            latency_ms=1.0,
        )


def _scores_fn(
    rows: list[dict[str, Any]]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    per_row = [
        {"faithfulness": 1.0, "answer_relevancy": 0.5, "context_precision": 0.25}
        for _ in rows
    ]
    return per_row, {"faithfulness": 1.0, "answer_relevancy": 0.5, "context_precision": 0.25}


@pytest.mark.asyncio
async def test_sampling_is_reproducible(tmp_path: Path) -> None:
    dataset = [QAPair(question=f"q{i}", ground_truth="gt") for i in range(10)]
    settings = EvaluationSettings(
        eval_output_dir=str(tmp_path), eval_sample_size=5, random_seed=123
    )
    pipeline = DummyPipeline(fail_on=set())

    evaluator1 = RagasEvaluator(pipeline=pipeline, settings=settings, ragas_scores_fn=_scores_fn)  # type: ignore[arg-type]
    evaluator2 = RagasEvaluator(pipeline=pipeline, settings=settings, ragas_scores_fn=_scores_fn)  # type: ignore[arg-type]

    report1 = await evaluator1.evaluate(dataset)
    report2 = await evaluator2.evaluate(dataset)

    q1 = [r.question for r in report1.per_question] + [e.question for e in report1.errors]
    q2 = [r.question for r in report2.per_question] + [e.question for e in report2.errors]

    assert q1 == q2


@pytest.mark.asyncio
async def test_errors_are_recorded_and_evaluation_continues(tmp_path: Path) -> None:
    dataset = [
        QAPair(question="ok1", ground_truth="gt"),
        QAPair(question="bad", ground_truth="gt"),
        QAPair(question="ok2", ground_truth="gt"),
    ]
    settings = EvaluationSettings(
        eval_output_dir=str(tmp_path), eval_sample_size=None, random_seed=42
    )
    pipeline = DummyPipeline(fail_on={"bad"})
    evaluator = RagasEvaluator(pipeline=pipeline, settings=settings, ragas_scores_fn=_scores_fn)  # type: ignore[arg-type]

    report = await evaluator.evaluate(dataset)

    assert [e.question for e in report.errors] == ["bad"]
    assert [r.question for r in report.per_question] == ["ok1", "ok2"]
    assert report.aggregate["faithfulness"] == 1.0


@pytest.mark.asyncio
async def test_report_is_persisted_as_json(tmp_path: Path) -> None:
    dataset = [QAPair(question="q", ground_truth="gt")]
    settings = EvaluationSettings(
        eval_output_dir=str(tmp_path), eval_sample_size=None, random_seed=42
    )
    pipeline = DummyPipeline(fail_on=set())
    evaluator = RagasEvaluator(pipeline=pipeline, settings=settings, ragas_scores_fn=_scores_fn)  # type: ignore[arg-type]

    report = await evaluator.evaluate(dataset)

    files = list(tmp_path.glob("report_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))

    assert data["timestamp"] == report.timestamp
    assert data["aggregate"] == report.aggregate
    assert data["errors"] == []
    assert data["per_question"][0]["question"] == "q"
