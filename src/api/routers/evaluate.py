"""Router for Ragas evaluation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import EvaluateRequest, EvaluateResponse
from src.exceptions import EvaluationError

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: Request, payload: EvaluateRequest) -> EvaluateResponse:
    """Run Ragas evaluation on the provided dataset.

    Args:
        request: FastAPI request object.
        payload: EvaluateRequest payload.

    Returns:
        EvaluateResponse containing the evaluation report.
    """
    evaluator = request.app.state.evaluator
    try:
        report = await evaluator.evaluate(payload.qa_dataset)
        return EvaluateResponse(report=report)
    except EvaluationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
