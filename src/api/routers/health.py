"""Router for health check endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Check the health status of Qdrant and LLM components.

    Args:
        request: FastAPI request object.

    Returns:
        A HealthResponse containing ok status and details.
    """
    qdrant_ok = False
    llm_ok = False
    details: dict[str, Any] = {}

    try:
        client = request.app.state.qdrant_client
        client.get_collections()
        qdrant_ok = True
    except Exception as exc:  # noqa: BLE001
        details["qdrant_error"] = str(exc)

    try:
        if request.app.state.generator:
            llm_ok = True
    except Exception as exc:  # noqa: BLE001
        details["llm_error"] = str(exc)

    ok = qdrant_ok and llm_ok
    return HealthResponse(ok=ok, qdrant_ok=qdrant_ok, llm_ok=llm_ok, details=details)
