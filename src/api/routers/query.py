"""Router for query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Run the query pipeline non-streamingly.

    Args:
        request: FastAPI request object.
        payload: QueryRequest payload.

    Returns:
        QueryResponse containing answer, context, scores, and latency.
    """
    pipeline = request.app.state.pipeline
    result = await pipeline.query(payload.query, payload.top_k)
    return QueryResponse(
        answer=result.answer,
        context_chunks=result.context_chunks,
        reranked_scores=result.reranked_scores,
        latency_ms=result.latency_ms,
    )


@router.post("/query/stream")
async def query_stream(request: Request, payload: QueryRequest) -> StreamingResponse:
    """Run the query pipeline streamingly.

    Args:
        request: FastAPI request object.
        payload: QueryRequest payload.

    Returns:
        A StreamingResponse yielding tokens/chunks.
    """
    pipeline = request.app.state.pipeline

    async def token_generator():
        try:
            async for token in pipeline.stream(payload.query, payload.top_k):
                yield token
        except Exception as exc:  # noqa: BLE001
            # Send an error token frame
            yield f"\ndata: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
