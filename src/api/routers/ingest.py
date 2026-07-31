"""Router for document ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import IngestRequest, IngestResponse
from src.exceptions import IngestionError

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, payload: IngestRequest) -> IngestResponse:
    """Ingest documents into the Qdrant hybrid-vector collection.

    Args:
        request: FastAPI request object.
        payload: IngestRequest payload.

    Returns:
        IngestResponse containing count of chunks ingested and collection name.
    """
    ingestion_service = request.app.state.ingestion_service
    write_lock = request.app.state.write_lock

    async with write_lock:
        try:
            result = await ingestion_service.ingest(payload.documents)
            return IngestResponse(
                chunks_ingested=result.chunks_ingested,
                collection_name=result.collection_name,
            )
        except IngestionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
