import uuid
from datetime import UTC, datetime

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.data.connection import Database
from app.data.db import get_db
from app.schemas.events import IngestResponse, UsageEvent

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/events",
    tags=["Events"],
    dependencies=[db_dep],
)


@router.post(
    "/ingest",
    summary="Ingest a usage event",
    description=(
        "Accepts a `UsageEvent` JSON body, auto-generates `event_id` and "
        "`timestamp` if omitted, then persists it into the MongoDB collection "
        "named by `event_type`. Collections are created on first insert."
    ),
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    response_description="Confirmation with stored event identifiers",
    operation_id="ingestUsageEvent",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid payload or database insertion error",
        },
    },
)
async def ingest_event(
    event: UsageEvent,
    db: Database = db_dep,
) -> IngestResponse:
    if not event.event_id:
        event.event_id = str(uuid.uuid4())
    if not event.timestamp:
        event.timestamp = datetime.now(UTC)

    coll = db.get_collection(event.event_type)
    doc = event.model_dump()

    try:
        result = await coll.insert_one(doc)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to insert event: {exc}",
        ) from exc

    return IngestResponse(
        status="ok",
        event_type=event.event_type,
        event_id=event.event_id,
        inserted_id=str(result.inserted_id),
    )
