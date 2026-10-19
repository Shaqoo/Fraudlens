"""Schedule definition routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from fraudlens.api.dependencies import get_db
from fraudlens.api.schemas import (
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleUpdateRequest,
)
from fraudlens.database.models import Schedule

router = APIRouter(prefix="/schedules", tags=["schedules"])


def _require_db(db: Session | None) -> Session:
    if db is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    return db


@router.post("", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
def create_schedule(
    payload: ScheduleCreateRequest,
    db: Session | None = Depends(get_db),
) -> ScheduleResponse:
    """Create a schedule definition. The scheduler process interprets it."""
    session = _require_db(db)
    existing = session.query(Schedule).filter(Schedule.name == payload.name).one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="a schedule with this name already exists")

    schedule = Schedule(**payload.model_dump())
    session.add(schedule)
    session.commit()
    session.refresh(schedule)
    return ScheduleResponse.model_validate(schedule)


@router.get("", response_model=list[ScheduleResponse])
def list_schedules(db: Session | None = Depends(get_db)) -> list[ScheduleResponse]:
    """Configuration definitions, sorted by name."""
    rows = _require_db(db).query(Schedule).order_by(Schedule.name).all()
    return [ScheduleResponse.model_validate(schedule) for schedule in rows]


@router.get("/{schedule_id}", response_model=ScheduleResponse)
def get_schedule(schedule_id: int, db: Session | None = Depends(get_db)) -> ScheduleResponse:
    schedule = _require_db(db).get(Schedule, schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="schedule not found")
    return ScheduleResponse.model_validate(schedule)


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
def update_schedule(
    schedule_id: int,
    payload: ScheduleUpdateRequest,
    db: Session | None = Depends(get_db),
) -> ScheduleResponse:
    session = _require_db(db)
    schedule = session.get(Schedule, schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="schedule not found")

    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(schedule, field, value)
    session.commit()
    session.refresh(schedule)
    return ScheduleResponse.model_validate(schedule)


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_schedule(schedule_id: int, db: Session | None = Depends(get_db)) -> None:
    session = _require_db(db)
    schedule = session.get(Schedule, schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="schedule not found")
    session.delete(schedule)
    session.commit()
