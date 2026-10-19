"""Operational metrics route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fraudlens.api.dependencies import get_db
from fraudlens.api.schemas import MetricsResponse
from fraudlens.database.models import EtlRun, Prediction

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", response_model=MetricsResponse)
def metrics(db: Session | None = Depends(get_db)) -> MetricsResponse:
    """Operational metrics. Distinct from model-quality metrics."""
    if db is None:
        raise HTTPException(status_code=503, detail="database unavailable")

    total_predictions = db.execute(select(func.count()).select_from(Prediction)).scalar_one()
    flagged = db.execute(
        select(func.count()).select_from(Prediction).where(Prediction.predicted_label == 1)
    ).scalar_one()

    finished = db.execute(
        select(EtlRun.status, func.count()).select_from(EtlRun).group_by(EtlRun.status)
    ).all()
    status_counts = dict(finished)
    succeeded = status_counts.get("succeeded", 0)
    failed = status_counts.get("failed", 0)
    completion_rate = succeeded / (succeeded + failed) if (succeeded + failed) else None

    return MetricsResponse(
        request_count=total_predictions,
        error_count=failed,
        predictions_total=total_predictions,
        predictions_flagged=flagged,
        etl_success_rate=completion_rate,
    )
