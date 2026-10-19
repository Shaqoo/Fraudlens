"""ETL run request and status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from fraudlens.api.dependencies import get_db
from fraudlens.api.schemas import EtlRunRequest, EtlRunResponse
from fraudlens.database.models import EtlRun
from fraudlens.database.repositories import record_etl_run
from fraudlens.scheduler.jobs import run_etl_job

router = APIRouter(prefix="/etl/runs", tags=["etl"])


def _require_db(db):
    if db is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    return db


@router.post("", response_model=EtlRunResponse, status_code=status.HTTP_202_ACCEPTED)
@router.post("/", response_model=EtlRunResponse, status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
def request_etl_run(payload: EtlRunRequest, db: Session | None = Depends(get_db)) -> EtlRunResponse:
    """Enqueue an ETL execution and return a tracking identifier.

    The pipeline runs outside the request. A local synchronous runner is used
    until a background worker is available.
    """
    run = record_etl_run(_require_db(db), dataset_source_id=None, status="queued")
    run_id = run.id
    db.commit()

    try:
        summary = run_etl_job(source_type=payload.source_type, dataset_id=payload.dataset_id)
    except Exception as exc:  # noqa: BLE001 - reported through the run record
        summary = {"status": "failed", "error_message": str(exc)}

    run = db.get(EtlRun, run_id)
    run.status = summary.get("status", "failed")
    run.rows_extracted = summary.get("rows_extracted", 0)
    run.rows_processed = summary.get("rows_processed", 0)
    run.rows_loaded = summary.get("rows_loaded", 0)
    run.rows_rejected = summary.get("rows_rejected", 0)
    run.error_message = summary.get("error_message")
    db.commit()

    return EtlRunResponse(
        run_id=run_id,
        status=run.status,
        message="ETL execution has been scheduled",
        rows_extracted=run.rows_extracted,
        rows_processed=run.rows_processed,
        rows_loaded=run.rows_loaded,
        rows_rejected=run.rows_rejected,
    )


@router.get("", response_model=list[EtlRunResponse])
def list_etl_runs(db: Session | None = Depends(get_db)) -> list[EtlRunResponse]:
    """Historical ETL executions, newest first."""
    runs = _require_db(db).query(EtlRun).order_by(EtlRun.started_at.desc()).limit(100).all()
    return [
        EtlRunResponse(
            run_id=run.id,
            status=run.status,
            rows_extracted=run.rows_extracted,
            rows_processed=run.rows_processed,
            rows_loaded=run.rows_loaded,
            rows_rejected=run.rows_rejected,
        )
        for run in runs
    ]


@router.get("/{run_id}", response_model=EtlRunResponse)
def get_etl_run(run_id: int, db: Session | None = Depends(get_db)) -> EtlRunResponse:
    """Status and row-count reconciliation for a single ETL run."""
    run = _require_db(db).get(EtlRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="etl run not found")

    return EtlRunResponse(
        run_id=run.id,
        status=run.status,
        message=run.error_message,
        rows_extracted=run.rows_extracted,
        rows_processed=run.rows_processed,
        rows_loaded=run.rows_loaded,
        rows_rejected=run.rows_rejected,
    )
