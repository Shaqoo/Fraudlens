"""Persistence operations used by the API, ETL and scheduler."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fraudlens.database.models import ApplicationEvent, EtlRun, Prediction, ScheduleRun


def record_etl_run(
    session: Session,
    dataset_source_id: int | None,
    status: str,
    rows_extracted: int = 0,
    rows_processed: int = 0,
    rows_loaded: int = 0,
    rows_rejected: int = 0,
    error_message: str | None = None,
) -> EtlRun:
    run = EtlRun(
        dataset_source_id=dataset_source_id,
        status=status,
        rows_extracted=rows_extracted,
        rows_processed=rows_processed,
        rows_loaded=rows_loaded,
        rows_rejected=rows_rejected,
        error_message=error_message,
    )
    if status in {"succeeded", "failed", "partial"}:
        run.finished_at = func.now()
    session.add(run)
    session.flush()
    return run


def record_prediction(
    session: Session,
    model_version_id: int,
    fraud_probability: float,
    predicted_label: int,
    transaction_id: int | None = None,
) -> Prediction:
    prediction = Prediction(
        transaction_id=transaction_id,
        model_version_id=model_version_id,
        fraud_probability=fraud_probability,
        predicted_label=predicted_label,
        review_status="pending",
    )
    session.add(prediction)
    session.flush()
    return prediction


def update_review_status(session: Session, prediction_id: int, review_status: str, note: str | None = None) -> Prediction | None:
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        return None
    prediction.review_status = review_status
    prediction.review_note = note
    return prediction


def list_predictions(
    session: Session,
    predicted_label: int | None = None,
    min_probability: float | None = None,
    review_status: str | None = None,
    model_version: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[int, list[Prediction]]:
    statement = select(Prediction)
    count_statement = select(func.count()).select_from(Prediction)

    if predicted_label is not None:
        statement = statement.where(Prediction.predicted_label == predicted_label)
        count_statement = count_statement.where(Prediction.predicted_label == predicted_label)
    if min_probability is not None:
        statement = statement.where(Prediction.fraud_probability >= min_probability)
        count_statement = count_statement.where(Prediction.fraud_probability >= min_probability)
    if review_status is not None:
        statement = statement.where(Prediction.review_status == review_status)
        count_statement = count_statement.where(Prediction.review_status == review_status)

    total = session.execute(count_statement).scalar_one()
    rows = session.execute(statement.order_by(Prediction.created_at.desc()).limit(limit).offset(offset)).scalars().all()
    return total, list(rows)


def record_schedule_run(session: Session, schedule_id: int, status: str, attempt_number: int = 1, error_message: str | None = None, result_summary: dict | None = None) -> ScheduleRun:
    run = ScheduleRun(
        schedule_id=schedule_id,
        status=status,
        attempt_number=attempt_number,
        error_message=error_message,
        result_summary=result_summary,
    )
    session.add(run)
    session.flush()
    return run


def record_event(session: Session, severity: str, event_type: str, message: str, component: str | None = None, context: dict | None = None) -> ApplicationEvent:
    event = ApplicationEvent(
        severity=severity,
        event_type=event_type,
        component=component,
        message=message,
        context=context or {},
    )
    session.add(event)
    return event
