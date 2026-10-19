"""Database layer tests using an in-memory SQLite database."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from fraudlens.database.models import ApplicationEvent, Prediction, Schedule, ScheduleRun
from fraudlens.database.repositories import (
    list_predictions,
    record_etl_run,
    record_event,
    record_prediction,
    record_schedule_run,
    update_review_status,
)


def test_record_etl_run_stores_row_counts(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        run = record_etl_run(session, None, "succeeded", 1000, 990, 990, 10)
        session.commit()
        assert run.rows_extracted - run.rows_loaded == run.rows_rejected


def test_predictions_are_traceable_to_a_model_version(sqlite_engine) -> None:
    from fraudlens.database.models import ModelVersion

    with Session(sqlite_engine) as session:
        version = ModelVersion(version="xgb-v1", artifact_uri="models/xgb-v1.joblib", status="active")
        session.add(version)
        session.flush()

        first = record_prediction(session, version.id, 0.91, 1)
        record_prediction(session, version.id, 0.12, 0)
        session.commit()

        assert first.model_version_id == version.id
        assert session.execute(select(Prediction)).scalars().all()[0].model_version_id == version.id


def test_repeated_predictions_are_not_overwritten(sqlite_engine) -> None:
    from fraudlens.database.models import ModelVersion

    with Session(sqlite_engine) as session:
        version = ModelVersion(version="xgb-v1", artifact_uri="models/xgb-v1.joblib")
        session.add(version)
        session.flush()

        record_prediction(session, version.id, 0.4, 0)
        record_prediction(session, version.id, 0.8, 1)
        session.commit()

        assert session.execute(select(Prediction)).scalars().all().__len__() == 2


def test_review_status_is_separate_from_predicted_label(sqlite_engine) -> None:
    from fraudlens.database.models import ModelVersion

    with Session(sqlite_engine) as session:
        version = ModelVersion(version="xgb-v1", artifact_uri="models/xgb-v1.joblib")
        session.add(version)
        session.flush()
        prediction = record_prediction(session, version.id, 0.6, 1)
        session.commit()

        updated = update_review_status(session, prediction.id, "legitimate", "checked by reviewer")
        session.commit()

        assert updated.predicted_label == 1
        assert updated.review_status == "legitimate"


def test_list_predictions_applies_filters(sqlite_engine) -> None:
    from fraudlens.database.models import ModelVersion

    with Session(sqlite_engine) as session:
        version = ModelVersion(version="xgb-v1", artifact_uri="models/xgb-v1.joblib")
        session.add(version)
        session.flush()

        record_prediction(session, version.id, 0.02, 0)
        record_prediction(session, version.id, 0.95, 1)
        record_prediction(session, version.id, 0.75, 1)
        session.commit()

        total, rows = list_predictions(session, predicted_label=1)
        assert total == 2
        assert all(row.predicted_label == 1 for row in rows)

        total, rows = list_predictions(session, min_probability=0.8)
        assert total == 1
        assert rows[0].fraud_probability == pytest.approx(0.95)


def test_list_predictions_paginates(sqlite_engine) -> None:
    from fraudlens.database.models import ModelVersion

    with Session(sqlite_engine) as session:
        version = ModelVersion(version="xgb-v1", artifact_uri="models/xgb-v1.joblib")
        session.add(version)
        session.flush()
        for _ in range(5):
            record_prediction(session, version.id, 0.5, 0)
        session.commit()

        total, rows = list_predictions(session, limit=2, offset=0)
        assert total == 5
        assert len(rows) == 2


def test_schedule_and_schedule_runs(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        schedule = Schedule(name="daily-etl", job_type="etl", cron_expression="0 2 * * *")
        session.add(schedule)
        session.flush()

        run = record_schedule_run(session, schedule.id, "succeeded", result_summary={"rows_loaded": 10})
        session.commit()

        assert schedule.next_run_at is None
        assert run.result_summary == {"rows_loaded": 10}
        assert session.execute(select(ScheduleRun)).scalars().all()


def test_application_events_are_queryable(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        record_event(session, "ERROR", "etl_failed", "boom", component="etl", context={"run_id": 1})
        session.commit()

        events = session.execute(
            select(ApplicationEvent).where(ApplicationEvent.severity == "ERROR")
        ).scalars().all()
        assert len(events) == 1
        assert events[0].context == {"run_id": 1}
