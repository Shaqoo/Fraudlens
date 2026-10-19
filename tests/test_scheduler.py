"""Scheduler, job and reliability tests."""

from __future__ import annotations

import pytest

from fraudlens.scheduler.jobs import JOB_TYPES, run_data_quality_job
from fraudlens.scheduler.scheduler import MAX_ATTEMPTS, FraudLensScheduler, _execute


def test_all_documented_job_types_are_registered() -> None:
    assert {"etl", "model_training", "batch_prediction", "data_quality_check"} <= set(JOB_TYPES)


def test_unknown_job_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        _execute("not_a_job")


def test_unknown_job_type_cannot_be_registered() -> None:
    scheduler = FraudLensScheduler()
    with pytest.raises(ValueError):
        scheduler.add_schedule("bad", "not_a_job", "0 2 * * *")


def test_failed_job_returns_failed_status(monkeypatch) -> None:
    def boom(**_: object) -> dict:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setitem(JOB_TYPES, "etl", boom)
    result = _execute("etl")
    assert result["status"] == "failed"
    assert "upstream unavailable" in result["error_message"]


def test_transient_failure_is_retried(monkeypatch) -> None:
    attempts = {"count": 0}

    def flaky(**_: object) -> dict:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("temporary")
        return {"status": "succeeded"}

    monkeypatch.setitem(JOB_TYPES, "etl", flaky)
    assert _execute("etl")["status"] == "succeeded"
    assert attempts["count"] == 2
    assert MAX_ATTEMPTS >= 2


def test_data_quality_job_reports_missing_dataset(monkeypatch, tmp_path) -> None:
    from fraudlens.config import get_settings

    monkeypatch.setenv("DATABASE_URL", "")
    settings = get_settings()
    monkeypatch.setattr(type(settings), "processed_dir", property(lambda _: tmp_path / "absent"))
    assert run_data_quality_job()["status"] == "failed"


def test_scheduler_coalesce_defaults_are_configured() -> None:
    scheduler = FraudLensScheduler()
    defaults = getattr(scheduler.scheduler, "_job_defaults", {})
    assert defaults.get("max_instances") == 1
    assert defaults.get("coalesce") is True


def test_scheduler_prevents_overlapping_instances() -> None:
    scheduler = FraudLensScheduler()
    scheduler.add_schedule("daily-etl", "etl", "0 2 * * *")
    scheduler.add_schedule("daily-etl", "etl", "30 3 * * *")  # redefine, not duplicate

    assert len(scheduler.jobs()) == 1
    assert scheduler.jobs()[0].id == "daily-etl"

    scheduler.remove_schedule("daily-etl")
    assert scheduler.jobs() == []


def test_schedule_can_be_removed() -> None:
    scheduler = FraudLensScheduler()
    scheduler.add_schedule("daily-etl", "etl", "0 2 * * *", timezone="UTC")
    scheduler.remove_schedule("daily-etl")
    assert scheduler.jobs() == []
