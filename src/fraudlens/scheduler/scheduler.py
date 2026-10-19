"""APScheduler wiring with execution locks and retry handling."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from fraudlens.scheduler.jobs import JOB_TYPES

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
JOB_DEFAULTS = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 300}


def _execute(job_type: str, **kwargs) -> dict:
    """Run a job, recording outcome. Overlapping runs are prevented by max_instances."""
    handler = JOB_TYPES.get(job_type)
    if handler is None:
        raise ValueError(f"unknown job type: {job_type}")

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return handler(**kwargs)
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            logger.warning(
                "job_attempt_failed",
                extra={"job_type": job_type, "attempt": attempt, "error": str(exc)},
            )

    logger.error("job_failed", extra={"job_type": job_type, "attempts": MAX_ATTEMPTS})
    return {"status": "failed", "error_message": str(last_error)}


class FraudLensScheduler:
    """Thin wrapper that keeps schedule definitions and job handlers separate."""

    def __init__(self, timezone: str = "UTC") -> None:
        self.scheduler = BackgroundScheduler(timezone=timezone, job_defaults=JOB_DEFAULTS)

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("scheduler_started")

    def shutdown(self, wait: bool = True) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("scheduler_stopped")

    def add_schedule(self, name: str, job_type: str, cron_expression: str, timezone: str = "UTC") -> None:
        if job_type not in JOB_TYPES:
            raise ValueError(f"unknown job type: {job_type}")
        # APScheduler's replace_existing only dedupes in jobstores once the
        # scheduler is running; purge pending duplicates explicitly.
        if self.scheduler.get_job(name) is not None:
            self.scheduler.remove_job(name)
        self.scheduler.add_job(
            _execute,
            trigger=CronTrigger.from_crontab(cron_expression, timezone=timezone),
            args=[job_type],
            id=name,
            name=name,
            replace_existing=True,
        )
        logger.info("schedule_added", extra={"schedule_name": name, "job_type": job_type})

    def remove_schedule(self, name: str) -> None:
        self.scheduler.remove_job(name)
        logger.info("schedule_removed", extra={"schedule_name": name})

    def jobs(self):
        return self.scheduler.get_jobs()


def build_scheduler_from_db(timezone: str = "UTC") -> FraudLensScheduler:
    """Load enabled schedules from PostgreSQL into a running scheduler."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from fraudlens.database.connection import get_engine
    from fraudlens.database.models import Schedule

    scheduler = FraudLensScheduler(timezone=timezone)
    with Session(get_engine()) as session:
        schedules = session.execute(
            select(Schedule).where(Schedule.enabled.is_(True))
        ).scalars().all()
        for record in schedules:
            scheduler.add_schedule(
                record.name, record.job_type, record.cron_expression, record.timezone or timezone
            )
    scheduler.start()
    return scheduler
