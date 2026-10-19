"""Command line entry points for the ETL, training and scheduler stages.

Each command returns a meaningful exit code so it can be called safely by a
scheduler or a CI workflow: 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from fraudlens.monitoring.logging_config import configure_logging

logger = logging.getLogger("fraudlens.cli")


def _emit(summary: dict) -> int:
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("status") == "succeeded" else 1


def etl_main(argv: list[str] | None = None) -> int:
    """Run extract -> validate -> transform -> save -> load."""
    parser = argparse.ArgumentParser(description="Run the FraudLens ETL pipeline")
    parser.add_argument(
        "--source",
        choices=["huggingface", "kaggle"],
        default=None,
        help="dataset source; defaults to the configured Kaggle dataset",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    from fraudlens.scheduler.jobs import run_etl_job

    return _emit(run_etl_job(source_type=args.source))


def train_main(argv: list[str] | None = None) -> int:
    """Train a candidate model and write the evaluation report."""
    parser = argparse.ArgumentParser(description="Train the FraudLens XGBoost model")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    from fraudlens.scheduler.jobs import run_training_job

    return _emit(run_training_job())


def scheduler_main(argv: list[str] | None = None) -> int:
    """Run the scheduler as a standalone process."""
    parser = argparse.ArgumentParser(description="Run the FraudLens scheduler")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    import time

    from fraudlens.scheduler.scheduler import build_scheduler_from_db

    scheduler = build_scheduler_from_db(timezone=args.timezone)
    logger.info("scheduler_cli_running")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
    return 0


if __name__ == "__main__":
    commands = {"etl": etl_main, "train": train_main, "scheduler": scheduler_main}
    name = sys.argv[1] if len(sys.argv) > 1 else "etl"
    if name not in commands:
        print(f"unknown command: {name}. expected one of {sorted(commands)}", file=sys.stderr)
        sys.exit(2)
    sys.exit(commands[name](sys.argv[2:]))
