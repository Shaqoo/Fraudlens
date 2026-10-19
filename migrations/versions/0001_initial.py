"""Initial schema for FraudLens.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_sources",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("dataset_key", sa.String(255), nullable=False, unique=True),
        sa.Column("source_name", sa.String(255)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("version", sa.String(64), nullable=False, unique=True),
        sa.Column("artifact_uri", sa.String(512), nullable=False),
        sa.Column("threshold", sa.Float, server_default="0.5"),
        sa.Column("evaluation_metrics", sa.JSON),
        sa.Column("training_dataset_id", sa.BigInteger, sa.ForeignKey("dataset_sources.id")),
        sa.Column("status", sa.String(32), server_default="candidate"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "transactions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("dataset_source_id", sa.BigInteger, sa.ForeignKey("dataset_sources.id")),
        sa.Column("source_transaction_id", sa.String(255)),
        sa.Column("features", sa.JSON, nullable=False),
        sa.Column("actual_label", sa.SmallInteger),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("transaction_id", sa.BigInteger, sa.ForeignKey("transactions.id")),
        sa.Column("model_version_id", sa.BigInteger, sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("fraud_probability", sa.Float, nullable=False),
        sa.Column("predicted_label", sa.SmallInteger, nullable=False),
        sa.Column("review_status", sa.String(32), server_default="pending"),
        sa.Column("review_note", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("dataset_source_id", sa.BigInteger, sa.ForeignKey("dataset_sources.id")),
        sa.Column("status", sa.String(32), server_default="running"),
        sa.Column("metrics", sa.JSON),
        sa.Column("config", sa.JSON),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "etl_runs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("dataset_source_id", sa.BigInteger, sa.ForeignKey("dataset_sources.id")),
        sa.Column("status", sa.String(32), server_default="running"),
        sa.Column("rows_extracted", sa.Integer, server_default="0"),
        sa.Column("rows_processed", sa.Integer, server_default="0"),
        sa.Column("rows_loaded", sa.Integer, server_default="0"),
        sa.Column("rows_rejected", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "schedules",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("job_type", sa.String(64), nullable=False),
        sa.Column("cron_expression", sa.String(128), nullable=False),
        sa.Column("timezone", sa.String(64), server_default="UTC"),
        sa.Column("enabled", sa.Boolean, server_default=sa.true()),
        sa.Column("next_run_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "schedule_runs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("schedule_id", sa.BigInteger, sa.ForeignKey("schedules.id"), nullable=False),
        sa.Column("status", sa.String(32), server_default="running"),
        sa.Column("attempt_number", sa.Integer, server_default="1"),
        sa.Column("error_message", sa.Text),
        sa.Column("result_summary", sa.JSON),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "application_events",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("event_type", sa.String(128), nullable=False),
        sa.Column("component", sa.String(64)),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("context", sa.JSON),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    for table, columns in {
        "model_versions": ["version", "status"],
        "predictions": ["transaction_id", "model_version_id", "predicted_label", "review_status", "created_at"],
        "etl_runs": ["dataset_source_id", "status"],
        "schedules": ["name", "job_type", "enabled"],
        "schedule_runs": ["schedule_id", "status"],
        "application_events": ["severity", "event_type", "component", "created_at"],
        "transactions": ["source_transaction_id"],
    }.items():
        for column in columns:
            op.create_index(op.f(f"ix_{table}_{column}"), table, [column])


def downgrade() -> None:
    for table in (
        "application_events",
        "schedule_runs",
        "schedules",
        "etl_runs",
        "training_runs",
        "predictions",
        "transactions",
        "model_versions",
        "dataset_sources",
    ):
        op.drop_table(table)
