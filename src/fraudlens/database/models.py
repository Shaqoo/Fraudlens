"""SQLAlchemy ORM entities mirroring the FraudLens schema."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


class DatasetSource(Base):
    __tablename__ = "dataset_sources"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    dataset_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    source_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    etl_runs: Mapped[list["EtlRun"]] = relationship(back_populates="dataset_source")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    dataset_source_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_sources.id"))
    source_transaction_id: Mapped[str | None] = mapped_column(String(255), index=True)
    features: Mapped[dict] = mapped_column(JSON)
    actual_label: Mapped[int | None] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="transaction")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    artifact_uri: Mapped[str] = mapped_column(String(512))
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    evaluation_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    training_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_sources.id"))
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="model_version")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    dataset_source_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_sources.id"))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="training_runs")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    transaction_id: Mapped[int | None] = mapped_column(ForeignKey("transactions.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    fraud_probability: Mapped[float] = mapped_column(Float)
    predicted_label: Mapped[int] = mapped_column(SmallInteger, index=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    review_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    transaction: Mapped["Transaction | None"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")


class EtlRun(Base):
    __tablename__ = "etl_runs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    dataset_source_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_sources.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    rows_extracted: Mapped[int] = mapped_column(Integer, default=0)
    rows_processed: Mapped[int] = mapped_column(Integer, default=0)
    rows_loaded: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    dataset_source: Mapped[DatasetSource | None] = relationship(back_populates="etl_runs")


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    job_type: Mapped[str] = mapped_column(String(64), index=True)
    cron_expression: Mapped[str] = mapped_column(String(128))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    runs: Mapped[list["ScheduleRun"]] = relationship(back_populates="schedule")


class ScheduleRun(Base):
    __tablename__ = "schedule_runs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("schedules.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[str | None] = mapped_column(Text)
    result_summary: Mapped[dict | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    schedule: Mapped[Schedule] = relationship(back_populates="runs")


class ApplicationEvent(Base):
    __tablename__ = "application_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    component: Mapped[str | None] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
