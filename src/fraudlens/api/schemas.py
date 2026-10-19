"""Pydantic request and response models for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REVIEW_STATUSES = {"pending", "confirmed_fraud", "legitimate", "inconclusive"}


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    threshold: float
    status: str


class TextAnalyzeRequest(BaseModel):
    """Free-text transaction input, e.g. 'Amount 2350.00, Type TRANSFER, oldbal 12000'."""

    text: str = Field(min_length=1, max_length=2000)
    transaction_id: str | None = None


class AnalyzedPredictionResponse(BaseModel):
    """Response shared by the OCR (image) and free-text (text) endpoints."""

    prediction_id: int | None = None
    transaction_id: str | None = None
    source: str              # image | text
    ocr_engine: str | None = None
    raw_text: str | None = None
    extracted_fields: dict[str, float | int | str | None] = {}
    parsed_features: dict[str, Any] = {}
    imputed_features: list[str] = []
    fraud_probability: float
    predicted_label: int
    risk_level: str
    status: str              # flagged | approved
    threshold: float
    model_version: str
    review_status: str = "pending"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = None
    features: dict[str, Any]


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[PredictRequest] = Field(min_length=1)


class PredictionItem(BaseModel):
    transaction_id: str | None = None
    fraud_probability: float
    predicted_label: int
    risk_level: str | None = None


class PredictResponse(PredictionItem):
    prediction_id: int | None = None
    threshold: float
    model_version: str
    review_status: str = "pending"


class BatchPredictResponse(BaseModel):
    count: int
    model_version: str
    predictions: list[PredictionItem]


class PredictionListItem(PredictionItem):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_version: str
    review_status: str
    created_at: datetime


class PredictionListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionListItem]


class ReviewUpdateRequest(BaseModel):
    review_status: str = Field(pattern="^(pending|confirmed_fraud|legitimate|inconclusive)$")
    review_note: str | None = None


class EtlRunRequest(BaseModel):
    source_type: str | None = Field(default=None, pattern="^(huggingface|kaggle)$")
    dataset_id: str | None = None


class EtlRunResponse(BaseModel):
    run_id: int
    status: str
    message: str | None = None
    rows_extracted: int | None = None
    rows_processed: int | None = None
    rows_loaded: int | None = None
    rows_rejected: int | None = None


class ScheduleCreateRequest(BaseModel):
    name: str
    job_type: str = Field(pattern="^(etl|model_training|batch_prediction|data_quality_check)$")
    cron_expression: str
    timezone: str = "UTC"
    enabled: bool = True


class ScheduleUpdateRequest(BaseModel):
    cron_expression: str | None = None
    timezone: str | None = None
    enabled: bool | None = None


class ScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    schedule_id: int = Field(validation_alias="id")
    name: str
    job_type: str
    cron_expression: str
    timezone: str
    enabled: bool


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    predictions_total: int
    predictions_flagged: int
    etl_success_rate: float | None = None
