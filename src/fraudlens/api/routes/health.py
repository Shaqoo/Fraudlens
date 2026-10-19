"""Health and model metadata routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from fraudlens.api.dependencies import check_database, get_registry
from fraudlens.api.schemas import HealthResponse, ModelInfoResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is responding."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(response: Response) -> ReadinessResponse:
    """Readiness: dependencies required to serve predictions are available."""
    database_connected = check_database()
    model_loaded = get_registry().active_version() is not None

    if not (database_connected and model_loaded):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if database_connected and model_loaded else "not_ready",
        model_loaded=model_loaded,
        database_connected=database_connected,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Metadata for the currently deployed model version."""
    try:
        _, metadata = get_registry().load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelInfoResponse(
        model_name="FraudLens XGBoost Classifier",
        model_version=metadata.version,
        model_type="XGBClassifier",
        threshold=metadata.threshold,
        status=metadata.status,
    )
