"""Prediction and prediction history routes."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from fraudlens.api.dependencies import get_db, get_deployed_model
from fraudlens.api.schemas import (
    AnalyzedPredictionResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionListResponse,
    PredictRequest,
    PredictResponse,
    ReviewUpdateRequest,
    TextAnalyzeRequest,
)
from fraudlens.config import get_settings
from fraudlens.database.models import ModelVersion, Prediction
from fraudlens.database.repositories import list_predictions, record_prediction, update_review_status
from fraudlens.ml.predict import MissingFeatureError, predict_batch, predict_single, score_partial
from fraudlens.ocr import (
    SUPPORTED_SUFFIXES,
    OcrError,
    TYPE_LABEL_PATTERN,
    parse_transaction_text,
    read_image_with_engine,
)

logger = logging.getLogger("fraudlens.api.predictions")

router = APIRouter(prefix="/predictions", tags=["predictions"])

NO_DB = "database unavailable"
NO_MODEL = "no active model is deployed"


def _require_model(deployed) -> dict:
    if deployed is None:
        raise HTTPException(status_code=503, detail=NO_MODEL)
    return deployed


def _to_response(prediction: Prediction, version: str, threshold: float) -> PredictResponse:
    return PredictResponse(
        prediction_id=prediction.id,
        fraud_probability=prediction.fraud_probability,
        predicted_label=prediction.predicted_label,
        risk_level="high" if prediction.predicted_label else "low",
        threshold=threshold,
        model_version=version,
        review_status=prediction.review_status,
    )


def _resolve_model_version(db: Session, version: str, threshold: float) -> ModelVersion:
    record = db.query(ModelVersion).filter(ModelVersion.version == version).one_or_none()
    if record is None:
        record = ModelVersion(version=version, artifact_uri="", threshold=threshold, status="active")
        db.add(record)
        db.flush()
    return record


def _persist(
    db: Session | None,
    deployed: dict,
    result: dict,
) -> int | None:
    """Persist the prediction, degrading gracefully when no database is available."""
    if not get_settings().persist_predictions or db is None:
        return None
    try:
        version = _resolve_model_version(db, deployed["version"], deployed["threshold"])
        return record_prediction(
            db,
            model_version_id=version.id,
            fraud_probability=result["fraud_probability"],
            predicted_label=result["predicted_label"],
        ).id
    except Exception as exc:  # noqa: BLE001 - serving must continue without storage
        logger.warning("prediction_persistence_failed", extra={"error": str(exc)})
        db.rollback()
        return None


@router.post("", response_model=PredictResponse, status_code=201)
def create_prediction(
    payload: PredictRequest,
    db: Session | None = Depends(get_db),
    deployed=Depends(get_deployed_model),
) -> PredictResponse:
    """Score a single transaction and persist the result when storage is available."""
    try:
        result = predict_single(
            _require_model(deployed)["model"],
            payload.features,
            deployed["feature_names"],
            deployed["threshold"],
            deployed["feature_medians"],
        )
    except MissingFeatureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction_id = _persist(db, deployed, result)
    return PredictResponse(
        prediction_id=prediction_id,
        transaction_id=payload.transaction_id,
        model_version=deployed["version"],
        review_status="pending" if prediction_id is not None else "not_recorded",
        **result,
    )


@router.post("/batch", response_model=BatchPredictResponse)
def create_batch_predictions(
    payload: BatchPredictRequest,
    deployed=Depends(get_deployed_model),
) -> BatchPredictResponse:
    """Score many transactions with one vectorized model call."""
    settings = get_settings()
    if len(payload.transactions) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f"maximum batch size is {settings.max_batch_size}")

    try:
        results = predict_batch(
            _require_model(deployed)["model"],
            [item.features for item in payload.transactions],
            deployed["feature_names"],
            deployed["threshold"],
            deployed["feature_medians"],
        )
    except MissingFeatureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return BatchPredictResponse(
        count=len(results),
        model_version=deployed["version"],
        predictions=[
            {**result, "transaction_id": item.transaction_id}
            for item, result in zip(payload.transactions, results, strict=True)
        ],
    )


# --------------------------------------------------------------------------
# Analyzed input: free text and image (OCR)
# --------------------------------------------------------------------------

_VERDICT = {0: "approved", 1: "flagged"}


def _analyzed_response(
    source: str,
    deployed: dict,
    parsed_fields: dict,
    values: dict,
    imputed: list[str],
    prediction: dict,
    prediction_id: int | None,
    transaction_id: str | None = None,
    raw_text: str | None = None,
    ocr_engine: str | None = None,
) -> AnalyzedPredictionResponse:
    return AnalyzedPredictionResponse(
        prediction_id=prediction_id,
        transaction_id=transaction_id,
        source=source,
        ocr_engine=ocr_engine,
        raw_text=raw_text,
        extracted_fields=parsed_fields,
        parsed_features=values,
        imputed_features=sorted(imputed),
        fraud_probability=prediction["fraud_probability"],
        predicted_label=prediction["predicted_label"],
        risk_level=prediction["risk_level"],
        status=_VERDICT[prediction["predicted_label"]],
        threshold=prediction["threshold"],
        model_version=deployed["version"],
    )


def _map_ocr_fields(deployed: dict, parsed_fields: dict) -> dict:
    """Map parsed text fields onto the deployed model's feature names."""
    feature_names = deployed["feature_names"]
    aliases = {
        "amount": "amount",
        "oldbalance": "oldbalanceorg",
        "newbalance": "newbalanceorig",
    }
    values: dict = {}
    for parsed_key, value in parsed_fields.items():
        if value is None:
            continue
        feature = aliases.get(parsed_key, parsed_key)
        if feature in feature_names:
            values[feature] = value
    return values


def _analyze(
    source: str,
    text: str,
    deployed: dict,
    db: Session | None,
    transaction_id: str | None = None,
    ocr_engine: str | None = None,
) -> AnalyzedPredictionResponse:
    """Shared pipeline: parse text -> map features -> impute -> score -> persist."""
    parsed_fields = parse_transaction_text(text)

    if parsed_fields.get("amount") is None and parsed_fields.get("type_code") is None:
        raise HTTPException(
            status_code=422,
            detail="could not parse an amount or transaction type from the input",
        )
    unknown = parsed_fields.get("type_code") is None and TYPE_LABEL_PATTERN.search(text or "")
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unrecognized transaction type: {unknown.group(1)!r}",
        )

    values = _map_ocr_fields(deployed, parsed_fields)
    prediction, imputed = score_partial(
        deployed["model"],
        values,
        deployed["feature_names"],
        deployed["threshold"],
        deployed["feature_medians"],
    )

    prediction_id = _persist(db, deployed, prediction)
    return _analyzed_response(
        source=source,
        deployed=deployed,
        parsed_fields={key: value for key, value in parsed_fields.items()},
        values=values,
        imputed=imputed,
        prediction=prediction,
        prediction_id=prediction_id or None,
        transaction_id=transaction_id,
        raw_text=text or None,
        ocr_engine=ocr_engine,
    )


@router.post("/text", response_model=AnalyzedPredictionResponse, status_code=201)
def analyze_text(
    payload: TextAnalyzeRequest,
    db: Session | None = Depends(get_db),
    deployed=Depends(get_deployed_model),
) -> AnalyzedPredictionResponse:
    """Score a transaction described in free text (Amount, Type, balances)."""
    return _analyze(
        source="text",
        text=payload.text,
        deployed=_require_model(deployed),
        db=db,
        transaction_id=payload.transaction_id,
    )


@router.post("/image", response_model=AnalyzedPredictionResponse, status_code=201)
async def analyze_image(
    file: UploadFile = File(...),
    transaction_id: str | None = None,
    db: Session | None = Depends(get_db),
    deployed=Depends(get_deployed_model),
) -> AnalyzedPredictionResponse:
    """Score a transaction screenshot: OCR the image, parse fields, run the model."""
    uploaded = _require_model(deployed)
    settings = get_settings()

    data = await file.read()
    if len(data) > settings.max_extract_bytes:
        raise HTTPException(status_code=413, detail=f"maximum upload size is {settings.max_extract_mb} MB")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"unsupported image extension: {suffix or '(none)'}")

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, dir=settings.raw_dir, delete=False) as handle:
        handle.write(data)
        image_path = Path(handle.name)

    try:
        try:
            raw_text, ocr_engine = read_image_with_engine(image_path)
        except (OcrError, ImageTooLargeError, UnsupportedImageError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return _analyze(
            source="image",
            text=raw_text,
            deployed=deployed,
            db=db,
            transaction_id=transaction_id,
            ocr_engine=ocr_engine,
        )
    finally:
        image_path.unlink(missing_ok=True)


@router.get("", response_model=PredictionListResponse)
def list_prediction_records(
    predicted_label: int | None = Query(None, ge=0, le=1),
    min_probability: float | None = Query(None, ge=0.0, le=1.0),
    review_status: str | None = Query(None),
    model_version: str | None = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session | None = Depends(get_db),
) -> PredictionListResponse:
    """Paginated, filterable prediction history."""
    if db is None:
        raise HTTPException(status_code=503, detail=NO_DB)

    total, rows = list_predictions(
        db,
        predicted_label=predicted_label,
        min_probability=min_probability,
        review_status=review_status,
        model_version=model_version,
        limit=limit,
        offset=offset,
    )
    return PredictionListResponse(total=total, limit=limit, offset=offset, items=rows)


@router.get("/{prediction_id}", response_model=PredictResponse)
def get_prediction(
    prediction_id: int,
    db: Session | None = Depends(get_db),
) -> PredictResponse:
    """Retrieve a single stored prediction."""
    if db is None:
        raise HTTPException(status_code=503, detail=NO_DB)

    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="prediction not found")

    version = db.get(ModelVersion, prediction.model_version_id)
    return _to_response(
        prediction,
        version.version if version else "unknown",
        version.threshold if version else 0.0,
    )


@router.patch("/{prediction_id}/review", response_model=PredictResponse)
def review_prediction(
    prediction_id: int,
    payload: ReviewUpdateRequest,
    db: Session | None = Depends(get_db),
) -> PredictResponse:
    """Update the review outcome. Requires authentication before deployment."""
    if db is None:
        raise HTTPException(status_code=503, detail=NO_DB)

    prediction = update_review_status(db, prediction_id, payload.review_status, payload.review_note)
    if prediction is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    db.commit()
    return get_prediction(prediction_id, db)
