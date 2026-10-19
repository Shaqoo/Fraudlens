"""Job handlers executed by the scheduler."""

from __future__ import annotations

import json
import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

JOB_TYPES: dict[str, Callable[..., dict]] = {}


def _register(name: str):
    def decorator(func):
        JOB_TYPES[name] = func
        return func

    return decorator


@_register("etl")
def run_etl_job(source_type: str | None = None, dataset_id: str | None = None) -> dict:
    """Extract, validate, transform and load. Returns a row-count summary."""
    from fraudlens.config import get_settings as _settings
    from fraudlens.database.connection import get_engine
    from fraudlens.etl.datasets import get_spec
    from fraudlens.etl.extract import extract_dataset
    from fraudlens.etl.load import load_to_postgres
    from fraudlens.etl.transform import save_processed, validate_and_prepare

    extraction = extract_dataset(source_type=source_type, dataset_id=dataset_id)
    spec = get_spec(extraction.source_type)

    prepared, report = validate_and_prepare(extraction.frame, spec)
    if not report["passed"]:
        return {
            "status": "failed",
            "rows_extracted": extraction.rows_extracted,
            "error_message": f"validation failed: missing={report['missing_columns']}",
        }

    save_processed(prepared)
    settings = _settings()
    metadata_path = settings.processed_dir / "dataset_meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_type": extraction.source_type,
                "dataset_key": extraction.dataset_key,
                "target_column": spec.target_column,
                "feature_columns": spec.feature_columns(prepared),
                "categorical_column": spec.categorical_column,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = load_to_postgres(get_engine(), prepared, target_table="transaction_features")

    summary = {
        "status": "succeeded",
        "dataset_key": extraction.dataset_key,
        "source_type": extraction.source_type,
        "target_column": spec.target_column,
        "feature_columns": spec.feature_columns(prepared),
        "rows_extracted": extraction.rows_extracted,
        "rows_processed": report["row_count"],
        "rows_loaded": result.rows_inserted,
        "rows_rejected": report["duplicate_rows"],
    }
    logger.info("etl_job_completed", extra=summary)
    return summary


@_register("model_training")
def run_training_job(source_type: str | None = None) -> dict:
    """Train a candidate model from the processed dataset."""
    from fraudlens.config import get_settings
    from fraudlens.etl.datasets import KAGGLE, get_spec
    from fraudlens.ml.evaluate import evaluate_model
    from fraudlens.ml.model_registry import ModelRegistry, ModelVersion
    from fraudlens.ml.train import train_model

    settings = get_settings()
    processed = settings.processed_dir / "transactions.parquet"
    if not processed.exists():
        return {"status": "failed", "error_message": f"processed dataset not found: {processed}"}

    meta_path = settings.processed_dir / "dataset_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    source_type = source_type or metadata.get("source_type") or KAGGLE
    spec = get_spec(source_type)

    frame = pd.read_parquet(processed)
    target = metadata.get("target_column") or spec.target_column
    if target not in frame.columns:
        return {"status": "failed", "error_message": f"target column {target!r} missing from processed dataset"}

    result = train_model(frame, target_column=target)
    metrics = evaluate_model(result.model, result.x_test, result.y_test)

    version = ModelVersion(
        version=f"xgb-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
        artifact_path=settings.model_path,
        threshold=metrics["threshold"],
        feature_names=result.feature_names,
        feature_medians=result.feature_medians,
        evaluation_metrics={**metrics, "dataset": spec.label, "source_type": source_type},
        status="candidate",
    )
    registry = ModelRegistry()
    bundle = {
        "model": result.model,
        "feature_names": result.feature_names,
        "feature_medians": result.feature_medians,
    }
    registry.save(bundle, version)

    summary = {
        "status": "succeeded",
        "version": version.version,
        "source_type": source_type,
        "dataset": spec.label,
        "feature_count": len(result.feature_names),
        "metrics": {
            key: metrics[key]
            for key in ("precision", "recall", "f1", "average_precision", "threshold")
            if key in metrics
        },
    }
    logger.info("training_job_completed", extra={"version": version.version})
    return summary


@_register("batch_prediction")
def run_batch_prediction_job(limit: int = 10_000) -> dict:
    """Score recently ingested transactions that have no prediction yet."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from fraudlens.api.dependencies import get_deployed_model
    from fraudlens.database.connection import get_engine
    from fraudlens.database.models import ModelVersion, Prediction, Transaction
    from fraudlens.database.repositories import record_prediction
    from fraudlens.ml.predict import predict_batch

    deployed = get_deployed_model()
    engine = get_engine()
    scored = 0

    with Session(engine) as session:
        unscored = session.execute(
            select(Transaction)
            .outerjoin(Prediction, Prediction.transaction_id == Transaction.id)
            .where(Prediction.id.is_(None))
            .limit(limit)
        ).scalars().all()
        if not unscored:
            return {"status": "succeeded", "rows_scored": 0}

        records = [row.features for row in unscored]
        results = predict_batch(
            deployed["model"], records, deployed["feature_names"], deployed["threshold"], deployed["feature_medians"]
        )

        version = session.query(ModelVersion).filter(ModelVersion.version == deployed["version"]).one_or_none()
        if version is None:
            version = ModelVersion(
                version=deployed["version"],
                artifact_uri=str(deployed["metadata"].artifact_path),
                threshold=deployed["threshold"],
                status="active",
            )
            session.add(version)
            session.flush()

        for transaction, result in zip(unscored, results, strict=True):
            record_prediction(
                session,
                model_version_id=version.id,
                fraud_probability=result["fraud_probability"],
                predicted_label=result["predicted_label"],
                transaction_id=transaction.id,
            )
            scored += 1
        session.commit()

    logger.info("batch_prediction_job_completed", extra={"rows_scored": scored})
    return {"status": "succeeded", "rows_scored": scored}


@_register("data_quality_check")
def run_data_quality_job() -> dict:
    """Re-validate the processed dataset against the expected schema."""
    from fraudlens.config import get_settings
    from fraudlens.etl.datasets import KAGGLE, get_spec
    from fraudlens.etl.validate import validate_dataframe

    settings = get_settings()
    processed = settings.processed_dir / "transactions.parquet"
    if not processed.exists():
        return {"status": "failed", "error_message": f"processed dataset not found: {processed}"}

    meta_path = settings.processed_dir / "dataset_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    spec = get_spec(metadata.get("source_type") or KAGGLE)
    target = metadata.get("target_column") or spec.target_column

    frame = pd.read_parquet(processed)
    report = validate_dataframe(frame, required_columns=[target], target_column=target)
    summary = {
        "status": "succeeded" if report.passed else "partial",
        "dataset": metadata.get("dataset_key"),
        "target_column": target,
        **report.as_dict(),
    }
    logger.info("data_quality_job_completed", extra={"passed": report.passed})
    return summary
