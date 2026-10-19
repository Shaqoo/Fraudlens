"""Inference helpers shared by the API and the batch scoring job."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MissingFeatureError(ValueError):
    """Raised when a request omits features required by the deployed model."""


def _build_row(features: dict, feature_names: list[str]) -> dict:
    """Validate one request against the model's feature schema and order it."""
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise MissingFeatureError(f"missing required features: {', '.join(sorted(missing))}")
    return {name: features[name] for name in feature_names}


def _risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "high"
    if probability >= 0.5:
        return "medium"
    return "low"


def predict_single(
    model,
    features: dict,
    feature_names: list[str],
    threshold: float,
    medians: dict | None = None,
) -> dict:
    """Score one transaction and return the threshold-based classification."""
    frame = pd.DataFrame([_build_row(features, feature_names)], columns=feature_names)
    if medians:
        frame = frame.fillna(medians)

    probability = float(model.predict_proba(frame)[0, 1])
    result = {
        "fraud_probability": probability,
        "predicted_label": int(probability >= threshold),
        "risk_level": _risk_level(probability),
        "threshold": float(threshold),
    }
    logger.info("prediction_completed", extra={"fraud_probability": probability})
    return result


def predict_batch(
    model,
    records: list[dict],
    feature_names: list[str],
    threshold: float,
    medians: dict | None = None,
) -> list[dict]:
    """Score many transactions with a single vectorized model call."""
    if not records:
        return []

    rows = [_build_row(record, feature_names) for record in records]
    frame = pd.DataFrame(rows, columns=feature_names)
    if medians:
        frame = frame.fillna(medians)

    probabilities = model.predict_proba(frame)[:, 1]
    return [
        {
            "fraud_probability": float(probability),
            "predicted_label": int(probability >= threshold),
            "risk_level": _risk_level(float(probability)),
        }
        for probability in probabilities
    ]


def score_partial(
    model,
    values: dict,
    feature_names: list[str],
    threshold: float,
    medians: dict | None = None,
) -> tuple[dict, list[str]]:
    """Score one transaction even when some features were not supplied.

    Used by the OCR and free-text endpoints: fields that were not extracted
    (for example a missing balance) are imputed with the training medians.
    Returns ``(prediction, imputed_feature_names)``.
    """
    medians = medians or {}
    imputed = [name for name in feature_names if name not in values]

    row = {name: values.get(name, medians.get(name, 0.0)) for name in feature_names}
    frame = pd.DataFrame([row], columns=feature_names)

    probability = float(model.predict_proba(frame)[0, 1])
    prediction = {
        "fraud_probability": probability,
        "predicted_label": int(probability >= threshold),
        "risk_level": _risk_level(probability),
        "threshold": float(threshold),
    }
    logger.info(
        "prediction_completed",
        extra={"fraud_probability": probability, "imputed_features": len(imputed)},
    )
    return prediction, imputed
