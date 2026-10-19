"""Evaluation metrics and decision threshold selection."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    """Compute imbalance-aware metrics plus a supplementary accuracy figure."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy_supplementary": float((predictions == np.asarray(y_test)).mean()),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
    }
    logger.info("evaluation_completed", extra={"threshold": float(threshold)})
    return metrics


def select_threshold(
    model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    min_recall: float = 0.80,
) -> tuple[float, dict]:
    """Pick the lowest threshold on validation data meeting a recall floor."""
    probabilities = model.predict_proba(x_validation)[:, 1]
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold, best_score, best_metrics = 0.5, -np.inf, {}

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        recall = float(recall_score(y_validation, predictions, zero_division=0))
        precision = float(precision_score(y_validation, predictions, zero_division=0))
        if recall < min_recall:
            continue
        score = precision if precision > 0 else -np.inf
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
            best_metrics = {"recall": recall, "precision": precision, "f1": float(
                f1_score(y_validation, predictions, zero_division=0)
            )}

    logger.info(
        "threshold_selected",
        extra={"threshold": best_threshold, "min_recall": min_recall, **best_metrics},
    )
    return best_threshold, best_metrics
