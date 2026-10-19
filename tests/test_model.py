"""Model training, evaluation and registry tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.ml.evaluate import evaluate_model, select_threshold
from fraudlens.ml.model_registry import ModelRegistry, ModelVersion
from fraudlens.ml.predict import MissingFeatureError, predict_batch, predict_single
from fraudlens.ml.train import train_model


def test_training_dataset_contains_both_classes(sample_frame: pd.DataFrame) -> None:
    assert set(sample_frame["class"].unique()) == {0, 1}


def test_training_rejects_single_class_data(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        train_model(sample_frame.assign(**{"class": 0}), target_column="class")


def test_training_splits_data_without_leakage(sample_frame: pd.DataFrame) -> None:
    result = train_model(sample_frame, target_column="class", test_size=0.2)
    assert result.train_size + result.test_size == len(sample_frame)
    assert "class" not in result.feature_names
    assert 0 < result.positive_rate < 1


def test_training_rejects_missing_target(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        train_model(sample_frame, target_column="fraud")


def test_training_uses_chronological_split_when_time_column_present(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.assign(ts=np.arange(len(sample_frame)))
    result = train_model(frame, target_column="class", test_size=0.25, time_column="ts")
    assert result.test_size == int(len(frame) * 0.25)


def test_evaluation_reports_expected_metrics(sample_frame: pd.DataFrame, trained_model) -> None:
    features = sample_frame.drop(columns=["class"])
    metrics = evaluate_model(trained_model, features, sample_frame["class"], threshold=0.5)

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert set(metrics["confusion_matrix"]) == {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    }


def test_threshold_selection_respects_recall_floor(sample_frame: pd.DataFrame, trained_model) -> None:
    features = sample_frame.drop(columns=["class"])
    threshold, metrics = select_threshold(trained_model, features, sample_frame["class"], min_recall=0.5)
    assert 0.0 < threshold < 1.0
    assert metrics.get("recall", 1.0) >= 0.5


def test_predict_single_returns_risk_band(sample_frame: pd.DataFrame, trained_model) -> None:
    features = sample_frame.drop(columns=["class"]).iloc[0].to_dict()
    result = predict_single(trained_model, features, list(sample_frame.drop(columns=["class"]).columns), 0.5)
    assert result["predicted_label"] in {0, 1}
    assert result["risk_level"] in {"low", "medium", "high"}


def test_predict_single_rejects_missing_features(trained_model) -> None:
    with pytest.raises(MissingFeatureError):
        predict_single(trained_model, {"amount": 1.0}, ["amount", "v1", "v2", "hour"], 0.5)


def test_predict_batch_is_vectorized(sample_frame: pd.DataFrame, trained_model) -> None:
    columns = list(sample_frame.drop(columns=["class"]).columns)
    records = sample_frame.head(10).drop(columns=["class"]).to_dict("records")
    results = predict_batch(trained_model, records, columns, 0.5)
    assert len(results) == 10


def test_predict_batch_handles_empty_input(trained_model) -> None:
    assert predict_batch(trained_model, [], ["amount"], 0.5) == []


def test_registry_round_trip(tmp_path, trained_model, sample_frame: pd.DataFrame) -> None:
    registry = ModelRegistry(root=tmp_path)
    version = ModelVersion(version="xgb-v1", artifact_path=tmp_path / "xgb-v1.joblib", threshold=0.42)
    registry.save({"model": trained_model, "feature_names": ["amount"]}, version)

    saved = next(m for m in registry.versions() if m.version == "xgb-v1")
    assert saved.status == "candidate"
    assert saved.threshold == 0.42

    bundle, metadata = registry.load("xgb-v1")
    assert metadata.threshold == 0.42
    assert bundle["feature_names"] == ["amount"]
    assert bundle["model"].n_features_in_ == trained_model.n_features_in_

    registry.promote("xgb-v1")
    assert registry.active_version() == "xgb-v1"


def test_registry_promotion_retires_previous(tmp_path, trained_model) -> None:
    registry = ModelRegistry(root=tmp_path)
    for name in ("xgb-v1", "xgb-v2"):
        registry.save({"model": trained_model}, ModelVersion(version=name, artifact_path=tmp_path / f"{name}.joblib"))

    registry.promote("xgb-v1")
    assert registry.active_version() == "xgb-v1"

    registry.promote("xgb-v2")
    assert registry.active_version() == "xgb-v2"
    assert next(m for m in registry.versions() if m.version == "xgb-v1").status == "retired"


def test_registry_rejects_invalid_status() -> None:
    with pytest.raises(ValueError):
        ModelVersion(version="x", artifact_path="x", status="bogus").validate()


def test_registry_raises_for_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelRegistry(root=tmp_path).load("does-not-exist")
