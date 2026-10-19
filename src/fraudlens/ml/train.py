"""Train the XGBoost fraud classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraudlens.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: XGBClassifier
    feature_names: list[str]
    feature_medians: dict[str, float] = field(default_factory=dict)
    x_test: pd.DataFrame | None = field(default=None, repr=False)
    y_test: pd.Series | None = field(default=None, repr=False)
    train_size: int = 0
    test_size: int = 0
    positive_rate: float = 0.0


def train_model(
    frame: pd.DataFrame,
    target_column: str = "class",
    test_size: float = 0.2,
    random_state: int = 42,
    use_class_weight: bool = True,
    time_column: str | None = None,
) -> TrainingResult:
    """Split, fit and train the classifier.

    Preprocessing statistics are fitted on the training split only. When
    ``time_column`` is supplied the split is chronological rather than random.
    """
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} is not present in the dataset")

    features = frame.drop(columns=[target_column])
    target = frame[target_column].astype(int)

    positive_rate = float(target.mean())
    if not 0 < positive_rate < 1:
        raise ValueError("training data must contain both fraud and legitimate classes")

    if time_column and time_column in features.columns:
        ordered = frame.sort_values(time_column)
        features = ordered.drop(columns=[target_column])
        target = ordered[target_column].astype(int)
        split = int(len(ordered) * (1 - test_size))
        x_train, x_test = features.iloc[:split], features.iloc[split:]
        y_train, y_test = target.iloc[:split], target.iloc[split:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, stratify=target, random_state=random_state
        )

    medians = x_train.median(numeric_only=True).to_dict()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        scale_pos_weight=(1 / positive_rate - 1) if use_class_weight else 1.0,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    result = TrainingResult(
        model=model,
        feature_names=list(features.columns),
        feature_medians={k: float(v) for k, v in medians.items()},
        x_test=x_test,
        y_test=y_test,
        train_size=int(len(x_train)),
        test_size=int(len(x_test)),
        positive_rate=positive_rate,
    )
    logger.info(
        "training_completed",
        extra={
            "train_size": result.train_size,
            "test_size": result.test_size,
            "positive_rate": result.positive_rate,
        },
    )
    return result
