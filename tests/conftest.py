"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from xgboost import XGBClassifier


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small, deterministic dataset containing both classes."""
    rng = np.random.default_rng(42)
    size = 500
    frame = pd.DataFrame(
        {
            "amount": rng.normal(100, 40, size).round(2),
            "v1": rng.normal(0, 1, size).round(4),
            "v2": rng.normal(0, 1, size).round(4),
            "hour": rng.integers(0, 24, size),
        }
    )
    frame["class"] = (frame["amount"] > 140).astype(int)
    return frame


@pytest.fixture
def trained_model(sample_frame: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")
    model.fit(sample_frame.drop(columns=["class"]), sample_frame["class"])
    return model


@pytest.fixture
def sqlite_engine():
    """An in-memory SQLite engine with the FraudLens schema created."""
    from fraudlens.database.models import Base

    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
