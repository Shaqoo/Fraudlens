"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.orm import Session

from fraudlens.database.connection import get_engine, get_session_factory
from fraudlens.ml.model_registry import ModelRegistry


def get_db() -> Generator[Session | None, None, None]:
    """Yield a session, or ``None`` when no database is configured.

    Route handlers decide whether a missing database is fatal (reads) or
    degradable (prediction persistence).
    """
    try:
        session = get_session_factory()()
    except Exception:  # noqa: BLE001 - degraded mode is handled by handlers
        yield None
        return
    try:
        yield session
    finally:
        session.close()


def check_database() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - health checks must never raise
        return False


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry()


def get_deployed_model() -> dict | None:
    """Load the active model bundle, or ``None`` when nothing is deployed.

    Handlers raise a 503 only when they actually need the model, so request
    validation still runs first.
    """
    try:
        bundle, metadata = get_registry().load()
    except Exception:  # noqa: BLE001 - no active model or unreadable artifact
        return None

    return {
        "model": bundle["model"],
        "metadata": metadata,
        "feature_names": bundle["feature_names"],
        "feature_medians": bundle.get("feature_medians", {}),
        "threshold": metadata.threshold,
        "version": metadata.version,
    }
