"""Clean and prepare validated transaction records for storage and training."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraudlens.config import get_settings
from fraudlens.etl.datasets import DatasetSpec
from fraudlens.etl.validate import validate_dataframe

logger = logging.getLogger(__name__)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and trim every column name."""
    return frame.rename(columns=lambda name: str(name).strip().lower())


def prepare_dataset(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Prepare one dataset according to its spec: drop identifiers, encode categories."""
    prepared = normalize_columns(frame)
    prepared = prepared.drop(columns=[c for c in spec.drop_columns if c in prepared.columns])

    if spec.categorical_column and spec.categorical_column in prepared.columns:
        values = prepared[spec.categorical_column].dropna().unique().tolist()
        unknown = spec.unknown_categories(values)
        if unknown:
            raise ValueError(f"unknown category values in {spec.categorical_column}: {unknown}")

        missing = int(prepared[spec.categorical_column].isna().sum())
        code_name = spec.code_column or f"{spec.categorical_column}_code"
        codes = prepared[spec.categorical_column].map(
            lambda value: spec.category_codes[str(value).upper().replace(" ", "_")]
        )
        position = list(prepared.columns).index(spec.categorical_column)
        prepared = prepared.drop(columns=[spec.categorical_column])
        prepared.insert(position, code_name, codes.astype("Int64"))
        if missing:
            logger.warning("categorical_values_missing", extra={"column": spec.categorical_column, "count": missing})

    frame_columns = list(prepared.columns)
    for column in frame_columns:
        if column == spec.target_column:
            continue
        if str(prepared[column].dtype) in ("object", "string"):
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    feature_names = [column for column in prepared.columns if column != spec.target_column]
    rejected = int(prepared[feature_names].isna().any(axis=1).sum())
    if rejected:
        logger.warning("rows_rejected_unparseable_features", extra={"count": rejected})
    prepared = prepared.dropna(subset=feature_names)

    prepared[spec.target_column] = pd.to_numeric(prepared[spec.target_column], errors="coerce")
    prepared = prepared.dropna(subset=[spec.target_column])
    prepared[spec.target_column] = prepared[spec.target_column].astype("int8")

    before = len(prepared)
    prepared = prepared.drop_duplicates()
    dropped = before - len(prepared)
    if dropped:
        logger.info("duplicates_removed", extra={"count": int(dropped)})

    if spec.categorical_column:
        code_name = spec.code_column or f"{spec.categorical_column}_code"
        features = prepared.dropna(subset=[code_name])
        dropped = len(prepared) - len(features)
        if dropped:
            logger.warning("rows_dropped_unknown_category", extra={"count": int(dropped)})
        features[code_name] = features[code_name].astype("int8")
        prepared = features

    logger.info("dataset_prepared", extra={"row_count": int(len(prepared)), "dataset": spec.label})
    return prepared.reset_index(drop=True)


def validate_and_prepare(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    """Validate raw input, then prepare it; returns the prepared frame and report."""
    normalized = normalize_columns(frame)
    required = [spec.target_column]
    if spec.categorical_column:
        required.append(spec.categorical_column)

    report = validate_dataframe(normalized, required, spec.target_column, valid_labels={0, 1})
    if not report.passed:
        return prepared_empty(spec), report.as_dict()

    return prepare_dataset(normalized, spec), report.as_dict()


def prepared_empty(spec: DatasetSpec) -> pd.DataFrame:
    return pd.DataFrame(columns=[spec.target_column])


def save_processed(frame: pd.DataFrame, destination: Path | None = None) -> Path:
    """Write the processed dataset to Parquet."""
    destination = Path(destination or get_settings().processed_dir / "transactions.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    logger.info("processed_saved", extra={"path": str(destination)})
    return destination
