"""Validation tests: required columns, duplicates, nulls and labels."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudlens.etl.validate import validate_dataframe


def test_validation_passes_on_clean_data(sample_frame: pd.DataFrame) -> None:
    report = validate_dataframe(sample_frame, required_columns=["amount", "class"], target_column="class")
    assert report.passed
    assert report.row_count == len(sample_frame)


def test_validation_detects_missing_column(sample_frame: pd.DataFrame) -> None:
    report = validate_dataframe(sample_frame, required_columns=["amount", "absent"], target_column="class")
    assert not report.passed
    assert "absent" in report.missing_columns


def test_validation_detects_invalid_labels(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.copy()
    frame.loc[0, "class"] = 7
    report = validate_dataframe(frame, required_columns=["class"], target_column="class")
    assert not report.passed
    assert report.invalid_labels == 1


def test_validation_counts_duplicates(sample_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([sample_frame, sample_frame.head(3)], ignore_index=True)
    report = validate_dataframe(duplicated, required_columns=["class"], target_column="class")
    assert report.duplicate_rows == 3


def test_validation_reports_null_counts(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.copy()
    frame.loc[range(5), "v1"] = None
    report = validate_dataframe(frame, required_columns=["class"], target_column="class")
    assert report.null_counts["v1"] == 5


@pytest.mark.parametrize("required", [[], ["class"]])
def test_report_serializes(sample_frame: pd.DataFrame, required: list[str]) -> None:
    report = validate_dataframe(sample_frame, required_columns=required, target_column="class")
    assert set(report.as_dict()) == {
        "passed",
        "row_count",
        "missing_columns",
        "duplicate_rows",
        "null_counts",
        "invalid_labels",
    }
