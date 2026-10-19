"""Schema, label and completeness validation for raw transaction data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when the source data violates a required expectation."""


@dataclass
class ValidationReport:
    passed: bool
    row_count: int
    missing_columns: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    invalid_labels: int = 0

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "missing_columns": self.missing_columns,
            "duplicate_rows": self.duplicate_rows,
            "null_counts": self.null_counts,
            "invalid_labels": self.invalid_labels,
        }


def validate_dataframe(
    frame: pd.DataFrame,
    required_columns: list[str],
    target_column: str,
    valid_labels: tuple | set | None = None,
) -> ValidationReport:
    """Check required columns, duplicates, nulls and target label values."""
    valid_labels = set(valid_labels or {0, 1})
    missing = [name for name in required_columns if name not in frame.columns]

    null_counts = {name: int(frame[name].isna().sum()) for name in frame.columns}
    duplicate_rows = int(frame.duplicated().sum())

    invalid_labels = 0
    if target_column in frame.columns and not missing:
        labels = frame[target_column].dropna().unique().tolist()
        invalid_labels = int(sum(1 for label in labels if label not in valid_labels))

    report = ValidationReport(
        passed=not missing and invalid_labels == 0,
        row_count=int(len(frame)),
        missing_columns=missing,
        duplicate_rows=duplicate_rows,
        null_counts=null_counts,
        invalid_labels=invalid_labels,
    )

    if not report.passed:
        logger.warning(
            "validation_failed",
            extra={"missing_columns": missing, "invalid_labels": invalid_labels},
        )
    else:
        logger.info("validation_passed", extra={"row_count": report.row_count})

    return report
