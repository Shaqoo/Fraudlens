"""Dataset registry describing every supported fraud-detection source."""

from __future__ import annotations

import pandas as pd

HUGGINGFACE = "huggingface"
KAGGLE = "kaggle"
SUPPORTED_SOURCES = {HUGGINGFACE, KAGGLE}


class UnknownSourceError(ValueError):
    """Raised when a dataset source is not registered."""


class DatasetSpec:
    """Static knowledge about one dataset: source, schema and preparation."""

    def __init__(
        self,
        source_type: str,
        default_id: str,
        label: str,
        target_column: str,
        drop_columns: tuple[str, ...],
        categorical_column: str | None = None,
        category_codes: dict[str, int] | None = None,
        code_column: str | None = None,
        file_name: str | None = None,
    ) -> None:
        self.source_type = source_type
        self.default_id = default_id
        self.label = label
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.categorical_column = categorical_column
        self.category_codes = category_codes or {}
        self.code_column = code_column
        self.file_name = file_name

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Feature names present in the prepared frame, minus identifier columns."""
        identifiers = self.drop_columns + [self.target_column]
        if self.categorical_column:
            identifiers.append(self.categorical_column)
        return [column for column in frame.columns if column not in identifiers]

    def unknown_categories(self, values) -> list[str]:
        allowed = {code.casefold() for code in self.category_codes}
        return sorted({str(value).casefold() for value in values} - allowed)

    def decode_category(self, code: int) -> str | None:
        for name, value in self.category_codes.items():
            if value == code:
                return name
        return None

    def target_meaning(self) -> dict[int, str]:
        return {0: "legitimate", 1: "fraudulent"}


CREDITCARD = DatasetSpec(
    source_type=HUGGINGFACE,
    default_id="David-Egea/Creditcard-fraud-detection",
    label="Credit Card Fraud Detection",
    target_column="class",
    drop_columns=("time",),
    file_name="creditcard.csv",
)

PAYSIM = DatasetSpec(
    source_type=KAGGLE,
    default_id="ealaxi/paysim1",
    label="PaySim Synthetic Financial Transactions",
    target_column="isfraud",
    drop_columns=("step", "nameorig", "namedest", "isflaggedfraud"),
    categorical_column="type",
    category_codes={"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4},
    code_column="type_code",
    file_name="PS_20174392719_1491204439457_log.csv",
)

DATASETS: dict[str, DatasetSpec] = {
    dataset.source_type: dataset
    for dataset in (CREDITCARD, PAYSIM)
}


def get_spec(source_type: str) -> DatasetSpec:
    """Look up the bound specification for a registered source type."""
    if source_type not in DATASETS:
        raise UnknownSourceError(
            f"unknown dataset source {source_type!r}; expected one of {sorted(DATASETS)}"
        )
    return DATASETS[source_type]
