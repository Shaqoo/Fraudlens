"""ETL tests: dataset preparation, transformation and staging load counts."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudlens.etl.datasets import CREDITCARD, PAYSIM, get_spec
from fraudlens.etl.load import LoadResult, load_to_postgres
from fraudlens.etl.transform import normalize_columns, prepare_dataset, validate_and_prepare


@pytest.fixture
def paysim_frame() -> pd.DataFrame:
    """A tiny PaySim-shaped sample containing both classes."""
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 3],
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [5000.0, 120.0, 300.0, 15000.0],
            "nameOrig": ["C1", "C2", "C3", "C4"],
            "oldbalanceOrg": [10000.0, 900.0, 800.0, 0.0],
            "newbalanceOrig": [5000.0, 780.0, 500.0, 0.0],
            "nameDest": ["M1", "M2", "C5", "M3"],
            "oldbalanceDest": [0.0, 10.0, 0.0, 100.0],
            "newbalanceDest": [5000.0, 130.0, 0.0, 15100.0],
            "isFraud": [1, 0, 0, 1],
            "isFlaggedFraud": [0, 0, 0, 0],
        }
    )


def test_dataset_registry_contains_both_sources() -> None:
    from fraudlens.etl.datasets import UnknownSourceError

    assert set(get_spec("kaggle").category_codes) == {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    assert get_spec("huggingface").target_column == "class"
    with pytest.raises(UnknownSourceError):
        get_spec("data.world")


def test_prepare_dataset_normalizes_and_encodes(paysim_frame: pd.DataFrame) -> None:
    prepared = prepare_dataset(paysim_frame, PAYSIM)

    assert "type_code" in prepared.columns
    assert "type" not in prepared.columns
    assert prepared["type_code"].tolist() == [4, 3, 1, 4]


def test_prepare_dataset_drops_identifier_columns(paysim_frame: pd.DataFrame) -> None:
    prepared = prepare_dataset(paysim_frame, PAYSIM)
    for column in ("step", "nameorig", "namedest", "isflaggedfraud"):
        assert column not in prepared.columns
    assert list(prepared.columns) == ["type_code", "amount", "oldbalanceorg", "newbalanceorig", "oldbalancedest", "newbalancedest", "isfraud"]


def test_prepare_dataset_rejects_unknown_type(paysim_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        prepare_dataset(paysim_frame.assign(type=["TRANSFER", "ROBBERY", None, None]), PAYSIM)


def test_prepare_dataset_coerces_numeric_and_target(paysim_frame: pd.DataFrame) -> None:
    frame = paysim_frame.copy()
    frame["amount"] = frame["amount"].astype("object")
    frame.loc[0, "amount"] = "5000.00"
    prepared = prepare_dataset(frame, PAYSIM)
    assert prepared["amount"].iloc[0] == 5000.0
    assert prepared["isfraud"].dtype == "int8"


def test_prepare_dataset_drops_unparseable_amount(paysim_frame: pd.DataFrame) -> None:
    frame = paysim_frame.copy()
    frame["amount"] = frame["amount"].astype("object")
    frame.loc[0, "amount"] = "5,000.00"  # thousands separators are not coerced; the row is dropped
    prepared = prepare_dataset(frame, PAYSIM)
    assert len(prepared) == len(paysim_frame) - 1


def test_normalize_columns_trims_and_lowercases() -> None:
    frame = pd.DataFrame({" Amount ": [1.0], "Type": ["TRANSFER"]})
    normalized = normalize_columns(frame)
    assert list(normalized.columns) == ["amount", "type"]


def test_validate_and_prepare_reports_missing_columns(paysim_frame: pd.DataFrame) -> None:
    prepared, report = validate_and_prepare(paysim_frame.drop(columns=["type"]), PAYSIM)
    assert not report["passed"]
    assert "type" in report["missing_columns"]
    assert prepared.empty


def test_validate_and_prepare_drops_duplicates(paysim_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([paysim_frame, paysim_frame.head(1)], ignore_index=True)
    prepared, report = validate_and_prepare(duplicated, PAYSIM)
    assert report["passed"]
    assert len(prepared) == len(paysim_frame)


def test_creditcard_preparation() -> None:
    frame = pd.DataFrame(
        {
            "Time": [0.0, 1.0],
            "V1": [0.1, -0.2],
            "V2": [0.5, 0.4],
            "V3": [1.0, 2.0],
            "Amount": [10.0, 250.0],
            "Class": [0, 1],
        }
    )
    prepared = prepare_dataset(frame, CREDITCARD)
    assert "time" not in prepared.columns
    assert set(prepared.columns) == {"v1", "v2", "v3", "amount", "class"}
    assert prepared["class"].tolist() == [0, 1]


def test_load_result_defaults() -> None:
    assert LoadResult().rows_inserted == 0


def test_load_is_a_noop_for_empty_frames(sqlite_engine) -> None:
    result = load_to_postgres(sqlite_engine, pd.DataFrame({"amount": []}))
    assert result.rows_inserted == 0


def test_load_stage_is_removed(sqlite_engine, paysim_frame: pd.DataFrame) -> None:
    from sqlalchemy import inspect, text

    prepared = prepare_dataset(paysim_frame, PAYSIM)
    load_to_postgres(sqlite_engine, prepared, target_table="etl_transactions")

    tables = inspect(sqlite_engine).get_table_names()
    assert "etl_transactions" in tables
    assert "etl_transactions_staging" not in tables
    with sqlite_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM etl_transactions")).scalar_one() == len(prepared)


def test_load_replaces_destination_without_keys(sqlite_engine, paysim_frame: pd.DataFrame) -> None:
    prepared = prepare_dataset(paysim_frame, PAYSIM)
    load_to_postgres(sqlite_engine, prepared, target_table="etl_transactions")
    result = load_to_postgres(sqlite_engine, prepared.head(2), target_table="etl_transactions")

    assert result.rows_inserted == 2


def test_load_with_key_columns_avoids_duplicates(sqlite_engine, paysim_frame: pd.DataFrame) -> None:
    from sqlalchemy import text

    prepared = prepare_dataset(paysim_frame, PAYSIM).reset_index().rename(columns={"index": "txn_id"})
    load_to_postgres(sqlite_engine, prepared, target_table="etl_transactions", key_columns=["txn_id"])

    updated = prepared.copy()
    updated.loc[0, "amount"] = 999.0
    result = load_to_postgres(sqlite_engine, updated, target_table="etl_transactions", key_columns=["txn_id"])

    assert result.rows_inserted == len(prepared)
    with sqlite_engine.connect() as connection:
        values = [row[0] for row in connection.execute(text("SELECT amount FROM etl_transactions ORDER BY txn_id"))]
    assert values[0] == 999.0
