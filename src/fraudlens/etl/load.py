"""Load processed transactions into PostgreSQL via a staging merge."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_rejected: int = 0
    rows_skipped: int = 0


def _quoted(columns: list[str]) -> str:
    return ", ".join(f'"{name}"' for name in columns)


def load_to_postgres(
    engine: Engine,
    frame: pd.DataFrame,
    target_table: str = "transaction_features",
    key_columns: list[str] | None = None,
    batch_size: int = 5000,
) -> LoadResult:
    """Stage, deduplicate and merge rows, then report the row counts.

    With ``key_columns`` the destination is deduplicated against the staging
    keys before inserting (row-value comparison, portable across PostgreSQL
    and SQLite), so a rerun does not create duplicates. Without keys the
    destination is replaced entirely: no keys means there is no safe merge
    contract, so a rerun rewrites the dataset from scratch.
    """
    result = LoadResult()
    if frame.empty:
        return result

    staging = f"{target_table}_staging"
    names = list(frame.columns)

    with engine.begin() as connection:
        connection.execute(text(f'DROP TABLE IF EXISTS "{staging}"'))
        frame.to_sql(staging, connection, index=False, if_exists="replace", chunksize=batch_size)

        if key_columns and set(key_columns) <= set(names):
            connection.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{target_table}" '
                    f'AS SELECT * FROM "{staging}" WHERE 1 = 0'
                )
            )
            keys = _quoted(key_columns)
            connection.execute(
                text(
                    f'DELETE FROM "{target_table}" '
                    f'WHERE ({keys}) IN (SELECT {keys} FROM "{staging}")'
                )
            )
        else:
            connection.execute(text(f'DROP TABLE IF EXISTS "{target_table}"'))
            connection.execute(
                text(
                    f'CREATE TABLE "{target_table}" '
                    f'AS SELECT * FROM "{staging}" WHERE 1 = 0'
                )
            )

        connection.execute(
            text(
                f'INSERT INTO "{target_table}" ({_quoted(names)}) '
                f'SELECT {_quoted(names)} FROM "{staging}"'
            )
        )
        result.rows_updated = len(frame)
        result.rows_inserted = int(connection.execute(
            text(f'SELECT count(*) FROM "{target_table}"')
        ).scalar_one())

        connection.execute(text(f'DROP TABLE "{staging}"'))

    logger.info("load_completed", extra={"rows_inserted": result.rows_inserted})
    return result


def count_rows(engine: Engine, target_table: str = "transaction_features") -> int:
    """Row-count reconciliation for the destination table."""
    with engine.begin() as connection:
        return int(connection.execute(
            text(f'SELECT count(*) FROM "{target_table}"')
        ).scalar_one())
