"""Extract fraud datasets from Hugging Face and Kaggle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraudlens.config import get_settings
from fraudlens.etl.datasets import HUGGINGFACE, KAGGLE, DatasetSpec, get_spec

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Raised when a source dataset cannot be retrieved."""


@dataclass(frozen=True)
class ExtractionResult:
    frame: pd.DataFrame
    dataset_key: str
    source_type: str
    raw_path: Path
    rows_extracted: int

    def as_dict(self) -> dict:
        return {
            "dataset_key": self.dataset_key,
            "source_type": self.source_type,
            "raw_path": str(self.raw_path),
            "rows_extracted": self.rows_extracted,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }


def extract_dataset(
    source_type: str | None = None,
    dataset_id: str | None = None,
) -> ExtractionResult:
    """Download the requested dataset and return it as a raw DataFrame."""
    source_type = source_type or KAGGLE
    spec: DatasetSpec = get_spec(source_type)
    dataset_id = dataset_id or spec.default_id

    frame, raw_path = _extract_source(source_type, spec, dataset_id)

    result = ExtractionResult(
        frame=frame,
        dataset_key=dataset_id,
        source_type=source_type,
        raw_path=raw_path,
        rows_extracted=int(len(frame)),
    )
    logger.info("extraction_completed", extra=result.as_dict())
    return result


def _extract_source(source_type: str, spec: DatasetSpec, dataset_id: str) -> tuple[pd.DataFrame, Path]:
    if source_type == HUGGINGFACE:
        frame, raw_path = _extract_huggingface(dataset_id, spec)
    elif source_type == KAGGLE:
        frame, raw_path = _extract_kaggle(dataset_id)
    else:
        raise ExtractionError(f"unsupported source type: {source_type!r}; expected one of {sorted({HUGGINGFACE, KAGGLE})}")
    return frame, raw_path


def _extract_huggingface(dataset_id: str, spec: DatasetSpec) -> tuple[pd.DataFrame, Path]:
    """Download a CSV hosted on the Hugging Face Hub."""
    settings = get_settings()
    destination = settings.raw_dir / "huggingface.csv"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ExtractionError("huggingface_hub is not installed: pip install huggingface-hub") from exc

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=spec.file_name or "creditcard.csv",
            token=settings.hf_token,
            local_dir=settings.raw_dir,
        )
        destination = Path(downloaded)
        frame = pd.read_csv(destination)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        logger.exception("huggingface_extraction_failed", extra={"dataset_id": dataset_id})
        raise ExtractionError(str(exc)) from exc

    return frame, destination


def _extract_kaggle(dataset_id: str) -> tuple[pd.DataFrame, Path]:
    """Download an authenticated Kaggle dataset."""
    settings = get_settings()
    destination = settings.raw_dir / "kaggle.csv"

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ExtractionError("kaggle is not installed: pip install kaggle") from exc

    if not settings.is_kaggle_configured:
        raise ExtractionError(
            "Kaggle credentials are not configured; set KAGGLE_USERNAME and KAGGLE_KEY"
        )

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_id, path=str(settings.raw_dir), unzip=True)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        logger.exception("kaggle_extraction_failed", extra={"dataset_id": dataset_id})
        raise ExtractionError(str(exc)) from exc

    candidates = sorted(
        settings.raw_dir.glob("*.csv"), key=lambda path: path.stat().st_size, reverse=True
    )
    if not candidates:
        raise ExtractionError(f"no CSV file was downloaded from {dataset_id!r}")

    destination = candidates[0]
    frame = pd.read_csv(destination)

    return frame, destination


def extract(source_type: str, dataset_id: str | None = None) -> ExtractionResult:
    """Backwards-compatible alias for extract_dataset."""
    return extract_dataset(source_type=source_type, dataset_id=dataset_id)
