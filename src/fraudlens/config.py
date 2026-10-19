"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    database_url: str | None = None
    model_path: Path = PROJECT_ROOT / "models" / "fraud_model.joblib"
    log_level: str = "INFO"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    log_dir: Path = PROJECT_ROOT / "logs"
    predict_threshold: float = 0.5
    max_batch_size: int = 1000
    persist_predictions: bool = True

    # Dataset sources
    hf_dataset_id: str = "David-Egea/Creditcard-fraud-detection"
    hf_token: str | None = None
    kaggle_dataset_id: str = "ealaxi/paysim1"

    # Analyzed prediction (OCR / text)
    ocr_backend: str = "auto"          # auto | tesseract | easyocr
    max_extract_mb: int = 5            # max upload size in megabytes

    extra: dict[str, str] = field(default_factory=dict)

    @property
    def max_extract_bytes(self) -> int:
        return self.max_extract_mb * 1024 * 1024

    @property
    def is_kaggle_configured(self) -> bool:
        return all(os.getenv(k) for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    api_env = {k: v for k, v in os.environ.items() if k.startswith(("HF_", "KAGGLE_"))}
    return Settings(
        database_url=os.getenv("DATABASE_URL"),
        model_path=Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "fraud_model.joblib")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        predict_threshold=float(os.getenv("PREDICT_THRESHOLD", "0.5")),
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
        persist_predictions=os.getenv("PERSIST_PREDICTIONS", "true").lower() == "true",
        hf_dataset_id=os.getenv("HF_DATASET_ID", "David-Egea/Creditcard-fraud-detection"),
        hf_token=os.getenv("HF_TOKEN"),
        kaggle_dataset_id=os.getenv("KAGGLE_DATASET_ID", "ealaxi/paysim1"),
        ocr_backend=os.getenv("OCR_BACKEND", "auto"),
        max_extract_mb=int(os.getenv("MAX_EXTRACT_MB", "5")),
        extra=api_env,
    )
