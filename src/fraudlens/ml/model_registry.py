"""Model artifact persistence and version lifecycle management."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from fraudlens.config import get_settings

logger = logging.getLogger(__name__)

VALID_STATUSES = {"candidate", "active", "retired", "rejected"}


@dataclass
class ModelVersion:
    version: str
    artifact_path: Path
    threshold: float = 0.5
    feature_names: list[str] = field(default_factory=list)
    feature_medians: dict = field(default_factory=dict)
    evaluation_metrics: dict = field(default_factory=dict)
    training_dataset: str | None = None
    training_config: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "candidate"

    def validate(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")


class ModelRegistry:
    """Stores model artifacts on disk and their metadata as sidecar JSON."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or get_settings().model_path).parent
        self.root.mkdir(parents=True, exist_ok=True)

    def _artifact(self, version: str) -> Path:
        return self.root / f"{version}.joblib"

    def _metadata(self, version: str) -> Path:
        return self.root / f"{version}.json"

    def save(self, bundle: dict, metadata: ModelVersion) -> ModelVersion:
        metadata.validate()
        joblib.dump(bundle, self._artifact(metadata.version))
        self._metadata(metadata.version).write_text(
            json.dumps({**asdict(metadata), "artifact_path": str(self._artifact(metadata.version))}, indent=2),
            encoding="utf-8",
        )
        logger.info("model_saved", extra={"version": metadata.version, "status": metadata.status})
        return metadata

    def load(self, version: str | None = None) -> tuple[dict, ModelVersion]:
        version = version or self.active_version()
        if version is None:
            raise FileNotFoundError("no active model version is registered")
        artifact = self._artifact(version)
        if not artifact.exists():
            raise FileNotFoundError(f"model artifact not found: {artifact}")
        metadata = ModelVersion(**json.loads(self._metadata(version).read_text(encoding="utf-8")))
        return joblib.load(artifact), metadata

    def versions(self) -> list[ModelVersion]:
        return [
            ModelVersion(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.root.glob("*.json"))
        ]

    def active_version(self) -> str | None:
        for metadata in self.versions():
            if metadata.status == "active":
                return metadata.version
        return None

    def promote(self, version: str) -> ModelVersion:
        """Retire the current active model and mark ``version`` as active."""
        for metadata in self.versions():
            if metadata.status == "active":
                metadata.status = "retired"
                self._metadata(metadata.version).write_text(
                    json.dumps(asdict(metadata), indent=2), encoding="utf-8"
                )
        candidate = next(m for m in self.versions() if m.version == version)
        candidate.status = "active"
        self._metadata(version).write_text(json.dumps(asdict(candidate), indent=2), encoding="utf-8")
        logger.info("model_promoted", extra={"version": version})
        return candidate
