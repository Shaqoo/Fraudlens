"""Machine learning: training, evaluation, inference and model versioning."""

from fraudlens.ml.evaluate import evaluate_model, select_threshold
from fraudlens.ml.model_registry import ModelRegistry
from fraudlens.ml.predict import predict_batch, predict_single
from fraudlens.ml.train import train_model

__all__ = [
    "train_model",
    "evaluate_model",
    "select_threshold",
    "predict_single",
    "predict_batch",
    "ModelRegistry",
]
