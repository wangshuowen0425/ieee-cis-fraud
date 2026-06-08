"""Training helpers for sklearn pipelines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib


def train_model(pipeline: Any, X_train: Any, y_train: Any) -> Any:
    """Fit a sklearn pipeline and return it."""

    return pipeline.fit(X_train, y_train)


def measure_training_time(pipeline: Any, X_train: Any, y_train: Any) -> tuple[Any, float]:
    """Fit a pipeline and return the fitted object with elapsed seconds."""

    start_time = time.perf_counter()
    trained_pipeline = train_model(pipeline, X_train, y_train)
    elapsed_seconds = time.perf_counter() - start_time
    return trained_pipeline, elapsed_seconds


def save_model(model: Any, output_path: Path) -> Path:
    """Persist a trained model with joblib."""

    if "data/raw" in output_path.as_posix():
        raise ValueError("Models must not be saved under data/raw.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path
