"""Evaluation and plotting helpers for stage 1 model experiments."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

from src.metrics import build_result_record, calculate_binary_metrics, get_probability_scores


LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X: Any,
    y: Any,
    threshold: float,
    model_name: str,
    feature_group: str,
    split_name: str,
    training_time: float = 0.0,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Evaluate a fitted model and return a result record plus scores."""

    start_time = time.perf_counter()
    y_score = get_probability_scores(model, X)
    prediction_time = time.perf_counter() - start_time
    metrics = calculate_binary_metrics(y, y_score, threshold=threshold)
    record = build_result_record(
        model_name=model_name,
        feature_group=feature_group,
        split_name=split_name,
        metrics=metrics,
        training_time=training_time,
        prediction_time=prediction_time,
    )
    return record, y_score


def save_results_table(records: list[dict[str, Any]], output_path: Path) -> Path:
    """Save result records to a CSV table."""

    if "data/raw" in output_path.as_posix():
        raise ValueError("Result tables must not be saved under data/raw.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def plot_precision_recall_curve(
    y_true: Any,
    y_score: Any,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Save a precision-recall curve PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots()
    try:
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axis)
    except ValueError:
        LOGGER.warning("Precision-recall curve is unavailable for the provided labels.")
        axis.text(0.5, 0.5, "Precision-recall curve unavailable", ha="center", va="center")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
    if title:
        axis.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_confusion_matrix(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Save a simple confusion matrix PNG."""

    matrix = np.array([[tn, fp], [fn, tp]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], labels=["True 0", "True 1"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    if title:
        axis.set_title(title)
    plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
