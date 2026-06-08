"""Metric registry and binary classification metric helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


LOGGER = logging.getLogger(__name__)

SUPPORTED_METRICS: frozenset[str] = frozenset(
    {
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "mcc",
        "accuracy",
    }
)

RESULT_COLUMNS: tuple[str, ...] = (
    "model_name",
    "feature_group",
    "split",
    "training_time_seconds",
    "prediction_time_seconds",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "mcc",
    "accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
    "n_samples",
    "positive_support",
)


def validate_metric_names(metric_names: list[str] | tuple[str, ...] | set[str]) -> None:
    """Validate that every configured metric name is supported."""

    unsupported = sorted(set(metric_names) - SUPPORTED_METRICS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        invalid = ", ".join(unsupported)
        raise ValueError(f"Unsupported metric name(s): {invalid}. Supported metrics: {supported}.")


def get_probability_scores(model: Any, X: Any) -> np.ndarray:
    """Return probability-like scores for binary ranking metrics."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return at least two class columns.")
        return np.asarray(probabilities[:, 1])

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))

    raise ValueError("Model must provide predict_proba or decision_function for ranking metrics.")


def calculate_confusion_counts(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    """Calculate confusion matrix counts with fixed labels [0, 1]."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def calculate_binary_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate binary classification metrics from probability-like scores."""

    y_true_array = np.asarray(y_true)
    y_score_array = np.asarray(y_score)
    y_pred = (y_score_array >= threshold).astype(int)
    unique_labels = np.unique(y_true_array)

    try:
        pr_auc = float(average_precision_score(y_true_array, y_score_array))
    except ValueError:
        LOGGER.warning("PR-AUC is undefined because y_true contains insufficient class variation.")
        pr_auc = float("nan")

    if len(unique_labels) < 2:
        LOGGER.warning("ROC-AUC is undefined because y_true contains only one class.")
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true_array, y_score_array))

    tn, fp, fn, tp = calculate_confusion_counts(y_true_array, y_pred)
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_array, y_pred)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_samples": int(len(y_true_array)),
        "positive_support": int(np.sum(y_true_array == 1)),
    }


def build_result_record(
    model_name: str,
    feature_group: str,
    split_name: str,
    metrics: dict[str, float | int],
    training_time: float,
    prediction_time: float,
) -> dict[str, float | int | str]:
    """Build one result table record."""

    return {
        "model_name": model_name,
        "feature_group": feature_group,
        "split": split_name,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        **metrics,
    }
