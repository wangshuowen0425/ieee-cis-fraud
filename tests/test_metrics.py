"""Tests for metric helpers and small train/evaluate flows."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from src.evaluate import evaluate_model
from src.metrics import (
    SUPPORTED_METRICS,
    calculate_binary_metrics,
    calculate_confusion_counts,
    validate_metric_names,
)
from src.models import build_dummy_model, build_logistic_regression_model, build_training_pipeline
from src.preprocessing import build_preprocessor, split_features_target
from src.train import train_model


def _toy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "product": ["a", "a", "b", "b", "c", "c"],
            "isFraud": [0, 0, 0, 1, 1, 1],
        }
    )


def test_valid_metric_names_pass() -> None:
    validate_metric_names(["pr_auc", "roc_auc", "precision", "recall", "f1", "mcc", "accuracy"])


def test_unsupported_metric_names_are_rejected() -> None:
    try:
        validate_metric_names(["pr_auc", "balanced_accuracy"])
    except ValueError as error:
        assert "balanced_accuracy" in str(error)
    else:
        raise AssertionError("Expected unsupported metric name to raise ValueError.")


def test_supported_metrics_match_model_config() -> None:
    with Path("configs/model_config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert set(config["metrics"]) == SUPPORTED_METRICS


def test_pr_auc_uses_scores_not_hard_predictions() -> None:
    metrics_from_scores = calculate_binary_metrics(
        [0, 1, 0, 1],
        [0.1, 0.4, 0.35, 0.8],
        threshold=0.5,
    )
    metrics_from_hard_predictions = calculate_binary_metrics(
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        threshold=0.5,
    )

    assert metrics_from_scores["pr_auc"] != metrics_from_hard_predictions["pr_auc"]


def test_confusion_matrix_order_is_fixed() -> None:
    assert calculate_confusion_counts([0, 0, 1, 1], [0, 1, 0, 1]) == (1, 1, 1, 1)


def test_single_class_roc_auc_returns_nan() -> None:
    metrics = calculate_binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])

    assert math.isnan(float(metrics["roc_auc"]))


def test_dummy_model_can_train_and_evaluate() -> None:
    dataframe = _toy_dataframe()
    X_train, y_train = split_features_target(dataframe, ["amount", "product"], "isFraud")
    preprocessor = build_preprocessor(["amount"], ["product"], {"scale_numeric_for_logistic": True})
    pipeline = build_training_pipeline(preprocessor, build_dummy_model())

    trained = train_model(pipeline, X_train, y_train)
    record, y_score = evaluate_model(
        trained, X_train, y_train, 0.5, "dummy", "transaction_basic", "train"
    )

    assert record["model_name"] == "dummy"
    assert len(y_score) == len(dataframe)


def test_logistic_regression_model_can_train_and_evaluate() -> None:
    dataframe = _toy_dataframe()
    X_train, y_train = split_features_target(dataframe, ["amount", "product"], "isFraud")
    preprocessor = build_preprocessor(
        ["amount"],
        ["product"],
        {
            "scale_numeric_for_logistic": True,
            "one_hot_handle_unknown": "ignore",
            "one_hot_min_frequency": 1,
            "one_hot_max_categories": 10,
        },
    )
    classifier = build_logistic_regression_model(42, {"max_iter": 500})
    pipeline = build_training_pipeline(preprocessor, classifier)

    trained = train_model(pipeline, X_train, y_train)
    record, y_score = evaluate_model(
        trained, X_train, y_train, 0.5, "logistic_regression", "transaction_basic", "train"
    )

    assert record["model_name"] == "logistic_regression"
    assert len(y_score) == len(dataframe)
