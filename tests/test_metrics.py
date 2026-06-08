"""Tests for metric helpers and small train/evaluate flows."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from src.evaluate import evaluate_model
from src.experiment_runner import (
    build_selection_details,
    extract_tree_feature_importance,
    plot_tree_feature_importance_top20,
)
from src.metrics import (
    SUPPORTED_METRICS,
    calculate_binary_metrics,
    calculate_confusion_counts,
    validate_metric_names,
)
from src.models import (
    build_dummy_model,
    build_logistic_regression_model,
    build_training_pipeline,
    resolve_tree_model,
)
from src.preprocessing import build_preprocessor, build_tree_preprocessor, split_features_target
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


def test_tree_preprocessor_uses_ordinal_encoding_without_scaling() -> None:
    dataframe = _toy_dataframe()
    preprocessor = build_tree_preprocessor(
        ["amount"],
        ["product"],
        {"numeric_imputer": "median", "tree_categorical_fill_value": "**MISSING**"},
    )

    transformed = preprocessor.fit_transform(dataframe[["amount", "product"]])

    assert transformed.shape == (len(dataframe), 2)


def test_resolve_tree_model_records_lightgbm_disabled_reason() -> None:
    config = {
        "models": {
            "lightgbm": {"enabled": False},
            "hist_gradient_boosting": {"enabled": True, "max_iter": 2},
            "random_forest": {"enabled": True, "n_estimators": 2},
        },
        "tree_model_fallback_order": ["lightgbm", "hist_gradient_boosting", "random_forest"],
    }

    actual_name, estimator, reason = resolve_tree_model(config, random_seed=42)

    assert actual_name == "hist_gradient_boosting"
    assert estimator.__class__.__name__ == "HistGradientBoostingClassifier"
    assert "lightgbm disabled" in str(reason)


def test_selection_details_are_computed_from_validation_ablation_only() -> None:
    ablation = pd.DataFrame(
        [
            {"feature_group": "transaction_basic", "pr_auc": 0.60, "recall": 0.3, "feature_count": 52},
            {"feature_group": "transaction_identity", "pr_auc": 0.55, "recall": 0.9, "feature_count": 69},
            {"feature_group": "transaction_identity_missing", "pr_auc": 0.59, "recall": 0.4, "feature_count": 77},
        ]
    )

    details = build_selection_details(ablation, "transaction_basic", "highest valid PR-AUC")

    assert details["selected_feature_group"] == "transaction_basic"
    assert details["selection_metric"] == "valid PR-AUC"
    assert details["validation_selection_score"] == 0.60
    assert details["runner_up_feature_group"] == "transaction_identity_missing"
    assert details["runner_up_validation_score"] == 0.59
    assert details["score_difference"] == 0.010000000000000009
    assert "test" not in details


class _FakePreprocessor:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def get_feature_names_out(self) -> list[str]:
        return self._names


class _FakeClassifier:
    def __init__(self, importances: list[float]) -> None:
        self.feature_importances_ = importances


class _FakePipeline:
    def __init__(self, names: list[str], importances: list[float]) -> None:
        self.named_steps = {
            "preprocessor": _FakePreprocessor(names),
            "classifier": _FakeClassifier(importances),
        }


def test_feature_importance_maps_when_lengths_match(tmp_path: Path) -> None:
    pipeline = _FakePipeline(["numeric__amount", "categorical__product"], [3.0, 1.0])

    importance = extract_tree_feature_importance(pipeline)
    figure_path = plot_tree_feature_importance_top20(pipeline, tmp_path / "importance.png")

    assert importance is not None
    assert importance["feature"].tolist() == ["amount", "product"]
    assert figure_path is not None
    assert figure_path.exists()


def test_feature_importance_refuses_mismatched_lengths(tmp_path: Path) -> None:
    pipeline = _FakePipeline(["numeric__amount", "categorical__product"], [3.0])

    importance = extract_tree_feature_importance(pipeline)
    figure_path = plot_tree_feature_importance_top20(pipeline, tmp_path / "importance.png")

    assert importance is None
    assert figure_path is None
    assert not (tmp_path / "importance.png").exists()
