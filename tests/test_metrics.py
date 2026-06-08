"""Tests for metric registry validation."""

from pathlib import Path

import yaml

from src.metrics import SUPPORTED_METRICS, validate_metric_names


def test_valid_metric_names_pass() -> None:
    """Configured metric names should pass validation when supported."""

    validate_metric_names(["pr_auc", "roc_auc", "precision", "recall", "f1", "mcc", "accuracy"])


def test_unsupported_metric_names_are_rejected() -> None:
    """Unknown metric names should fail with a clear validation error."""

    try:
        validate_metric_names(["pr_auc", "balanced_accuracy"])
    except ValueError as error:
        assert "balanced_accuracy" in str(error)
    else:
        raise AssertionError("Expected unsupported metric name to raise ValueError.")


def test_supported_metrics_match_model_config() -> None:
    """The model config should only reference supported metric names."""

    config_path = Path("configs/model_config.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    configured_metrics = {config["primary_metric"], *config["additional_metrics"]}
    assert configured_metrics == SUPPORTED_METRICS
