"""Metric registry and validation helpers for experiment configuration."""

from __future__ import annotations


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
    "feature_group",
    "model",
    "primary_metric",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "mcc",
    "accuracy",
)


def validate_metric_names(metric_names: list[str] | tuple[str, ...] | set[str]) -> None:
    """Validate that every configured metric name is supported.

    Parameters
    ----------
    metric_names:
        Metric names to validate.

    Raises
    ------
    ValueError
        If any metric name is not part of ``SUPPORTED_METRICS``.
    """

    unsupported = sorted(set(metric_names) - SUPPORTED_METRICS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        invalid = ", ".join(unsupported)
        raise ValueError(f"Unsupported metric name(s): {invalid}. Supported metrics: {supported}.")
