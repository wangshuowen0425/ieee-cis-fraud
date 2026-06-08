"""Data and feature quality reports for processed IEEE-CIS datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


def fraud_rate_difference(original_rate: float, sampled_rate: float) -> float:
    """Return the absolute fraud-rate difference."""
    return abs(float(sampled_rate) - float(original_rate))


def fraud_rate_check_passed(
    original_rate: float,
    sampled_rate: float,
    maximum_allowed_difference: float,
) -> bool:
    """Return whether the sampled fraud rate is close enough to the original rate."""
    return fraud_rate_difference(original_rate, sampled_rate) <= float(maximum_allowed_difference)


def count_duplicate_values(data: pd.DataFrame, id_column: str) -> int:
    """Count rows whose id value is duplicated."""
    return int(data[id_column].duplicated(keep=False).sum())


def count_id_overlaps(left: pd.DataFrame, right: pd.DataFrame, id_column: str) -> int:
    """Count overlapping IDs between two dataframes."""
    return len(set(left[id_column]) & set(right[id_column]))


def identify_feature_source(
    feature: str,
    transaction_columns: list[str],
    identity_columns: list[str],
    missing_indicator_columns: list[str],
) -> str:
    """Identify whether a feature comes from transaction, identity, or derived missingness."""
    if feature == "missing_count" or feature in missing_indicator_columns:
        return "derived_missing"
    if feature in identity_columns:
        return "identity"
    if feature in transaction_columns:
        return "transaction"
    return "unknown"


def build_feature_profile(
    data: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    transaction_columns: list[str],
    identity_columns: list[str],
    missing_indicator_columns: list[str],
) -> pd.DataFrame:
    """Build feature-level dtype, missingness, uniqueness, and group membership profile."""
    ordered_features = feature_groups["transaction_identity_missing"]
    rows: list[dict[str, Any]] = []
    basic = set(feature_groups["transaction_basic"])
    identity = set(feature_groups["transaction_identity"])
    missing = set(feature_groups["transaction_identity_missing"])
    for feature in ordered_features:
        rows.append(
            {
                "feature": feature,
                "source": identify_feature_source(
                    feature,
                    transaction_columns=transaction_columns,
                    identity_columns=identity_columns,
                    missing_indicator_columns=missing_indicator_columns,
                ),
                "dtype": str(data[feature].dtype),
                "missing_rate": float(data[feature].isna().mean()),
                "unique_count": int(data[feature].nunique(dropna=True)),
                "in_transaction_basic": feature in basic,
                "in_transaction_identity": feature in identity,
                "in_transaction_identity_missing": feature in missing,
            }
        )
    return pd.DataFrame(rows)


def build_data_quality_report(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    merged: pd.DataFrame,
    sampled: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    requested_sample_size: int,
    missing_candidate_columns_not_found: list[str],
    id_column: str,
    target_column: str,
) -> dict[str, Any]:
    """Build data-quality metrics from real dataframes."""
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]
    original_fraud_count = int(merged[target_column].sum())
    sampled_fraud_count = int(sampled[target_column].sum())
    original_fraud_rate = float(merged[target_column].mean()) if len(merged) else 0.0
    sampled_fraud_rate = float(sampled[target_column].mean()) if len(sampled) else 0.0
    identity_matched_rows = int(transaction[id_column].isin(set(identity[id_column])).sum())
    return {
        "transaction_rows": int(len(transaction)),
        "identity_rows": int(len(identity)),
        "merged_rows": int(len(merged)),
        "duplicate_transaction_ids": count_duplicate_values(transaction, id_column),
        "duplicate_identity_ids": count_duplicate_values(identity, id_column),
        "identity_matched_rows": identity_matched_rows,
        "identity_match_rate": float(identity_matched_rows / len(transaction)) if len(transaction) else 0.0,
        "original_fraud_count": original_fraud_count,
        "original_fraud_rate": original_fraud_rate,
        "requested_sample_size": int(requested_sample_size),
        "actual_sample_size": int(len(sampled)),
        "sampled_fraud_count": sampled_fraud_count,
        "sampled_fraud_rate": sampled_fraud_rate,
        "fraud_rate_absolute_difference": fraud_rate_difference(original_fraud_rate, sampled_fraud_rate),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_fraud_count": int(train[target_column].sum()),
        "valid_fraud_count": int(valid[target_column].sum()),
        "test_fraud_count": int(test[target_column].sum()),
        "train_fraud_rate": float(train[target_column].mean()) if len(train) else 0.0,
        "valid_fraud_rate": float(valid[target_column].mean()) if len(valid) else 0.0,
        "test_fraud_rate": float(test[target_column].mean()) if len(test) else 0.0,
        "train_valid_id_overlap": count_id_overlaps(train, valid, id_column),
        "train_test_id_overlap": count_id_overlaps(train, test, id_column),
        "valid_test_id_overlap": count_id_overlaps(valid, test, id_column),
        "basic_feature_count": len(feature_groups["transaction_basic"]),
        "identity_feature_count": len(feature_groups["transaction_identity"]),
        "missing_feature_count": len(feature_groups["transaction_identity_missing"]),
        "missing_candidate_columns_not_found": missing_candidate_columns_not_found,
    }
