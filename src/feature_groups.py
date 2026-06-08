"""Feature group builders for the IEEE-CIS fraud data pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
EXCLUDED_FEATURE_COLUMNS = {"TransactionID", "isFraud", "TransactionDT"}


def _ordered_unique(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for column in columns:
        if column not in seen and column not in EXCLUDED_FEATURE_COLUMNS:
            unique.append(column)
            seen.add(column)
    return unique


def build_transaction_basic_group(existing_transaction_columns: list[str], config: dict[str, Any]) -> list[str]:
    """Build the basic transaction feature group from configured transaction candidates."""
    existing = set(existing_transaction_columns)
    return _ordered_unique([column for column in config.get("transaction_columns", []) if column in existing])


def build_transaction_identity_group(
    transaction_basic: list[str],
    existing_identity_columns: list[str],
    config: dict[str, Any],
) -> list[str]:
    """Build transaction plus identity feature group."""
    existing_identity = set(existing_identity_columns)
    identity_features = [column for column in config.get("identity_columns", []) if column in existing_identity]
    return _ordered_unique(transaction_basic + identity_features)


def build_transaction_identity_missing_group(
    transaction_identity: list[str],
    missing_indicator_columns: list[str],
) -> list[str]:
    """Build transaction plus identity plus missingness feature group."""
    return _ordered_unique(transaction_identity + ["missing_count"] + missing_indicator_columns)


def validate_feature_groups(feature_groups: dict[str, list[str]]) -> None:
    """Validate feature groups exclude non-feature columns and preserve nesting."""
    for group_name, columns in feature_groups.items():
        invalid = [column for column in columns if column in EXCLUDED_FEATURE_COLUMNS]
        if invalid:
            raise ValueError(f"{group_name} contains non-feature columns: {invalid}")

    basic = feature_groups["transaction_basic"]
    identity = feature_groups["transaction_identity"]
    missing = feature_groups["transaction_identity_missing"]
    if not set(basic).issubset(identity):
        raise ValueError("transaction_basic must be a subset of transaction_identity")
    if not set(identity).issubset(missing):
        raise ValueError("transaction_identity must be a subset of transaction_identity_missing")


def save_feature_groups(feature_groups: dict[str, list[str]], output_path: str | Path) -> Path:
    """Save feature groups as JSON."""
    validate_feature_groups(feature_groups)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_groups, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved feature groups to %s", path)
    return path
