"""Preprocessing helpers for model-side stage 1 experiments."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


EXPECTED_FEATURE_GROUPS: frozenset[str] = frozenset(
    {"transaction_basic", "transaction_identity", "transaction_identity_missing"}
)


def load_feature_groups(feature_groups_path: Path) -> dict[str, list[str]]:
    """Load feature group definitions from JSON."""

    if not feature_groups_path.exists():
        raise FileNotFoundError(f"Feature groups file not found: {feature_groups_path}")

    with feature_groups_path.open("r", encoding="utf-8") as file:
        feature_groups = json.load(file)

    if not isinstance(feature_groups, dict):
        raise ValueError("Feature groups file must contain a JSON object.")

    if "groups" in feature_groups and isinstance(feature_groups["groups"], dict):
        feature_groups = {
            group_name: group_payload.get("features", [])
            for group_name, group_payload in feature_groups["groups"].items()
            if isinstance(group_payload, dict)
        }

    if not EXPECTED_FEATURE_GROUPS.intersection(feature_groups):
        expected = ", ".join(sorted(EXPECTED_FEATURE_GROUPS))
        raise ValueError(f"Feature groups must include at least one of: {expected}.")

    for group_name, columns in feature_groups.items():
        if not isinstance(group_name, str) or not isinstance(columns, list):
            raise ValueError("Each feature group must map a name to a list of columns.")
        if not all(isinstance(column, str) for column in columns):
            raise ValueError(f"Feature group '{group_name}' must contain only column names.")

    return feature_groups


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load processed data metadata from JSON."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata file must contain a JSON object.")

    missing_keys = [key for key in ("numeric_columns", "categorical_columns") if key not in metadata]
    if missing_keys:
        raise ValueError(f"Metadata missing required key(s): {', '.join(missing_keys)}.")

    for key in ("numeric_columns", "categorical_columns"):
        if not isinstance(metadata[key], list) or not all(
            isinstance(column, str) for column in metadata[key]
        ):
            raise ValueError(f"Metadata '{key}' must be a list of column names.")

    return metadata


def validate_requested_features(
    dataframe: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    feature_group_name: str,
    target_column: str,
    id_column: str,
    time_column: str,
) -> list[str]:
    """Validate and return model feature columns for one feature group."""

    if feature_group_name not in feature_groups:
        raise ValueError(f"Feature group not found: {feature_group_name}")

    blocked_columns = {target_column, id_column, time_column}
    requested_features = feature_groups[feature_group_name]
    illegal_features = [column for column in requested_features if column in blocked_columns]
    if illegal_features:
        raise ValueError(
            "Feature group contains columns that are not allowed as model features: "
            f"{', '.join(illegal_features)}."
        )

    missing_features = [column for column in requested_features if column not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Feature group columns missing from dataframe: {', '.join(missing_features)}.")

    return list(requested_features)


def build_numeric_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build the numeric preprocessing pipeline without fitting it."""

    strategy = config.get("numeric_imputer", "median")
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy=strategy))]
    if config.get("scale_numeric_for_logistic", True):
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def build_categorical_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build the categorical preprocessing pipeline without fitting it."""

    encoder_params: dict[str, Any] = {
        "handle_unknown": config.get("one_hot_handle_unknown", "ignore"),
        "min_frequency": config.get("one_hot_min_frequency", 20),
        "max_categories": config.get("one_hot_max_categories", 100),
    }
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_params["sparse_output"] = True
    else:
        encoder_params["sparse"] = True

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=config.get("categorical_imputer", "constant"),
                    fill_value=config.get("categorical_fill_value", "__MISSING__"),
                ),
            ),
            ("one_hot", OneHotEncoder(**encoder_params)),
        ]
    )


def build_tree_numeric_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build numeric preprocessing for tree models without scaling."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config.get("numeric_imputer", "median"))),
        ]
    )


def build_tree_categorical_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build categorical preprocessing for tree models using ordinal encoding."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=config.get("categorical_imputer", "constant"),
                    fill_value=config.get("tree_categorical_fill_value", "**MISSING**"),
                ),
            ),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: dict[str, Any],
) -> ColumnTransformer:
    """Build a column transformer for numeric and categorical columns."""

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", build_numeric_pipeline(config), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", build_categorical_pipeline(config), categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tree_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: dict[str, Any],
) -> ColumnTransformer:
    """Build a column transformer for tree models."""

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", build_tree_numeric_pipeline(config), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", build_tree_categorical_pipeline(config), categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_features_target(
    dataframe: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into model features and target."""

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    return dataframe.loc[:, features], dataframe[target_column]
