"""Tests for the feature optimization experiment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.feature_optimization_experiment import (
    UID_COLUMNS,
    add_domain_features,
    build_feature_summary,
    build_optimized_splits,
    fit_group_aggregations,
    get_final_feature_list,
    split_by_time,
    transform_group_aggregations,
)


def _sample_frame(n_rows: int = 24) -> pd.DataFrame:
    """Build a deterministic small fraud-like frame."""

    def repeat(values: list[object]) -> list[object]:
        repeats = (n_rows // len(values)) + 1
        return (values * repeats)[:n_rows]

    return pd.DataFrame(
        {
            "TransactionID": range(1000, 1000 + n_rows),
            "TransactionDT": np.arange(n_rows) * 86400,
            "isFraud": repeat([0, 1, 0, 0, 1, 0]),
            "TransactionAmt": np.linspace(10.25, 80.75, n_rows),
            "ProductCD": repeat(["W", "C", "W", "R"]),
            "card1": repeat([100, 100, 101, 101, 102, 102]),
            "card2": repeat([1, 2, 3, 4, 5, 6]),
            "card3": [150] * n_rows,
            "card4": repeat(["visa", "mastercard"]),
            "card5": [226] * n_rows,
            "card6": repeat(["debit", "credit"]),
            "addr1": repeat([10, 10, 11, 11, 12, 12]),
            "addr2": [87] * n_rows,
            "dist1": np.arange(n_rows, dtype=float),
            "C1": np.arange(n_rows) % 5,
            "C2": np.arange(n_rows) % 7,
            "C13": np.arange(n_rows) % 3,
            "D1": np.arange(n_rows) % 4,
            "D2": np.arange(n_rows) % 5,
            "D10": np.arange(n_rows) % 6,
            "D15": np.arange(n_rows) % 7,
            "M1": repeat(["T", None, "F", "T"]),
            "id_01": repeat([None, -5.0, None, None]),
            "DeviceType": repeat([None, "desktop", None, "mobile"]),
        }
    )


BASIC_FEATURES = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C13",
    "D1",
    "D2",
    "D10",
    "D15",
    "M1",
]


def test_domain_time_and_d1n_features_are_correct() -> None:
    frame = add_domain_features(_sample_frame(6), basic_features=BASIC_FEATURES)

    assert frame.loc[1, "TransactionDay"] == 1
    assert frame.loc[5, "TransactionWeek"] == 0
    assert frame.loc[5, "D1n"] == frame.loc[5, "TransactionDay"] - frame.loc[5, "D1"]


def test_amount_features_are_correct() -> None:
    frame = add_domain_features(_sample_frame(6), basic_features=BASIC_FEATURES)

    assert frame.loc[0, "TransactionAmt_log1p"] == pytest.approx(np.log1p(10.25))
    assert frame.loc[0, "TransactionAmt_cents"] == pytest.approx(0.25)


def test_identity_and_basic_missing_features_are_correct() -> None:
    frame = add_domain_features(_sample_frame(8), basic_features=BASIC_FEATURES)

    assert frame.loc[0, "identity_present"] == 0
    assert frame.loc[1, "identity_present"] == 1
    assert frame.loc[0, "identity_missing_count"] == 2
    assert frame.loc[1, "identity_missing_count"] == 0
    assert frame.loc[1, "basic_missing_count"] == 1


def test_uid_temporary_columns_do_not_enter_final_feature_list() -> None:
    train, valid, test = _sample_frame(12).iloc[:6], _sample_frame(12).iloc[6:9], _sample_frame(12).iloc[9:]
    opt_train, _, _, stats = build_optimized_splits(train, valid, test, BASIC_FEATURES)

    features = get_final_feature_list(opt_train, BASIC_FEATURES, stats)

    assert all(uid not in features for uid in UID_COLUMNS)
    assert any(feature.startswith("agg_uid_") for feature in features)


def test_group_aggregation_uses_train_mapping_and_unknown_uid_is_nan() -> None:
    train = add_domain_features(_sample_frame(8), basic_features=BASIC_FEATURES)
    stats = fit_group_aggregations(train)
    valid = add_domain_features(_sample_frame(4), basic_features=BASIC_FEATURES)
    valid.loc[:, "card1"] = 999
    valid = add_domain_features(valid.drop(columns=list(UID_COLUMNS), errors="ignore"), basic_features=BASIC_FEATURES)

    transformed = transform_group_aggregations(valid, stats)
    agg_columns = [column for column in transformed.columns if column.startswith("agg_uid_")]

    assert agg_columns
    assert transformed[agg_columns].isna().all().all()


def test_time_split_order_is_valid() -> None:
    frame = _sample_frame(24).sample(frac=1, random_state=42).reset_index(drop=True)

    train, valid, test = split_by_time(frame)

    assert train["TransactionDT"].max() <= valid["TransactionDT"].min()
    assert valid["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(train) == 14
    assert len(valid) == 5
    assert len(test) == 5


def test_feature_summary_structure() -> None:
    train, valid, test = _sample_frame(12).iloc[:6], _sample_frame(12).iloc[6:9], _sample_frame(12).iloc[9:]
    opt_train, _, _, stats = build_optimized_splits(train, valid, test, BASIC_FEATURES)
    features = get_final_feature_list(opt_train, BASIC_FEATURES, stats)

    summary = build_feature_summary(BASIC_FEATURES, features, stats)

    assert {"feature_name", "feature_type", "source_columns", "generated", "reason"}.issubset(
        summary.columns
    )
    assert "group_aggregation" in set(summary["feature_type"])


def test_cli_end_to_end_on_tmp_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = _sample_frame(30)
    frame.iloc[:12].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[12:21].to_parquet(data_dir / "valid.parquet", index=False)
    frame.iloc[21:].to_parquet(data_dir / "test.parquet", index=False)
    feature_groups_path = tmp_path / "feature_groups.json"
    feature_groups_path.write_text(
        json.dumps({"groups": {"transaction_basic": {"features": BASIC_FEATURES}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text(
        """
random_seed: 42
models:
  lightgbm:
    enabled: false
  hist_gradient_boosting:
    enabled: true
    max_iter: 5
    learning_rate: 0.1
tree_model_fallback_order:
  - lightgbm
  - hist_gradient_boosting
preprocessing:
  numeric_imputer: median
  tree_categorical_fill_value: "**MISSING**"
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.feature_optimization_experiment",
            "--data-dir",
            str(data_dir),
            "--feature-groups",
            str(feature_groups_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "optimization",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "tables/optimization_feature_summary.csv").exists()
    assert (output_dir / "tables/optimization_random_valid_comparison.csv").exists()
    assert (output_dir / "tables/optimization_time_split_comparison.csv").exists()
    assert (output_dir / "tables/optimization_selected_features.json").exists()
    assert (output_dir / "figures/optimization_pr_auc_comparison.png").exists()
    assert matplotlib.get_backend().lower() == "agg"
