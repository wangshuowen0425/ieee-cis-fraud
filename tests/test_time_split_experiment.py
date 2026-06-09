"""Tests for the time-split extension experiment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd

from src.time_split_experiment import (
    build_threshold_comparison,
    build_time_split_summary,
    build_time_splits,
    build_vs_random_comparison,
    validate_full_sample,
)


BASE_COMMAND: tuple[str, ...] = (sys.executable, "-m", "src.time_split_experiment")


def _full_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": list(range(10)),
            "TransactionDT": [50, 10, 80, 20, 90, 30, 70, 40, 60, 100],
            "TransactionAmt": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ProductCD": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
            "isFraud": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        }
    )


def test_time_split_order_and_ranges() -> None:
    full = _full_fixture()
    splits = {"train": full.iloc[:4], "valid": full.iloc[4:7], "test": full.iloc[7:]}

    time_splits = build_time_splits(splits)

    assert time_splits["train"]["TransactionDT"].is_monotonic_increasing
    assert time_splits["train"]["TransactionDT"].max() <= time_splits["valid"]["TransactionDT"].min()
    assert time_splits["valid"]["TransactionDT"].max() <= time_splits["test"]["TransactionDT"].min()


def test_data_summary_positive_rate() -> None:
    full = _full_fixture()
    time_splits = build_time_splits({"train": full.iloc[:6], "valid": full.iloc[6:8], "test": full.iloc[8:]})
    summary = build_time_split_summary(time_splits).set_index("split")

    assert summary.loc["train", "positive_rate"] == time_splits["train"]["isFraud"].mean()


def test_missing_transaction_dt_errors() -> None:
    full = _full_fixture().drop(columns=["TransactionDT"])

    try:
        validate_full_sample(full, len(full))
    except ValueError as error:
        assert "TransactionDT" in str(error)
    else:
        raise AssertionError("Expected missing TransactionDT to fail.")


def test_duplicate_transaction_id_errors() -> None:
    full = _full_fixture()
    full.loc[1, "TransactionID"] = full.loc[0, "TransactionID"]

    try:
        validate_full_sample(full, len(full))
    except ValueError as error:
        assert "TransactionID" in str(error)
    else:
        raise AssertionError("Expected duplicate TransactionID to fail.")


def test_threshold_metrics_share_same_auc_scores() -> None:
    comparison = build_threshold_comparison(
        [0, 1, 0, 1],
        [0.1, 0.8, 0.3, 0.7],
        0.5,
        0.2,
        "lightgbm",
        "transaction_basic",
        10,
        1,
    )

    assert comparison["pr_auc"].nunique() == 1
    assert comparison["roc_auc"].nunique() == 1


def test_vs_random_comparison_structure() -> None:
    random_results = pd.DataFrame(
        [
            {
                "threshold_type": "default_0_5",
                "threshold": 0.5,
                "pr_auc": 0.6,
                "roc_auc": 0.8,
                "precision": 0.5,
                "recall": 0.4,
                "f1": 0.45,
                "cost": 10,
                "fp": 1,
                "fn": 2,
                "tp": 3,
                "tn": 4,
                "n_samples": 10,
                "positive_support": 5,
            }
        ]
    )
    time_results = random_results.copy()
    comparison = build_vs_random_comparison(random_results, time_results)

    assert {
        "split_strategy",
        "threshold_type",
        "threshold",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "cost",
    }.issubset(comparison.columns)


def _write_cli_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    data_dir = tmp_path / "stage2_formal"
    data_dir.mkdir()
    full = pd.DataFrame(
        {
            "TransactionID": list(range(20)),
            "TransactionDT": list(range(20)),
            "TransactionAmt": [float(i) for i in range(20)],
            "ProductCD": ["a", "b"] * 10,
            "isFraud": [0, 0, 0, 1, 0] * 4,
        }
    )
    full.iloc[:12].to_parquet(data_dir / "train.parquet", index=False)
    full.iloc[12:16].to_parquet(data_dir / "valid.parquet", index=False)
    full.iloc[16:].to_parquet(data_dir / "test.parquet", index=False)
    metadata_path = data_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "stage": "stage2_formal",
                "train_rows": 12,
                "valid_rows": 4,
                "test_rows": 4,
                "target_column": "isFraud",
                "id_column": "TransactionID",
                "time_column": "TransactionDT",
                "numeric_columns": ["TransactionAmt"],
                "categorical_columns": ["ProductCD"],
            }
        ),
        encoding="utf-8",
    )
    feature_groups_path = tmp_path / "feature_groups.json"
    feature_groups_path.write_text(
        json.dumps({"groups": {"transaction_basic": {"features": ["TransactionAmt", "ProductCD"]}}}),
        encoding="utf-8",
    )
    random_results_path = tmp_path / "stage3_threshold_comparison_test.csv"
    random_results_path.write_text(
        "threshold_type,threshold,pr_auc,roc_auc,precision,recall,f1,cost,fp,fn,tp,tn,n_samples,positive_support\n"
        "default_0_5,0.5,0.7,0.8,0.5,0.5,0.5,10,1,1,1,17,20,2\n"
        "cost_sensitive,0.1,0.7,0.8,0.4,0.9,0.55,8,3,0,2,15,20,2\n",
        encoding="utf-8",
    )
    selected_path = tmp_path / "stage3_selected_threshold.json"
    selected_path.write_text(
        json.dumps({"selected_model": "logistic_regression", "selected_feature_group": "transaction_basic"}),
        encoding="utf-8",
    )
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text(
        """
random_seed: 42
default_threshold: 0.5
models:
  logistic_regression:
    class_weight: balanced
    solver: saga
    max_iter: 200
    n_jobs: -1
preprocessing:
  numeric_imputer: median
  categorical_imputer: constant
  categorical_fill_value: "__MISSING__"
  tree_categorical_fill_value: "**MISSING**"
  one_hot_handle_unknown: ignore
  one_hot_min_frequency: 1
  one_hot_max_categories: 20
  scale_numeric_for_logistic: true
threshold_optimization:
  minimum_threshold: 0.05
  maximum_threshold: 0.95
  threshold_step: 0.01
  false_negative_cost: 10
  false_positive_cost: 1
""",
        encoding="utf-8",
    )
    return data_dir, metadata_path, feature_groups_path, random_results_path, selected_path, config_path


def test_cli_end_to_end_and_agg_backend(tmp_path: Path) -> None:
    data_dir, metadata_path, feature_groups_path, random_results_path, selected_path, config_path = _write_cli_fixture(
        tmp_path
    )
    output_dir = tmp_path / "reports"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd())

    result = subprocess.run(
        [
            *BASE_COMMAND,
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--metadata",
            str(metadata_path),
            "--feature-groups",
            str(feature_groups_path),
            "--random-threshold-results",
            str(random_results_path),
            "--selected-threshold",
            str(selected_path),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "time_split",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert matplotlib.get_backend().lower() == "agg"
    assert (output_dir / "tables" / "time_split_threshold_comparison_test.csv").exists()
    assert (output_dir / "tables" / "time_split_vs_random_comparison.csv").exists()
    assert (output_dir / "predictions" / "time_split_test_predictions.parquet").exists()


def test_time_split_predictions_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "reports/predictions/time_split_test_predictions.parquet"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
