"""Tests for Stage 3 threshold analysis helpers and CLI flow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.threshold_analysis import (
    apply_fixed_threshold,
    assign_error_type,
    calculate_business_cost,
    generate_threshold_grid,
    search_optimal_threshold,
)


BASE_COMMAND: tuple[str, ...] = (sys.executable, "-m", "src.run_experiments")


def test_threshold_grid_includes_bounds() -> None:
    assert generate_threshold_grid(0.05, 0.95, 0.01)[[0, -1][0]] == 0.05
    grid = generate_threshold_grid(0.05, 0.95, 0.01)
    assert grid[-1] == 0.95
    assert len(grid) == 91


def test_cost_formula() -> None:
    assert calculate_business_cost(fn=3, fp=4, false_negative_cost=10, false_positive_cost=1) == 34


def test_minimum_cost_threshold_is_selected() -> None:
    selected, search = search_optimal_threshold([0, 1], [0.2, 0.8], [0.1, 0.5, 0.9])
    min_cost = search["cost"].min()

    assert search.loc[search["threshold"] == selected, "cost"].iloc[0] == min_cost


def test_ties_prefer_recall_then_precision_then_closest_to_half() -> None:
    selected_recall, _ = search_optimal_threshold([0, 1], [0.6, 0.6], [0.5, 0.7])
    selected_precision, _ = search_optimal_threshold([0, 1], [0.4, 0.6], [0.3, 0.5])
    selected_closest, _ = search_optimal_threshold([0, 1], [0.2, 0.8], [0.2, 0.4, 0.6])

    assert selected_recall == 0.5
    assert selected_precision == 0.5
    assert selected_closest == 0.4


def test_error_type_labels() -> None:
    labels = assign_error_type([1, 0, 0, 1], [1, 0, 1, 0])

    assert labels.tolist() == ["TP", "TN", "FP", "FN"]


def _write_stage3_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = tmp_path / "stage2_formal"
    tables_dir = tmp_path / "reports" / "tables"
    data_dir.mkdir()
    tables_dir.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [1, 2, 3, 4, 5, 6],
            "TransactionAmt": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "ProductCD": ["a", "a", "b", "b", "c", "c"],
            "isFraud": [0, 0, 0, 1, 1, 1],
        }
    )
    valid = pd.DataFrame(
        {
            "TransactionID": [7, 8, 9, 10],
            "TransactionDT": [7, 8, 9, 10],
            "TransactionAmt": [1.5, 2.5, 10.5, 11.5],
            "ProductCD": ["a", "new", "b", "c"],
            "isFraud": [0, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "TransactionID": [11, 12, 13, 14],
            "TransactionDT": [11, 12, 13, 14],
            "TransactionAmt": [1.2, 2.2, 10.2, 12.2],
            "ProductCD": ["a", "b", "b", "c"],
            "isFraud": [0, 0, 1, 1],
        }
    )
    train.to_parquet(data_dir / "train.parquet", index=False)
    valid.to_parquet(data_dir / "valid.parquet", index=False)
    test.to_parquet(data_dir / "test.parquet", index=False)
    metadata = {
        "stage": "stage2_formal",
        "random_seed": 42,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "target_column": "isFraud",
        "id_column": "TransactionID",
        "time_column": "TransactionDT",
        "numeric_columns": ["TransactionAmt"],
        "categorical_columns": ["ProductCD"],
    }
    metadata_path = data_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    feature_groups_path = tables_dir / "feature_groups.json"
    feature_groups_path.write_text(
        json.dumps({"groups": {"transaction_basic": {"features": ["TransactionAmt", "ProductCD"]}}}),
        encoding="utf-8",
    )
    stage2_columns = (
        "run_name,threshold,random_seed,model_name,feature_group,split,pr_auc,roc_auc,"
        "actual_model_name,selected_feature_group,validation_selection_score\n"
    )
    (tables_dir / "stage2_final_test.csv").write_text(
        stage2_columns
        + "stage2,0.5,42,logistic_regression,transaction_basic,test,0.9,0.9,"
        + "logistic_regression,transaction_basic,0.88\n",
        encoding="utf-8",
    )
    (tables_dir / "stage2_ablation_valid.csv").write_text(
        "run_name,threshold,random_seed,model_name,feature_group,split,pr_auc,actual_model_name\n"
        "stage2,0.5,42,logistic_regression,transaction_basic,valid,0.88,logistic_regression\n",
        encoding="utf-8",
    )
    (tables_dir / "stage2_model_comparison_valid.csv").write_text(
        "run_name,threshold,random_seed,model_name,feature_group,split,pr_auc,actual_model_name\n"
        "stage2,0.5,42,logistic_regression,transaction_basic,valid,0.88,logistic_regression\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text(
        """
random_seed: 42
default_threshold: 0.5
primary_metric: pr_auc
threshold_optimization:
  enabled: true
  selection_split: validation
  minimum_threshold: 0.05
  maximum_threshold: 0.95
  threshold_step: 0.01
  false_negative_cost: 10
  false_positive_cost: 1
  tie_breaker: [recall, precision, closest_to_0_5]
models:
  logistic_regression:
    class_weight: balanced
    solver: saga
    max_iter: 200
    n_jobs: -1
  dummy:
    strategy: prior
  lightgbm:
    enabled: false
  hist_gradient_boosting:
    enabled: false
  random_forest:
    enabled: false
fallback_model: hist_gradient_boosting
preprocessing:
  numeric_imputer: median
  categorical_imputer: constant
  categorical_fill_value: "__MISSING__"
  tree_categorical_fill_value: "**MISSING**"
  one_hot_handle_unknown: ignore
  one_hot_min_frequency: 1
  one_hot_max_categories: 20
  scale_numeric_for_logistic: true
metrics: [pr_auc, roc_auc, precision, recall, f1, mcc, accuracy]
positive_label: 1
feature_groups: [transaction_basic]
""",
        encoding="utf-8",
    )
    return data_dir, metadata_path, feature_groups_path, config_path


def test_cli_threshold_analysis_fixture(tmp_path: Path) -> None:
    data_dir, metadata_path, feature_groups_path, config_path = _write_stage3_fixture(tmp_path)
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
            "--mode",
            "threshold-analysis",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "stage3",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    comparison = pd.read_csv(output_dir / "tables" / "stage3_threshold_comparison_test.csv")
    predictions = pd.read_parquet(output_dir / "predictions" / "stage3_test_predictions.parquet")
    selected = json.loads((output_dir / "tables" / "stage3_selected_threshold.json").read_text())

    assert set(comparison["threshold_type"]) == {"default_0_5", "cost_sensitive"}
    assert comparison["pr_auc"].nunique() == 1
    assert comparison["roc_auc"].nunique() == 1
    assert selected["selection_split"] == "validation"
    assert {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "prediction_score",
        "prediction_at_0_5",
        "prediction_at_selected_threshold",
        "error_type_at_0_5",
        "error_type_at_selected_threshold",
    }.issubset(predictions.columns)
    assert not predictions.isin(["TODO", "PLACEHOLDER"]).any().any()


def test_predictions_are_gitignored() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "reports/predictions/stage3_test_predictions.parquet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
