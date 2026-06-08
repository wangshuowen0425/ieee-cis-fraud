"""CLI tests for stage 0 checks and stage 1 smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import run
import sys

import pandas as pd
import yaml


BASE_COMMAND: tuple[str, ...] = (sys.executable, "-m", "src.run_experiments")


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    reports_dir = tmp_path / "reports"
    feature_groups_path = reports_dir / "tables" / "feature_groups.json"
    feature_groups_path.parent.mkdir(parents=True)

    train_df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "amount": [1.0, 2.0, None, 10.0, 11.0, 12.0],
            "product": ["a", "a", "b", "b", None, "c"],
            "isFraud": [0, 0, 0, 1, 1, 1],
        }
    )
    valid_df = pd.DataFrame(
        {
            "TransactionID": [7, 8, 9, 10],
            "TransactionDT": [70, 80, 90, 100],
            "amount": [1.5, 2.5, 10.5, 11.5],
            "product": ["a", "new", "b", "c"],
            "isFraud": [0, 0, 1, 1],
        }
    )
    test_df = valid_df.copy()

    train_df.to_parquet(data_dir / "train.parquet", index=False)
    valid_df.to_parquet(data_dir / "valid.parquet", index=False)
    test_df.to_parquet(data_dir / "test.parquet", index=False)

    metadata_path = data_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "numeric_columns": ["amount"],
                "categorical_columns": ["product"],
                "target_column": "isFraud",
                "id_column": "TransactionID",
                "time_column": "TransactionDT",
            }
        ),
        encoding="utf-8",
    )
    feature_groups_path.write_text(
        json.dumps({"transaction_basic": ["amount", "product"]}),
        encoding="utf-8",
    )
    return data_dir, metadata_path, feature_groups_path, reports_dir


def test_help_succeeds() -> None:
    result = run([*BASE_COMMAND, "--help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "--dry-run" in result.stdout


def test_list_models_succeeds() -> None:
    result = run(
        [*BASE_COMMAND, "--config", "configs/model_config.yaml", "--list-models"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dummy" in result.stdout
    assert "LightGBM available:" in result.stdout


def test_dry_run_valid_config_succeeds() -> None:
    result = run(
        [*BASE_COMMAND, "--config", "configs/model_config.yaml", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Dry run passed" in result.stdout


def test_invalid_model_name_fails(tmp_path: Path) -> None:
    with Path("configs/model_config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["models"]["not_a_model"] = {}

    invalid_config_path = tmp_path / "invalid_model_config.yaml"
    invalid_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run(
        [*BASE_COMMAND, "--config", str(invalid_config_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unsupported model name" in result.stderr


def test_cli_stage1_end_to_end_with_fixture_data(tmp_path: Path) -> None:
    data_dir, metadata_path, feature_groups_path, reports_dir = _write_fixture_files(tmp_path)

    result = run(
        [
            *BASE_COMMAND,
            "--config",
            "configs/model_config.yaml",
            "--data-dir",
            str(data_dir),
            "--metadata",
            str(metadata_path),
            "--feature-groups",
            str(feature_groups_path),
            "--feature-group",
            "transaction_basic",
            "--models",
            "dummy",
            "logistic_regression",
            "--output-dir",
            str(reports_dir),
            "--run-name",
            "stage1_smoke",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    results_path = reports_dir / "tables" / "smoke_model_results.csv"
    pr_curve_path = reports_dir / "figures" / "smoke_pr_curve.png"
    confusion_path = reports_dir / "figures" / "smoke_confusion_matrix_logistic.png"
    results = pd.read_csv(results_path)

    assert {"model_name", "feature_group", "split", "pr_auc", "roc_auc"}.issubset(results.columns)
    assert pr_curve_path.exists()
    assert confusion_path.exists()


def test_missing_processed_parquet_fails_clearly(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    feature_groups_path = tmp_path / "feature_groups.json"
    metadata_path.write_text(
        json.dumps({"numeric_columns": ["amount"], "categorical_columns": ["product"]}),
        encoding="utf-8",
    )
    feature_groups_path.write_text(
        json.dumps({"transaction_basic": ["amount", "product"]}),
        encoding="utf-8",
    )

    result = run(
        [
            *BASE_COMMAND,
            "--data-dir",
            str(tmp_path / "missing_processed"),
            "--metadata",
            str(metadata_path),
            "--feature-groups",
            str(feature_groups_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "train parquet file" in result.stderr
