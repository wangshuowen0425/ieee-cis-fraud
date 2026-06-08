from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_loader import (
    inspect_csv_schema,
    load_selected_transaction_data,
    load_yaml_config,
    resolve_raw_paths,
    select_existing_columns,
    validate_raw_files,
)


def write_config(project_root: Path) -> Path:
    config_dir = project_root / "configs"
    config_dir.mkdir()
    config_path = config_dir / "data_config.yaml"
    config = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "transaction_file": "train_transaction.csv",
        "identity_file": "train_identity.csv",
        "target_column": "isFraud",
        "id_column": "TransactionID",
        "time_column": "TransactionDT",
        "random_seed": 42,
        "smoke_test_rows": 50000,
        "target_sample_size": 120000,
        "train_ratio": 0.60,
        "valid_ratio": 0.20,
        "test_ratio": 0.20,
        "sample_strategy": "stratified_preserve_ratio",
        "output_format": "parquet",
        "transaction_columns": ["TransactionAmt", "ProductCD", "card1", "V1"],
        "identity_columns": ["id_01", "DeviceType"],
        "missing_indicator_candidates": ["DeviceType", "id_01", "dist1"],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def write_raw_files(project_root: Path, transaction: bool = True, identity: bool = True) -> None:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    if transaction:
        (raw_dir / "train_transaction.csv").write_text("TransactionID,isFraud\n1,0\n", encoding="utf-8")
    if identity:
        (raw_dir / "train_identity.csv").write_text("TransactionID,id_01\n1,0\n", encoding="utf-8")


def test_config_file_can_be_loaded(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    config = load_yaml_config(config_path)

    assert config["raw_dir"] == "data/raw"
    assert config["random_seed"] == 42


def test_validate_raw_files_success_when_both_files_exist(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    write_raw_files(tmp_path)
    config = load_yaml_config(config_path)

    paths = validate_raw_files(config)

    assert paths["transaction"].name == "train_transaction.csv"
    assert paths["identity"].name == "train_identity.csv"


def test_missing_transaction_file_raises_clear_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    write_raw_files(tmp_path, transaction=False, identity=True)
    config = load_yaml_config(config_path)

    with pytest.raises(FileNotFoundError, match="Missing transaction raw file"):
        validate_raw_files(config)


def test_missing_identity_file_raises_clear_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    write_raw_files(tmp_path, transaction=True, identity=False)
    config = load_yaml_config(config_path)

    with pytest.raises(FileNotFoundError, match="Missing identity raw file"):
        validate_raw_files(config)


def test_path_resolution_uses_config_location_not_personal_absolute_path(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = load_yaml_config(config_path)

    paths = resolve_raw_paths(config)

    assert paths["transaction"] == tmp_path / "data" / "raw" / "train_transaction.csv"
    assert paths["identity"] == tmp_path / "data" / "raw" / "train_identity.csv"


def test_inspect_csv_schema_reads_small_sample(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("TransactionID,isFraud\n1,0\n2,1\n", encoding="utf-8")

    schema = inspect_csv_schema(csv_path, nrows=1)

    assert schema == {"TransactionID": "int64", "isFraud": "int64"}


def test_select_existing_columns_logs_missing_candidates(caplog: pytest.LogCaptureFixture) -> None:
    existing, missing = select_existing_columns(
        {"TransactionAmt", "ProductCD"},
        ["TransactionAmt", "missing_col"],
        context="transaction feature",
    )

    assert existing == ["TransactionAmt"]
    assert missing == ["missing_col"]
    assert "missing_col" in caplog.text


def test_load_selected_transaction_data_only_reads_actual_existing_fields(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
            "TransactionDT": [10, 20],
            "TransactionAmt": [100.0, 200.0],
            "ProductCD": ["W", "C"],
            "V1": [999, 888],
        }
    ).to_csv(raw_dir / "train_transaction.csv", index=False)
    (raw_dir / "train_identity.csv").write_text("TransactionID,id_01\n1,0\n", encoding="utf-8")
    config = load_yaml_config(config_path)
    config["transaction_columns"] = ["TransactionAmt", "ProductCD", "card1"]

    transaction, missing = load_selected_transaction_data(config)

    assert list(transaction.columns) == [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
    ]
    assert "V1" not in transaction.columns
    assert missing == ["card1"]
