from pathlib import Path

import pandas as pd

from src.data_pipeline import (
    add_missing_features,
    build_data_summary,
    build_metadata,
    merge_transaction_identity,
    save_processed_splits,
    stratified_sample,
    stratified_train_valid_test_split,
)
from src.feature_groups import (
    build_transaction_basic_group,
    build_transaction_identity_group,
    build_transaction_identity_missing_group,
    validate_feature_groups,
)


def make_config(tmp_path: Path) -> dict:
    return {
        "_project_root": tmp_path,
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "transaction_file": "train_transaction.csv",
        "identity_file": "train_identity.csv",
        "target_column": "isFraud",
        "id_column": "TransactionID",
        "time_column": "TransactionDT",
        "random_seed": 42,
        "smoke_test_rows": 50,
        "target_sample_size": 120000,
        "train_ratio": 0.60,
        "valid_ratio": 0.20,
        "test_ratio": 0.20,
        "sample_strategy": "stratified_preserve_ratio",
        "output_format": "parquet",
        "transaction_columns": ["TransactionAmt", "ProductCD", "card1", "dist1", "P_emaildomain"],
        "identity_columns": ["id_01", "id_02", "DeviceType", "DeviceInfo"],
        "missing_indicator_candidates": ["DeviceInfo", "DeviceType", "id_01", "id_02", "dist1"],
    }


def make_transaction(n_rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(1, n_rows + 1),
            "isFraud": [1 if index < n_rows // 5 else 0 for index in range(n_rows)],
            "TransactionDT": range(1000, 1000 + n_rows),
            "TransactionAmt": [float(index) for index in range(n_rows)],
            "ProductCD": ["W" if index % 2 == 0 else "C" for index in range(n_rows)],
            "dist1": [None if index % 3 == 0 else float(index) for index in range(n_rows)],
            "P_emaildomain": ["gmail.com" if index % 4 else None for index in range(n_rows)],
        }
    )


def make_identity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "id_01": [None, 1.0, 2.0, 3.0, 4.0],
            "id_02": [10.0, None, 12.0, 13.0, 14.0],
            "DeviceType": ["mobile", None, "desktop", "mobile", "desktop"],
            "DeviceInfo": [None, "ios", "chrome", "android", "edge"],
        }
    )


def test_transaction_identity_left_join() -> None:
    merged = merge_transaction_identity(make_transaction(6), make_identity())

    assert len(merged) == 6
    assert merged.loc[merged["TransactionID"] == 6, "DeviceType"].isna().all()


def test_duplicate_transaction_id_raises_clear_error() -> None:
    transaction = make_transaction(6)
    transaction.loc[1, "TransactionID"] = 1

    try:
        merge_transaction_identity(transaction, make_identity())
    except ValueError as exc:
        assert "Duplicate TransactionID values in transaction data" in str(exc)
    else:
        raise AssertionError("Expected duplicate TransactionID error")


def test_stratified_sample_preserves_class_ratio() -> None:
    data = make_transaction(100)

    sampled = stratified_sample(data, sample_size=50, random_state=42)

    assert len(sampled) == 50
    assert abs(sampled["isFraud"].mean() - data["isFraud"].mean()) < 0.03


def test_stratified_split_is_60_20_20_and_ids_do_not_overlap() -> None:
    data = make_transaction(100)

    train, valid, test = stratified_train_valid_test_split(data)

    assert (len(train), len(valid), len(test)) == (60, 20, 20)
    assert set(train["TransactionID"]).isdisjoint(valid["TransactionID"])
    assert set(train["TransactionID"]).isdisjoint(test["TransactionID"])
    assert set(valid["TransactionID"]).isdisjoint(test["TransactionID"])


def test_missing_count_and_indicators_are_correct() -> None:
    data = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
            "TransactionDT": [10, 20],
            "DeviceInfo": [None, "ios"],
            "dist1": [1.0, None],
        }
    )

    result, indicators = add_missing_features(data, ["DeviceInfo", "dist1", "missing_col"])

    assert indicators == ["DeviceInfo_is_missing", "dist1_is_missing"]
    assert result["missing_count"].tolist() == [1, 1]
    assert result["DeviceInfo_is_missing"].tolist() == [1, 0]
    assert result["dist1_is_missing"].tolist() == [0, 1]


def test_feature_groups_exclude_non_features_and_are_nested(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "id_01",
        "DeviceType",
        "missing_count",
        "DeviceType_is_missing",
    ]

    basic = build_transaction_basic_group(columns, config)
    identity = build_transaction_identity_group(basic, columns, config)
    missing = build_transaction_identity_missing_group(identity, ["DeviceType_is_missing"])
    groups = {
        "transaction_basic": basic,
        "transaction_identity": identity,
        "transaction_identity_missing": missing,
    }
    validate_feature_groups(groups)

    assert "TransactionID" not in missing
    assert "isFraud" not in missing
    assert "TransactionDT" not in missing
    assert set(basic).issubset(identity)
    assert set(identity).issubset(missing)


def test_parquet_save_can_be_read_back(tmp_path: Path) -> None:
    data = make_transaction(20)
    train, valid, test = data.iloc[:12], data.iloc[12:16], data.iloc[16:]

    paths = save_processed_splits(train, valid, test, tmp_path / "processed")

    assert len(pd.read_parquet(paths["train"])) == 12
    assert len(pd.read_parquet(paths["valid"])) == 4
    assert len(pd.read_parquet(paths["test"])) == 4


def test_metadata_numbers_come_from_real_data(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "train_transaction.csv").write_text("placeholder", encoding="utf-8")
    (raw_dir / "train_identity.csv").write_text("placeholder", encoding="utf-8")
    transaction = make_transaction(100)
    identity = make_identity()
    merged = merge_transaction_identity(transaction, identity)
    sampled = stratified_sample(merged, sample_size=50, random_state=42)
    sampled, indicators = add_missing_features(sampled, config["missing_indicator_candidates"])
    train, valid, test = stratified_train_valid_test_split(sampled)
    splits = {"train": train, "valid": valid, "test": test}
    outputs = save_processed_splits(train, valid, test, tmp_path / "data" / "processed")
    feature_groups = {
        "transaction_basic": ["TransactionAmt", "ProductCD"],
        "transaction_identity": ["TransactionAmt", "ProductCD", "id_01"],
        "transaction_identity_missing": ["TransactionAmt", "ProductCD", "id_01", "missing_count"],
    }

    metadata = build_metadata(
        config,
        transaction,
        identity,
        merged,
        sampled,
        splits,
        feature_groups,
        missing_transaction_columns=["card1"],
        missing_identity_columns=[],
        missing_indicator_columns=indicators,
        output_paths=outputs,
    )
    summary = build_data_summary(splits)

    assert metadata["transaction_rows"] == 100
    assert metadata["identity_rows"] == 5
    assert metadata["sampled_rows"] == 50
    assert metadata["stage"] == "stage1_smoke"
    assert metadata["sample_size"] == 50
    assert metadata["target_column"] == "isFraud"
    assert metadata["id_column"] == "TransactionID"
    assert metadata["time_column"] == "TransactionDT"
    assert metadata["train_rows"] == 30
    assert metadata["valid_rows"] == 10
    assert metadata["test_rows"] == 10
    assert metadata["original_fraud_rate"] == metadata["original_fraud_ratio"]
    assert metadata["sampled_fraud_rate"] == metadata["sampled_fraud_ratio"]
    assert metadata["train_fraud_rate"] == metadata["splits"]["train"]["fraud_ratio"]
    assert isinstance(metadata["numeric_columns"], list)
    assert isinstance(metadata["categorical_columns"], list)
    assert metadata["missing_raw_columns"] == ["card1"]
    assert set(metadata["missing_source_columns"]).issubset(
        {"DeviceInfo", "DeviceType", "id_01", "id_02", "dist1"}
    )
    assert len(metadata["source_files"]) == 2
    assert metadata["splits"]["train"]["rows"] == int(summary.loc[summary["split"] == "train", "rows"].iloc[0])
