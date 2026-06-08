"""Data pipeline for IEEE-CIS fraud smoke and formal datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_quality import (
    build_data_quality_report,
    build_feature_profile,
    fraud_rate_check_passed,
)
from src.data_loader import (
    load_selected_identity_data,
    load_selected_transaction_data,
    load_yaml_config,
    resolve_raw_paths,
)
from src.feature_groups import (
    build_transaction_basic_group,
    build_transaction_identity_group,
    build_transaction_identity_missing_group,
    save_feature_groups,
    validate_feature_groups,
)


LOGGER = logging.getLogger(__name__)


def _project_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config.get("_project_root", Path.cwd())) / path


def merge_transaction_identity(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    id_column: str = "TransactionID",
) -> pd.DataFrame:
    """Left join transaction and identity data, failing clearly on duplicate IDs."""
    duplicate_transaction = transaction.loc[transaction[id_column].duplicated(), id_column].head().tolist()
    duplicate_identity = identity.loc[identity[id_column].duplicated(), id_column].head().tolist()
    if duplicate_transaction:
        raise ValueError(f"Duplicate TransactionID values in transaction data: {duplicate_transaction}")
    if duplicate_identity:
        raise ValueError(f"Duplicate TransactionID values in identity data: {duplicate_identity}")

    try:
        return transaction.merge(identity, on=id_column, how="left", validate="one_to_one")
    except pd.errors.MergeError as exc:
        raise ValueError(f"Failed one-to-one merge on {id_column}: {exc}") from exc


def stratified_sample(
    data: pd.DataFrame,
    target_column: str = "isFraud",
    sample_size: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified sample up to sample_size rows while preserving target ratio."""
    if len(data) <= sample_size:
        return data.copy()
    _, sampled = train_test_split(
        data,
        test_size=sample_size,
        random_state=random_state,
        stratify=data[target_column],
    )
    return sampled.sort_index().reset_index(drop=True)


def add_missing_features(
    data: pd.DataFrame,
    missing_candidates: list[str],
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add missing_count and *_is_missing indicators for available candidate fields."""
    exclude = set(exclude_columns or ["TransactionID", "isFraud", "TransactionDT"])
    result = data.copy()
    count_columns = [column for column in result.columns if column not in exclude]
    result["missing_count"] = result[count_columns].isna().sum(axis=1)

    indicator_columns: list[str] = []
    for column in missing_candidates:
        if column in result.columns and column not in exclude:
            indicator = f"{column}_is_missing"
            result[indicator] = result[column].isna().astype("int8")
            indicator_columns.append(indicator)
        else:
            LOGGER.warning("Missing indicator candidate skipped: %s", column)
    return result, indicator_columns


def stratified_train_valid_test_split(
    data: pd.DataFrame,
    target_column: str = "isFraud",
    id_column: str = "TransactionID",
    train_ratio: float = 0.60,
    valid_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into stratified train, valid, and test frames."""
    ratio_total = train_ratio + valid_ratio + test_ratio
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_total}")

    train, temp = train_test_split(
        data,
        train_size=train_ratio,
        random_state=random_state,
        stratify=data[target_column],
    )
    relative_valid = valid_ratio / (valid_ratio + test_ratio)
    valid, test = train_test_split(
        temp,
        train_size=relative_valid,
        random_state=random_state,
        stratify=temp[target_column],
    )

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    _assert_disjoint_ids(train, valid, test, id_column)
    return train, valid, test


def _assert_disjoint_ids(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, id_column: str) -> None:
    train_ids = set(train[id_column])
    valid_ids = set(valid[id_column])
    test_ids = set(test[id_column])
    if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
        raise ValueError("TransactionID values overlap across train, valid, and test splits")


def infer_column_types(
    data: pd.DataFrame,
    target_column: str = "isFraud",
    id_column: str = "TransactionID",
    time_column: str = "TransactionDT",
) -> dict[str, list[str]]:
    """Infer numeric and categorical feature columns."""
    excluded = {target_column, id_column, time_column}
    feature_columns = [column for column in data.columns if column not in excluded]
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(data[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}


def save_processed_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: str | Path,
) -> dict[str, Path]:
    """Save processed train, valid, and test splits as parquet files."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_dir / "train.parquet",
        "valid": output_dir / "valid.parquet",
        "test": output_dir / "test.parquet",
    }
    train.to_parquet(outputs["train"], index=False)
    valid.to_parquet(outputs["valid"], index=False)
    test.to_parquet(outputs["test"], index=False)
    LOGGER.info("Saved processed splits to %s", output_dir)
    return outputs


def _save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _split_stats(data: pd.DataFrame, target_column: str) -> dict[str, Any]:
    fraud_count = int(data[target_column].sum())
    return {
        "rows": int(len(data)),
        "fraud_count": fraud_count,
        "fraud_ratio": float(data[target_column].mean()) if len(data) else 0.0,
        "column_count": int(data.shape[1]),
    }


def build_data_summary(splits: dict[str, pd.DataFrame], target_column: str = "isFraud") -> pd.DataFrame:
    """Build split-level row, fraud, ratio, and column summary."""
    rows = []
    for split_name, split in splits.items():
        stats = _split_stats(split, target_column)
        rows.append({"split": split_name, **stats})
    return pd.DataFrame(rows)


def build_metadata(
    config: dict[str, Any],
    stage_name: str,
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    merged: pd.DataFrame,
    sampled: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    missing_transaction_columns: list[str],
    missing_identity_columns: list[str],
    missing_indicator_columns: list[str],
    skipped_missing_indicators: list[str],
    requested_sample_size: int,
    fraud_rate_check: bool,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build metadata using actual dataframes and output paths."""
    target_column = config["target_column"]
    id_column = config["id_column"]
    time_column = config["time_column"]
    raw_paths = resolve_raw_paths(config)
    split_stats = {name: _split_stats(split, target_column) for name, split in splits.items()}
    column_types = infer_column_types(
        sampled,
        target_column=target_column,
        id_column=id_column,
        time_column=time_column,
    )
    missing_raw_columns = sorted(missing_transaction_columns + missing_identity_columns)
    missing_source_columns = [
        column.removesuffix("_is_missing") for column in missing_indicator_columns
    ]
    return {
        "stage": stage_name,
        "sample_strategy": config.get("sample_strategy"),
        "random_seed": int(config["random_seed"]),
        "requested_sample_size": int(requested_sample_size),
        "actual_sample_size": int(len(sampled)),
        "sample_size": int(len(sampled)),
        "target_column": target_column,
        "id_column": id_column,
        "time_column": time_column,
        "train_rows": split_stats["train"]["rows"],
        "valid_rows": split_stats["valid"]["rows"],
        "test_rows": split_stats["test"]["rows"],
        "original_fraud_rate": float(merged[target_column].mean()) if len(merged) else 0.0,
        "sampled_fraud_rate": float(sampled[target_column].mean()) if len(sampled) else 0.0,
        "train_fraud_rate": split_stats["train"]["fraud_ratio"],
        "valid_fraud_rate": split_stats["valid"]["fraud_ratio"],
        "test_fraud_rate": split_stats["test"]["fraud_ratio"],
        "numeric_columns": column_types["numeric_columns"],
        "categorical_columns": column_types["categorical_columns"],
        "missing_source_columns": missing_source_columns,
        "missing_raw_columns": missing_raw_columns,
        "source_feature_columns": feature_groups["transaction_identity_missing"],
        "feature_group_file": str(output_paths.get("feature_groups", "")),
        "data_quality_file": str(output_paths.get("data_quality", "")),
        "feature_profile_file": str(output_paths.get("feature_profile", "")),
        "created_missing_indicators": missing_indicator_columns,
        "skipped_missing_indicators": skipped_missing_indicators,
        "fraud_rate_check_passed": fraud_rate_check,
        "source_files": [str(path) for path in raw_paths.values()],
        "source_file_map": {name: str(path) for name, path in raw_paths.items()},
        "transaction_rows": int(len(transaction)),
        "identity_rows": int(len(identity)),
        "merged_rows": int(len(merged)),
        "sampled_rows": int(len(sampled)),
        "identity_match_rate": _identity_match_rate(transaction, identity, config["id_column"]),
        "original_fraud_ratio": float(merged[target_column].mean()) if len(merged) else 0.0,
        "sampled_fraud_ratio": float(sampled[target_column].mean()) if len(sampled) else 0.0,
        "splits": split_stats,
        "column_types": column_types,
        "feature_groups": feature_groups,
        "missing_columns": {
            "transaction": missing_transaction_columns,
            "identity": missing_identity_columns,
        },
        "missing_indicator_columns": missing_indicator_columns,
        "output_paths": {name: str(path) for name, path in output_paths.items()},
    }


def _identity_match_rate(transaction: pd.DataFrame, identity: pd.DataFrame, id_column: str) -> float:
    if transaction.empty:
        return 0.0
    return float(transaction[id_column].isin(set(identity[id_column])).mean())


def _stage_settings(config: dict[str, Any], stage: str) -> dict[str, Any]:
    processed_root = _project_path(config, config["processed_dir"])
    if stage == "smoke":
        return {
            "stage_name": "stage1_smoke",
            "sample_size": int(config["smoke_test_rows"]),
            "processed_dir": processed_root,
            "summary_file": "data_summary.csv",
            "feature_profile_file": None,
            "data_quality_file": None,
        }
    if stage == "formal":
        return {
            "stage_name": "stage2_formal",
            "sample_size": int(config.get("formal_sample_size", config["target_sample_size"])),
            "processed_dir": processed_root / str(config["formal_output_subdir"]),
            "summary_file": "stage2_data_summary.csv",
            "feature_profile_file": "stage2_feature_profile.csv",
            "data_quality_file": "stage2_data_quality.json",
        }
    raise ValueError(f"Unsupported stage: {stage}")


def _candidate_missing_sources(created_missing_indicators: list[str]) -> set[str]:
    return {column.removesuffix("_is_missing") for column in created_missing_indicators}


def run_pipeline(config: dict[str, Any], stage: str = "smoke") -> dict[str, Any]:
    """Run the configured data pipeline stage."""
    settings = _stage_settings(config, stage)

    transaction, missing_transaction = load_selected_transaction_data(config)
    identity, missing_identity = load_selected_identity_data(config)
    identity_match_rate = _identity_match_rate(transaction, identity, config["id_column"])
    merged = merge_transaction_identity(transaction, identity, config["id_column"])
    sampled = stratified_sample(
        merged,
        target_column=config["target_column"],
        sample_size=int(settings["sample_size"]),
        random_state=int(config["random_seed"]),
    )
    sampled, missing_indicators = add_missing_features(
        sampled,
        missing_candidates=list(config.get("missing_indicator_candidates", [])),
        exclude_columns=[config["id_column"], config["target_column"], config["time_column"]],
    )
    created_missing_sources = _candidate_missing_sources(missing_indicators)
    skipped_missing_indicators = [
        column for column in config.get("missing_indicator_candidates", [])
        if column not in created_missing_sources
    ]
    train, valid, test = stratified_train_valid_test_split(
        sampled,
        target_column=config["target_column"],
        id_column=config["id_column"],
        train_ratio=float(config["train_ratio"]),
        valid_ratio=float(config["valid_ratio"]),
        test_ratio=float(config["test_ratio"]),
        random_state=int(config["random_seed"]),
    )

    processed_dir = Path(settings["processed_dir"])
    output_paths = save_processed_splits(train, valid, test, processed_dir)

    basic_group = build_transaction_basic_group(list(sampled.columns), config)
    identity_group = build_transaction_identity_group(basic_group, list(sampled.columns), config)
    missing_group = build_transaction_identity_missing_group(identity_group, missing_indicators)
    feature_groups = {
        "transaction_basic": basic_group,
        "transaction_identity": identity_group,
        "transaction_identity_missing": missing_group,
    }
    validate_feature_groups(feature_groups)

    reports_dir = _project_path(config, "reports/tables")
    feature_groups_path = save_feature_groups(
        feature_groups,
        reports_dir / "feature_groups.json",
        stage=str(settings["stage_name"]),
    )

    splits = {"train": train, "valid": valid, "test": test}
    summary = build_data_summary(splits, config["target_column"])
    summary_path = reports_dir / str(settings["summary_file"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    output_paths = {**output_paths, "feature_groups": feature_groups_path, "data_summary": summary_path}

    transaction_feature_columns = [
        column for column in transaction.columns
        if column not in {config["id_column"], config["target_column"], config["time_column"]}
    ]
    identity_feature_columns = [column for column in identity.columns if column != config["id_column"]]
    max_allowed_difference = float(config.get("maximum_allowed_fraud_rate_difference", 0.002))
    fraud_rate_ok = fraud_rate_check_passed(
        float(merged[config["target_column"]].mean()) if len(merged) else 0.0,
        float(sampled[config["target_column"]].mean()) if len(sampled) else 0.0,
        max_allowed_difference,
    )

    if not fraud_rate_ok:
        LOGGER.warning(
            "Fraud-rate check failed: original=%.6f sampled=%.6f max_difference=%.6f",
            float(merged[config["target_column"]].mean()),
            float(sampled[config["target_column"]].mean()),
            max_allowed_difference,
        )

    if stage == "formal":
        feature_profile = build_feature_profile(
            sampled,
            feature_groups=feature_groups,
            transaction_columns=transaction_feature_columns,
            identity_columns=identity_feature_columns,
            missing_indicator_columns=missing_indicators,
        )
        feature_profile_path = reports_dir / str(settings["feature_profile_file"])
        feature_profile.to_csv(feature_profile_path, index=False)
        data_quality = build_data_quality_report(
            transaction=transaction,
            identity=identity,
            merged=merged,
            sampled=sampled,
            splits=splits,
            feature_groups=feature_groups,
            requested_sample_size=int(settings["sample_size"]),
            missing_candidate_columns_not_found=skipped_missing_indicators,
            id_column=config["id_column"],
            target_column=config["target_column"],
        )
        data_quality["fraud_rate_check_passed"] = fraud_rate_ok
        data_quality_path = _save_json(data_quality, reports_dir / str(settings["data_quality_file"]))
        output_paths["feature_profile"] = feature_profile_path
        output_paths["data_quality"] = data_quality_path

    metadata = build_metadata(
        config=config,
        stage_name=str(settings["stage_name"]),
        transaction=transaction,
        identity=identity,
        merged=merged,
        sampled=sampled,
        splits=splits,
        feature_groups=feature_groups,
        missing_transaction_columns=missing_transaction,
        missing_identity_columns=missing_identity,
        missing_indicator_columns=missing_indicators,
        skipped_missing_indicators=skipped_missing_indicators,
        requested_sample_size=int(settings["sample_size"]),
        fraud_rate_check=fraud_rate_ok,
        output_paths=output_paths,
    )
    metadata["identity_match_rate"] = identity_match_rate
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["metadata"] = metadata_path

    _log_run_summary(transaction, identity, merged, sampled, splits, identity_match_rate, output_paths)
    return {"metadata": metadata, "summary": summary, "outputs": output_paths, "feature_groups": feature_groups}


def _log_run_summary(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    merged: pd.DataFrame,
    sampled: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    identity_match_rate: float,
    output_paths: dict[str, Path],
) -> None:
    target = "isFraud"
    LOGGER.info("transaction rows: %s", len(transaction))
    LOGGER.info("identity rows: %s", len(identity))
    LOGGER.info("merged rows: %s", len(merged))
    LOGGER.info("identity match rate: %.6f", identity_match_rate)
    LOGGER.info("original fraud ratio: %.6f", merged[target].mean())
    LOGGER.info("sampled fraud ratio: %.6f", sampled[target].mean())
    for split_name, split in splits.items():
        LOGGER.info("%s rows: %s fraud_ratio: %.6f", split_name, len(split), split[target].mean())
    for name, path in output_paths.items():
        LOGGER.info("output %s: %s", name, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build IEEE-CIS processed data splits.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True, help="Pipeline stage to run.")
    return parser


def main() -> int:
    """Run the data pipeline CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    try:
        config = load_yaml_config(args.config)
        run_pipeline(config, stage=args.stage)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("Data pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
