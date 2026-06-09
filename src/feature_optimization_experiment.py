"""Feature engineering optimization experiment for IEEE-CIS fraud detection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluate import evaluate_model
from src.models import build_training_pipeline, resolve_tree_model
from src.preprocessing import build_tree_preprocessor, load_feature_groups
from src.train import measure_training_time


LOGGER = logging.getLogger(__name__)

RANDOM_SEED = 42
TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"
BASELINE_FEATURE_GROUP = "transaction_basic"
UID_COLUMNS: tuple[str, ...] = (
    "uid_card1_addr1_D1n",
    "uid_card1_D1n",
    "uid_card1_addr1",
)
BLOCKED_MODEL_COLUMNS: frozenset[str] = frozenset(
    {ID_COLUMN, TARGET_COLUMN, TIME_COLUMN, *UID_COLUMNS}
)
DOMAIN_TIME_COLUMNS: tuple[str, ...] = (
    "TransactionDay",
    "TransactionWeek",
    "TransactionMonthApprox",
    "D1n",
    "D2n",
    "D3n",
    "D4n",
    "D10n",
    "D15n",
)
DOMAIN_AMOUNT_COLUMNS: tuple[str, ...] = (
    "TransactionAmt_log1p",
    "TransactionAmt_cents",
)
MISSING_PATTERN_COLUMNS: tuple[str, ...] = (
    "identity_present",
    "identity_missing_count",
    "basic_missing_count",
)
AGGREGATION_KEYS: tuple[str, ...] = UID_COLUMNS
AGGREGATION_STATS: tuple[str, ...] = ("count", "mean", "std", "nunique")
AGGREGATION_PRIORITY_COLUMNS: tuple[str, ...] = (
    "TransactionAmt",
    "C13",
    "D1",
    "D10",
    "D15",
    "dist1",
    "TransactionAmt_log1p",
    "dist2",
    "C1",
    "C2",
    "C5",
    "C6",
    "C9",
    "C11",
    "D2",
    "D3",
    "D4",
)
MAX_AGGREGATION_FEATURES = 60


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run feature optimization experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/stage2_formal"))
    parser.add_argument("--feature-groups", type=Path, default=Path("reports/tables/feature_groups.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--run-name", default="optimization")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML model configuration."""

    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def read_split_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read existing random split parquet files without modifying them."""

    paths = {
        "train": data_dir / "train.parquet",
        "valid": data_dir / "valid.parquet",
        "test": data_dir / "test.parquet",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing processed split file(s): " + "; ".join(missing))
    return pd.read_parquet(paths["train"]), pd.read_parquet(paths["valid"]), pd.read_parquet(paths["test"])


def validate_input_frame(dataframe: pd.DataFrame) -> None:
    """Validate required input columns and basic label constraints."""

    required = [ID_COLUMN, TARGET_COLUMN, TIME_COLUMN, "card1", "addr1", "D1"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}.")
    if dataframe[ID_COLUMN].duplicated().any():
        raise ValueError("TransactionID must be unique across the full sample.")
    labels = set(dataframe[TARGET_COLUMN].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError("isFraud must contain only 0/1 labels.")
    if dataframe[TIME_COLUMN].isna().any():
        raise ValueError("TransactionDT must not contain missing values.")


def add_domain_features(dataframe: pd.DataFrame, basic_features: list[str] | None = None) -> pd.DataFrame:
    """Add lightweight time, amount, missing-pattern, and temporary uid features."""

    output = dataframe.copy()
    if TIME_COLUMN in output.columns:
        output["TransactionDay"] = np.floor(output[TIME_COLUMN] / 86400).astype("int64")
        output["TransactionWeek"] = np.floor(output["TransactionDay"] / 7).astype("int64")
        output["TransactionMonthApprox"] = np.floor(output["TransactionDay"] / 30).astype("int64")

    if "TransactionDay" in output.columns:
        for column in ("D1", "D2", "D3", "D4", "D10", "D15"):
            if column in output.columns:
                output[f"{column}n"] = output["TransactionDay"] - output[column]

    if "TransactionAmt" in output.columns:
        amount = pd.to_numeric(output["TransactionAmt"], errors="coerce")
        output["TransactionAmt_log1p"] = np.log1p(amount)
        output["TransactionAmt_cents"] = amount - np.floor(amount)

    identity_columns = [
        column
        for column in output.columns
        if column.startswith("id_") or column in {"DeviceType", "DeviceInfo"}
    ]
    if identity_columns:
        output["identity_present"] = output[identity_columns].notna().any(axis=1).astype(int)
        output["identity_missing_count"] = output[identity_columns].isna().sum(axis=1)
    else:
        output["identity_present"] = 0
        output["identity_missing_count"] = 0

    if basic_features is not None:
        basic_columns = [column for column in basic_features if column in output.columns]
    else:
        basic_columns = [
            column
            for column in output.columns
            if column not in {TARGET_COLUMN, ID_COLUMN, TIME_COLUMN}
            and not column.startswith("id_")
            and column not in {"DeviceType", "DeviceInfo"}
        ]
    output["basic_missing_count"] = output[basic_columns].isna().sum(axis=1) if basic_columns else 0

    if {"card1", "addr1", "D1n"}.issubset(output.columns):
        output["uid_card1_addr1_D1n"] = (
            output["card1"].astype("string")
            + "_"
            + output["addr1"].astype("string")
            + "_"
            + output["D1n"].round(0).astype("Int64").astype("string")
        )
    if {"card1", "D1n"}.issubset(output.columns):
        output["uid_card1_D1n"] = (
            output["card1"].astype("string")
            + "_"
            + output["D1n"].round(0).astype("Int64").astype("string")
        )
    if {"card1", "addr1"}.issubset(output.columns):
        output["uid_card1_addr1"] = output["card1"].astype("string") + "_" + output["addr1"].astype("string")

    return output


def _select_aggregation_plan(train_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Select a bounded aggregation feature plan."""

    plan: list[tuple[str, str, str]] = []
    keys = [key for key in AGGREGATION_KEYS if key in train_df.columns]
    value_columns = [column for column in AGGREGATION_PRIORITY_COLUMNS if column in train_df.columns]
    for key in keys:
        for value_column in value_columns:
            for stat in AGGREGATION_STATS:
                if len(plan) >= MAX_AGGREGATION_FEATURES:
                    return plan
                plan.append((key, value_column, stat))
    return plan


def fit_group_aggregations(train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit group aggregation mappings on the training split only."""

    plan = _select_aggregation_plan(train_df)
    fitted: dict[str, Any] = {"plan": plan, "tables": {}}
    for key, value_column, stat in plan:
        feature_name = f"agg_{key}_{value_column}_{stat}"
        series = pd.to_numeric(train_df[value_column], errors="coerce")
        grouped = train_df.assign(_aggregation_value=series).groupby(key, dropna=False)["_aggregation_value"]
        if stat == "count":
            mapping = grouped.count()
        elif stat == "mean":
            mapping = grouped.mean()
        elif stat == "std":
            mapping = grouped.std()
        elif stat == "nunique":
            mapping = grouped.nunique()
        else:
            raise ValueError(f"Unsupported aggregation statistic: {stat}")
        fitted["tables"][feature_name] = {"key": key, "mapping": mapping}
    return fitted


def transform_group_aggregations(df: pd.DataFrame, fitted_stats: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted group aggregation mappings to any split."""

    output = df.copy()
    for feature_name, payload in fitted_stats.get("tables", {}).items():
        key = payload["key"]
        mapping = payload["mapping"]
        output[feature_name] = output[key].map(mapping) if key in output.columns else np.nan
    return output


def build_optimized_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    basic_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add domain features and train-only group aggregation features to each split."""

    train_features = add_domain_features(train_df, basic_features=basic_features)
    valid_features = add_domain_features(valid_df, basic_features=basic_features)
    test_features = add_domain_features(test_df, basic_features=basic_features)
    fitted_stats = fit_group_aggregations(train_features)
    return (
        transform_group_aggregations(train_features, fitted_stats),
        transform_group_aggregations(valid_features, fitted_stats),
        transform_group_aggregations(test_features, fitted_stats),
        fitted_stats,
    )


def get_final_feature_list(
    dataframe: pd.DataFrame,
    baseline_features: list[str],
    fitted_stats: dict[str, Any],
) -> list[str]:
    """Build final optimized feature list while excluding temporary uid keys."""

    generated_columns = [
        *DOMAIN_TIME_COLUMNS,
        *DOMAIN_AMOUNT_COLUMNS,
        *MISSING_PATTERN_COLUMNS,
        *fitted_stats.get("tables", {}).keys(),
    ]
    features: list[str] = []
    for column in [*baseline_features, *generated_columns]:
        if column in dataframe.columns and column not in BLOCKED_MODEL_COLUMNS and column not in features:
            features.append(column)
    return features


def infer_column_types(dataframe: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns for tree preprocessing."""

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(dataframe[feature]):
            numeric_columns.append(feature)
        else:
            categorical_columns.append(feature)
    return numeric_columns, categorical_columns


def split_by_time(full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by TransactionDT and split 60/20/20 in memory."""

    validate_input_frame(full_df)
    sorted_df = full_df.sort_values([TIME_COLUMN, ID_COLUMN]).reset_index(drop=True)
    train_end = int(len(sorted_df) * 0.6)
    valid_end = int(len(sorted_df) * 0.8)
    train_df = sorted_df.iloc[:train_end].copy()
    valid_df = sorted_df.iloc[train_end:valid_end].copy()
    test_df = sorted_df.iloc[valid_end:].copy()
    if train_df[TIME_COLUMN].max() > valid_df[TIME_COLUMN].min():
        raise ValueError("Train and validation time ranges overlap.")
    if valid_df[TIME_COLUMN].max() > test_df[TIME_COLUMN].min():
        raise ValueError("Validation and test time ranges overlap.")
    return train_df, valid_df, test_df


def build_feature_summary(
    baseline_features: list[str],
    optimized_features: list[str],
    fitted_stats: dict[str, Any],
) -> pd.DataFrame:
    """Build the optimization feature inventory table."""

    rows: list[dict[str, Any]] = []
    for column in DOMAIN_TIME_COLUMNS:
        rows.append(
            {
                "feature_name": column,
                "feature_type": "domain_time",
                "source_columns": "TransactionDT/D columns",
                "generated": column in optimized_features,
                "reason": "Transaction time and D-relative date signal",
            }
        )
    for column in DOMAIN_AMOUNT_COLUMNS:
        rows.append(
            {
                "feature_name": column,
                "feature_type": "domain_amount",
                "source_columns": "TransactionAmt",
                "generated": column in optimized_features,
                "reason": "Amount scale and cents pattern",
            }
        )
    for column in MISSING_PATTERN_COLUMNS:
        rows.append(
            {
                "feature_name": column,
                "feature_type": "missing_pattern",
                "source_columns": "identity/basic feature columns",
                "generated": column in optimized_features,
                "reason": "Systematic missingness signal",
            }
        )
    for feature_name, payload in fitted_stats.get("tables", {}).items():
        source = feature_name.removeprefix("agg_")
        rows.append(
            {
                "feature_name": feature_name,
                "feature_type": "group_aggregation",
                "source_columns": f"{payload['key']} + {source}",
                "generated": feature_name in optimized_features,
                "reason": "Train-only uid group aggregation",
            }
        )

    baseline_set = set(baseline_features)
    rows.append(
        {
            "feature_name": "__baseline_transaction_basic__",
            "feature_type": "baseline",
            "source_columns": f"{len(baseline_set)} transaction_basic columns",
            "generated": True,
            "reason": "Current best Stage2 feature group",
        }
    )
    return pd.DataFrame(rows)


def train_and_evaluate(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    feature_set: str,
    split_strategy: str,
) -> tuple[dict[str, Any], Any]:
    """Fit one tree model and evaluate it on one validation split."""

    numeric_columns, categorical_columns = infer_column_types(train_df, features)
    preprocessor = build_tree_preprocessor(
        numeric_columns,
        categorical_columns,
        config.get("preprocessing", {}),
    )
    model_name, classifier, fallback_reason = resolve_tree_model(config, random_seed=RANDOM_SEED)
    pipeline = build_training_pipeline(preprocessor, classifier)
    trained_pipeline, training_time = measure_training_time(
        pipeline,
        train_df.loc[:, features],
        train_df[TARGET_COLUMN],
    )
    record, _ = evaluate_model(
        trained_pipeline,
        eval_df.loc[:, features],
        eval_df[TARGET_COLUMN],
        threshold=0.5,
        model_name=model_name,
        feature_group=feature_set,
        split_name=f"{split_strategy}_valid",
        training_time=training_time,
    )
    return {
        "split_strategy": split_strategy,
        "feature_set": feature_set,
        "model": model_name,
        "fallback_reason": fallback_reason,
        "n_features": len(features),
        "pr_auc": record["pr_auc"],
        "roc_auc": record["roc_auc"],
        "precision": record["precision"],
        "recall": record["recall"],
        "f1": record["f1"],
        "mcc": record["mcc"],
        "accuracy": record["accuracy"],
        "tn": record["tn"],
        "fp": record["fp"],
        "fn": record["fn"],
        "tp": record["tp"],
    }, trained_pipeline


def add_time_range_columns(
    record: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Attach time range metadata to a result record."""

    return {
        **record,
        "train_time_min": int(train_df[TIME_COLUMN].min()),
        "train_time_max": int(train_df[TIME_COLUMN].max()),
        "valid_time_min": int(valid_df[TIME_COLUMN].min()),
        "valid_time_max": int(valid_df[TIME_COLUMN].max()),
        "test_time_min": int(test_df[TIME_COLUMN].min()),
        "test_time_max": int(test_df[TIME_COLUMN].max()),
    }


def plot_pr_auc_comparison(random_df: pd.DataFrame, time_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot PR-AUC comparison for baseline and optimized feature sets."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = pd.concat([random_df, time_df], ignore_index=True)
    labels = [f"{row.split_strategy}\n{row.feature_set}" for row in plot_df.itertuples()]
    _, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, plot_df["pr_auc"].astype(float))
    axis.set_ylabel("PR-AUC")
    axis.set_title("Feature optimization PR-AUC comparison")
    axis.tick_params(axis="x", labelrotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_feature_importance_top20(pipeline: Any, output_path: Path) -> Path | None:
    """Plot top 20 feature importances when the fitted classifier exposes them."""

    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if not hasattr(classifier, "feature_importances_"):
        LOGGER.warning("Classifier does not expose feature_importances_; skipping importance plot.")
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{index}" for index in range(len(classifier.feature_importances_))])

    importances = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(20)
    if importances.empty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(8, 6))
    axis.barh(importances["feature_name"][::-1], importances["importance"][::-1])
    axis.set_xlabel("Importance")
    axis.set_title("Top 20 optimized feature importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def run_feature_optimization_experiment(
    data_dir: Path,
    feature_groups_path: Path,
    config_path: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """Run random-split and time-split feature optimization comparisons."""

    config = load_config(config_path)
    feature_groups = load_feature_groups(feature_groups_path)
    if BASELINE_FEATURE_GROUP not in feature_groups:
        raise ValueError(f"Feature group not found: {BASELINE_FEATURE_GROUP}")

    train_df, valid_df, test_df = read_split_data(data_dir)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    validate_input_frame(full_df)
    baseline_features = [
        column
        for column in feature_groups[BASELINE_FEATURE_GROUP]
        if column in train_df.columns and column not in BLOCKED_MODEL_COLUMNS
    ]
    if not baseline_features:
        raise ValueError("No baseline transaction_basic features are available.")

    random_opt_train, random_opt_valid, _, random_stats = build_optimized_splits(
        train_df,
        valid_df,
        test_df,
        baseline_features,
    )
    optimized_features = get_final_feature_list(random_opt_train, baseline_features, random_stats)

    random_records: list[dict[str, Any]] = []
    random_baseline_record, _ = train_and_evaluate(
        train_df,
        valid_df,
        baseline_features,
        config,
        "baseline_basic",
        "random",
    )
    random_optimized_record, random_optimized_pipeline = train_and_evaluate(
        random_opt_train,
        random_opt_valid,
        optimized_features,
        config,
        "optimized_domain_uid",
        "random",
    )
    random_records.extend([random_baseline_record, random_optimized_record])

    time_train, time_valid, time_test = split_by_time(full_df)
    time_opt_train, time_opt_valid, _, time_stats = build_optimized_splits(
        time_train,
        time_valid,
        time_test,
        baseline_features,
    )
    time_optimized_features = get_final_feature_list(time_opt_train, baseline_features, time_stats)
    time_baseline_record, _ = train_and_evaluate(
        time_train,
        time_valid,
        baseline_features,
        config,
        "baseline_basic",
        "time",
    )
    time_optimized_record, _ = train_and_evaluate(
        time_opt_train,
        time_opt_valid,
        time_optimized_features,
        config,
        "optimized_domain_uid",
        "time",
    )
    time_records = [
        add_time_range_columns(time_baseline_record, time_train, time_valid, time_test),
        add_time_range_columns(time_optimized_record, time_train, time_valid, time_test),
    ]

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_summary = build_feature_summary(baseline_features, optimized_features, random_stats)
    feature_summary_path = tables_dir / f"{run_name}_feature_summary.csv"
    random_path = tables_dir / f"{run_name}_random_valid_comparison.csv"
    time_path = tables_dir / f"{run_name}_time_split_comparison.csv"
    selected_path = tables_dir / f"{run_name}_selected_features.json"
    pr_auc_figure_path = figures_dir / f"{run_name}_pr_auc_comparison.png"
    importance_path = figures_dir / f"{run_name}_feature_importance_top20.png"

    random_df = pd.DataFrame(random_records)
    time_df = pd.DataFrame(time_records)
    feature_summary.to_csv(feature_summary_path, index=False)
    random_df.to_csv(random_path, index=False)
    time_df.to_csv(time_path, index=False)
    selected_payload = {
        "experiment": "feature_optimization",
        "baseline_feature_group": BASELINE_FEATURE_GROUP,
        "optimized_feature_set": "optimized_domain_uid",
        "model_preference": "lightgbm",
        "random_seed": RANDOM_SEED,
        "baseline_feature_count": len(baseline_features),
        "optimized_feature_count": len(optimized_features),
        "generated_feature_count": len(optimized_features) - len(baseline_features),
        "uid_columns_excluded_from_model": list(UID_COLUMNS),
        "aggregation_feature_count": len(random_stats.get("tables", {})),
    }
    selected_path.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    plot_pr_auc_comparison(random_df, time_df, pr_auc_figure_path)
    importance_written = plot_feature_importance_top20(random_optimized_pipeline, importance_path)

    baseline_random = random_df.loc[random_df["feature_set"] == "baseline_basic", "pr_auc"].iloc[0]
    optimized_random = random_df.loc[random_df["feature_set"] == "optimized_domain_uid", "pr_auc"].iloc[0]
    baseline_time = time_df.loc[time_df["feature_set"] == "baseline_basic", "pr_auc"].iloc[0]
    optimized_time = time_df.loc[time_df["feature_set"] == "optimized_domain_uid", "pr_auc"].iloc[0]
    return {
        "feature_summary_path": feature_summary_path,
        "random_path": random_path,
        "time_path": time_path,
        "selected_path": selected_path,
        "pr_auc_figure_path": pr_auc_figure_path,
        "importance_path": importance_written,
        "baseline_random_pr_auc": float(baseline_random),
        "optimized_random_pr_auc": float(optimized_random),
        "baseline_time_pr_auc": float(baseline_time),
        "optimized_time_pr_auc": float(optimized_time),
        "generated_feature_count": selected_payload["generated_feature_count"],
        "unknown_uid_supported": True,
    }


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    result = run_feature_optimization_experiment(
        data_dir=args.data_dir,
        feature_groups_path=args.feature_groups,
        config_path=args.config,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    random_delta = result["optimized_random_pr_auc"] - result["baseline_random_pr_auc"]
    time_delta = result["optimized_time_pr_auc"] - result["baseline_time_pr_auc"]
    print("Feature optimization experiment complete")
    print(f"Baseline random PR-AUC: {result['baseline_random_pr_auc']:.6f}")
    print(f"Optimized random PR-AUC: {result['optimized_random_pr_auc']:.6f}")
    print(f"Baseline time PR-AUC: {result['baseline_time_pr_auc']:.6f}")
    print(f"Optimized time PR-AUC: {result['optimized_time_pr_auc']:.6f}")
    print(f"Random PR-AUC delta: {random_delta:.6f}")
    print(f"Time PR-AUC delta: {time_delta:.6f}")
    print(f"Generated feature count: {result['generated_feature_count']}")
    print(f"Unknown uid handled with missing aggregate values: {result['unknown_uid_supported']}")


if __name__ == "__main__":
    main()
