"""Time-split extension experiment for IEEE-CIS fraud detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import plot_confusion_matrix
from src.metrics import get_probability_scores
from src.models import build_model, build_training_pipeline
from src.preprocessing import (
    build_preprocessor,
    build_tree_preprocessor,
    load_feature_groups,
    load_metadata,
    split_features_target,
    validate_requested_features,
)
from src.threshold_analysis import (
    apply_fixed_threshold,
    assign_error_type,
    evaluate_threshold,
    generate_threshold_grid,
    search_optimal_threshold,
)
from src.train import measure_training_time


TREE_MODELS = {"lightgbm", "hist_gradient_boosting", "random_forest"}
PREDICTION_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "prediction_score",
    "prediction_at_0_5",
    "prediction_at_selected_threshold",
    "error_type_at_0_5",
    "error_type_at_selected_threshold",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the time-split experiment."""

    parser = argparse.ArgumentParser(description="Run the time-split extension experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/stage2_formal"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/stage2_formal/metadata.json"))
    parser.add_argument("--feature-groups", type=Path, default=Path("reports/tables/feature_groups.json"))
    parser.add_argument(
        "--random-threshold-results",
        type=Path,
        default=Path("reports/tables/stage3_threshold_comparison_test.csv"),
    )
    parser.add_argument(
        "--selected-threshold",
        type=Path,
        default=Path("reports/tables/stage3_selected_threshold.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--run-name", default="time_split")
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML model configuration."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def load_frozen_selection(selection_path: Path) -> dict[str, Any]:
    """Load frozen model and feature-group choice from Stage 3 selection JSON."""

    if not selection_path.exists():
        return {
            "selected_model": "lightgbm",
            "selected_feature_group": "transaction_basic",
            "selection_source": "hard-coded fallback: frozen from Stage 2/3 result",
        }
    with selection_path.open("r", encoding="utf-8") as file:
        selection = json.load(file)
    return {
        "selected_model": selection.get("selected_model", "lightgbm"),
        "selected_feature_group": selection.get("selected_feature_group", "transaction_basic"),
        "selection_source": str(selection_path),
    }


def read_stage2_formal_sample(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read existing Stage 2 formal train, valid, and test parquet files."""

    splits = {}
    for split in ("train", "valid", "test"):
        split_path = data_dir / f"{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Required Stage 2 {split} parquet not found: {split_path}")
        splits[split] = pd.read_parquet(split_path)
    return splits


def build_time_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Concatenate Stage 2 splits, sort by TransactionDT, and split 60/20/20."""

    full_df = pd.concat([splits["train"], splits["valid"], splits["test"]], ignore_index=True)
    validate_full_sample(full_df, sum(len(split) for split in splits.values()))
    sorted_df = full_df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    n_rows = len(sorted_df)
    train_end = int(n_rows * 0.6)
    valid_end = int(n_rows * 0.8)
    time_splits = {
        "train": sorted_df.iloc[:train_end].copy(),
        "valid": sorted_df.iloc[train_end:valid_end].copy(),
        "test": sorted_df.iloc[valid_end:].copy(),
    }
    validate_time_order(time_splits)
    return time_splits


def validate_full_sample(full_df: pd.DataFrame, expected_rows: int) -> None:
    """Validate the concatenated formal sample before time splitting."""

    missing = [column for column in ("TransactionID", "TransactionDT", "isFraud") if column not in full_df.columns]
    if missing:
        raise ValueError(f"Full sample missing required column(s): {', '.join(missing)}.")
    if full_df["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique before time splitting.")
    if full_df["TransactionDT"].isna().any():
        raise ValueError("TransactionDT must not contain missing values.")
    if len(full_df) != expected_rows:
        raise ValueError(f"Full sample row count mismatch: expected {expected_rows}, got {len(full_df)}.")


def validate_time_order(time_splits: dict[str, pd.DataFrame]) -> None:
    """Ensure time ranges do not cross between train, valid, and test."""

    if time_splits["train"]["TransactionDT"].max() > time_splits["valid"]["TransactionDT"].min():
        raise ValueError("Train TransactionDT range overlaps validation range.")
    if time_splits["valid"]["TransactionDT"].max() > time_splits["test"]["TransactionDT"].min():
        raise ValueError("Validation TransactionDT range overlaps test range.")


def build_time_split_summary(time_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize time-split row counts, fraud rates, and time ranges."""

    rows = []
    for split, dataframe in time_splits.items():
        positive_count = int(dataframe["isFraud"].sum())
        rows.append(
            {
                "split": split,
                "n_rows": len(dataframe),
                "positive_count": positive_count,
                "positive_rate": positive_count / len(dataframe),
                "transaction_dt_min": dataframe["TransactionDT"].min(),
                "transaction_dt_max": dataframe["TransactionDT"].max(),
            }
        )
    return pd.DataFrame(rows)


def build_pipeline_for_model(
    config: dict[str, Any],
    metadata: dict[str, Any],
    features: list[str],
    model_name: str,
) -> object:
    """Build an unfitted pipeline for the frozen model and feature group."""

    numeric = [column for column in metadata["numeric_columns"] if column in features]
    categorical = [column for column in metadata["categorical_columns"] if column in features]
    params = dict(config["models"].get(model_name, {}) or {})
    params.pop("enabled", None)
    classifier = build_model(model_name, int(config.get("random_seed", 42)), params)
    preprocessor = (
        build_tree_preprocessor(numeric, categorical, config["preprocessing"])
        if model_name in TREE_MODELS
        else build_preprocessor(numeric, categorical, config["preprocessing"])
    )
    return build_training_pipeline(preprocessor, classifier)


def fit_and_score(
    pipeline: object,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[object, float, np.ndarray]:
    """Fit on one split and return scores for another split."""

    X_train, y_train = split_features_target(train_df, features, target)
    trained_pipeline, training_time = measure_training_time(pipeline, X_train, y_train)
    X_score, _ = split_features_target(score_df, features, target)
    return trained_pipeline, training_time, get_probability_scores(trained_pipeline, X_score)


def build_threshold_comparison(
    y_true: Any,
    y_score: Any,
    default_threshold: float,
    selected_threshold: float,
    model_name: str,
    feature_group: str,
    fn_cost: int,
    fp_cost: int,
) -> pd.DataFrame:
    """Build default-vs-cost-sensitive test comparison from one score vector."""

    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    rows = []
    for threshold_type, threshold in (("default_0_5", default_threshold), ("cost_sensitive", selected_threshold)):
        metrics = evaluate_threshold(y_true, y_score, threshold, fn_cost, fp_cost)
        rows.append(
            {
                "threshold_type": threshold_type,
                "threshold": threshold,
                "model": model_name,
                "feature_group": feature_group,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                **{key: value for key, value in metrics.items() if key != "threshold"},
            }
        )
    return pd.DataFrame(rows)


def save_prediction_file(
    dataframe: pd.DataFrame,
    y_score: Any,
    selected_threshold: float,
    output_path: Path,
) -> Path:
    """Save compact time-split predictions to parquet."""

    pred_default = apply_fixed_threshold(y_score, 0.5)
    pred_selected = apply_fixed_threshold(y_score, selected_threshold)
    predictions = dataframe.loc[:, [column for column in PREDICTION_COLUMNS[:3] if column in dataframe.columns]].copy()
    predictions["prediction_score"] = np.asarray(y_score)
    predictions["prediction_at_0_5"] = pred_default
    predictions["prediction_at_selected_threshold"] = pred_selected
    predictions["error_type_at_0_5"] = assign_error_type(dataframe["isFraud"], pred_default)
    predictions["error_type_at_selected_threshold"] = assign_error_type(dataframe["isFraud"], pred_selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    return output_path


def build_vs_random_comparison(random_results: pd.DataFrame, time_results: pd.DataFrame) -> pd.DataFrame:
    """Combine random Stage 3 and time-split test threshold results."""

    columns = [
        "split_strategy",
        "threshold_type",
        "threshold",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "cost",
        "fp",
        "fn",
        "tp",
        "tn",
        "n_samples",
        "positive_support",
    ]
    random_subset = random_results.assign(split_strategy="random_stage3")
    time_subset = time_results.assign(split_strategy="time_split")
    combined = pd.concat([random_subset, time_subset], ignore_index=True)
    return combined.loc[:, columns]


def plot_threshold_metric(search_df: pd.DataFrame, metric: str, selected_threshold: float, output_path: Path) -> Path:
    """Plot a validation threshold metric."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(8, 4))
    axis.plot(search_df["threshold"], search_df[metric])
    axis.axvline(selected_threshold, color="red", linestyle="--", label="selected threshold")
    axis.set_xlabel("threshold")
    axis.set_ylabel(metric)
    axis.set_title(f"Time Split Validation {metric}")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_vs_random(comparison: pd.DataFrame, metric: str, output_path: Path) -> Path:
    """Plot random vs time split comparison for one metric."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = comparison["split_strategy"] + " / " + comparison["threshold_type"]
    _, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, comparison[metric])
    axis.set_ylabel(metric)
    axis.set_title(f"Random vs Time Split {metric}")
    axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def run_time_split_experiment(
    config_path: Path,
    data_dir: Path,
    metadata_path: Path,
    feature_groups_path: Path,
    random_threshold_results_path: Path,
    selected_threshold_path: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """Run the full time-split extension experiment."""

    config = load_config(config_path)
    metadata = load_metadata(metadata_path)
    feature_groups = load_feature_groups(feature_groups_path)
    splits = read_stage2_formal_sample(data_dir)
    time_splits = build_time_splits(splits)
    summary = build_time_split_summary(time_splits)

    frozen = load_frozen_selection(selected_threshold_path)
    model_name = str(frozen["selected_model"])
    feature_group = str(frozen["selected_feature_group"])
    target = metadata.get("target_column", "isFraud")
    id_column = metadata.get("id_column", "TransactionID")
    time_column = metadata.get("time_column", "TransactionDT")
    features = validate_requested_features(time_splits["train"], feature_groups, feature_group, target, id_column, time_column)
    validate_requested_features(time_splits["valid"], feature_groups, feature_group, target, id_column, time_column)
    validate_requested_features(time_splits["test"], feature_groups, feature_group, target, id_column, time_column)

    threshold_config = config.get("threshold_optimization", {})
    thresholds = generate_threshold_grid(
        threshold_config.get("minimum_threshold", 0.05),
        threshold_config.get("maximum_threshold", 0.95),
        threshold_config.get("threshold_step", 0.01),
    )
    fn_cost = int(threshold_config.get("false_negative_cost", 10))
    fp_cost = int(threshold_config.get("false_positive_cost", 1))

    valid_pipeline = build_pipeline_for_model(config, metadata, features, model_name)
    _, _, valid_score = fit_and_score(valid_pipeline, time_splits["train"], time_splits["valid"], features, target)
    _, y_valid = split_features_target(time_splits["valid"], features, target)
    selected_threshold, search_df = search_optimal_threshold(y_valid, valid_score, thresholds, fn_cost, fp_cost)

    final_pipeline = build_pipeline_for_model(config, metadata, features, model_name)
    train_valid = pd.concat([time_splits["train"], time_splits["valid"]], ignore_index=True)
    _, _, test_score = fit_and_score(final_pipeline, train_valid, time_splits["test"], features, target)
    _, y_test = split_features_target(time_splits["test"], features, target)
    test_comparison = build_threshold_comparison(
        y_test,
        test_score,
        float(config.get("default_threshold", 0.5)),
        selected_threshold,
        model_name,
        feature_group,
        fn_cost,
        fp_cost,
    )
    random_results = pd.read_csv(random_threshold_results_path)
    vs_random = build_vs_random_comparison(random_results, test_comparison)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "time_split_data_summary.csv", index=False)
    search_df.to_csv(tables_dir / "time_split_threshold_search_valid.csv", index=False)
    selected_row = search_df.loc[search_df["threshold"] == selected_threshold].iloc[0]
    selected_summary = {
        "experiment": "time_split_extension",
        "selection_split": "validation",
        "selected_model": model_name,
        "selected_feature_group": feature_group,
        "selection_source": frozen["selection_source"],
        "selected_threshold": selected_threshold,
        "cost_formula": f"{fn_cost} * FN + {fp_cost} * FP",
        "validation_cost": int(selected_row["cost"]),
        "validation_precision": float(selected_row["precision"]),
        "validation_recall": float(selected_row["recall"]),
        "validation_f1": float(selected_row["f1"]),
        "validation_tn": int(selected_row["tn"]),
        "validation_fp": int(selected_row["fp"]),
        "validation_fn": int(selected_row["fn"]),
        "validation_tp": int(selected_row["tp"]),
    }
    (tables_dir / "time_split_selected_threshold.json").write_text(
        json.dumps(selected_summary, indent=2), encoding="utf-8"
    )
    test_comparison.to_csv(tables_dir / "time_split_threshold_comparison_test.csv", index=False)
    vs_random.to_csv(tables_dir / "time_split_vs_random_comparison.csv", index=False)

    save_prediction_file(
        time_splits["valid"], valid_score, selected_threshold, predictions_dir / "time_split_validation_predictions.parquet"
    )
    save_prediction_file(
        time_splits["test"], test_score, selected_threshold, predictions_dir / "time_split_test_predictions.parquet"
    )

    plot_threshold_metric(search_df, "cost", selected_threshold, figures_dir / "time_split_threshold_cost_valid.png")
    plot_threshold_metric(search_df, "precision", selected_threshold, figures_dir / "time_split_threshold_precision_valid.png")
    plot_threshold_metric(search_df, "recall", selected_threshold, figures_dir / "time_split_threshold_recall_valid.png")
    default_row = test_comparison.loc[test_comparison["threshold_type"] == "default_0_5"].iloc[0]
    selected_test_row = test_comparison.loc[test_comparison["threshold_type"] == "cost_sensitive"].iloc[0]
    plot_confusion_matrix(
        int(default_row["tn"]),
        int(default_row["fp"]),
        int(default_row["fn"]),
        int(default_row["tp"]),
        figures_dir / "time_split_confusion_matrix_default_test.png",
        title="Time Split Test Confusion Matrix: threshold 0.5",
    )
    plot_confusion_matrix(
        int(selected_test_row["tn"]),
        int(selected_test_row["fp"]),
        int(selected_test_row["fn"]),
        int(selected_test_row["tp"]),
        figures_dir / "time_split_confusion_matrix_cost_sensitive_test.png",
        title="Time Split Test Confusion Matrix: cost-sensitive threshold",
    )
    plot_vs_random(vs_random, "pr_auc", figures_dir / "time_split_vs_random_pr_auc.png")
    plot_vs_random(vs_random, "cost", figures_dir / "time_split_vs_random_cost.png")

    return {
        "summary": summary,
        "selected_threshold": selected_threshold,
        "test_comparison": test_comparison,
        "vs_random": vs_random,
        "model": model_name,
        "feature_group": feature_group,
        "full_rows": sum(len(split) for split in splits.values()),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the time-split experiment from the command line."""

    args = parse_args(argv)
    try:
        result = run_time_split_experiment(
            config_path=args.config,
            data_dir=args.data_dir,
            metadata_path=args.metadata,
            feature_groups_path=args.feature_groups,
            random_threshold_results_path=args.random_threshold_results,
            selected_threshold_path=args.selected_threshold,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    summary = result["summary"].set_index("split")
    test = result["test_comparison"].set_index("threshold_type")
    random_default = result["vs_random"].set_index(["split_strategy", "threshold_type"]).loc[
        ("random_stage3", "default_0_5")
    ]
    print("Time-split extension experiment complete")
    print(f"Full rows: {result['full_rows']}")
    for split in ("train", "valid", "test"):
        print(f"{split} rows: {int(summary.loc[split, 'n_rows'])}")
        print(f"{split} fraud rate: {summary.loc[split, 'positive_rate']:.6f}")
    print(f"Selected threshold: {result['selected_threshold']}")
    print(f"Time-split test PR-AUC: {test.loc['default_0_5', 'pr_auc']:.6f}")
    print(f"Default threshold cost: {int(test.loc['default_0_5', 'cost'])}")
    print(f"Selected threshold cost: {int(test.loc['cost_sensitive', 'cost'])}")
    print(
        "Random split PR-AUC vs time split PR-AUC: "
        f"{random_default['pr_auc']:.6f} vs {test.loc['default_0_5', 'pr_auc']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
