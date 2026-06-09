"""Threshold search and Stage 3 cost-sensitive evaluation workflow."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluate import plot_confusion_matrix
from src.metrics import calculate_confusion_counts, get_probability_scores
from src.models import build_model, build_training_pipeline
from src.preprocessing import (
    build_preprocessor,
    build_tree_preprocessor,
    load_feature_groups,
    load_metadata,
    split_features_target,
    validate_requested_features,
)
from src.train import measure_training_time


LOGGER = logging.getLogger(__name__)
STAGE2_FINAL_TEST_PATH = Path("reports/tables/stage2_final_test.csv")
STAGE2_ABLATION_VALID_PATH = Path("reports/tables/stage2_ablation_valid.csv")
STAGE2_MODEL_COMPARISON_VALID_PATH = Path("reports/tables/stage2_model_comparison_valid.csv")
TREE_MODELS = {"lightgbm", "hist_gradient_boosting", "random_forest"}
PREDICTION_CONTEXT_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "missing_count",
)


def generate_threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    """Generate a stable inclusive threshold grid."""

    minimum_dec = Decimal(str(minimum))
    maximum_dec = Decimal(str(maximum))
    step_dec = Decimal(str(step))
    thresholds: list[float] = []
    current = minimum_dec
    while current <= maximum_dec:
        thresholds.append(float(current.quantize(Decimal("0.01"))))
        current += step_dec
    return thresholds


def calculate_business_cost(
    fn: int,
    fp: int,
    false_negative_cost: int = 10,
    false_positive_cost: int = 1,
) -> int:
    """Calculate simple business cost from false negatives and false positives."""

    return int(false_negative_cost * fn + false_positive_cost * fp)


def apply_fixed_threshold(y_score: Any, threshold: float) -> np.ndarray:
    """Apply a fixed threshold to score values."""

    return (np.asarray(y_score) >= threshold).astype(int)


def assign_error_type(y_true: Any, y_pred: Any) -> np.ndarray:
    """Assign TP, TN, FP, or FN labels for each prediction."""

    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    labels = np.empty(len(true_values), dtype=object)
    labels[(true_values == 1) & (pred_values == 1)] = "TP"
    labels[(true_values == 0) & (pred_values == 0)] = "TN"
    labels[(true_values == 0) & (pred_values == 1)] = "FP"
    labels[(true_values == 1) & (pred_values == 0)] = "FN"
    return labels


def evaluate_threshold(
    y_true: Any,
    y_score: Any,
    threshold: float,
    fn_cost: int = 10,
    fp_cost: int = 1,
) -> dict[str, float | int]:
    """Evaluate classification metrics at a fixed threshold."""

    y_true_array = np.asarray(y_true)
    y_pred = apply_fixed_threshold(y_score, threshold)
    tn, fp, fn, tp = calculate_confusion_counts(y_true_array, y_pred)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "mcc": float(matthews_corrcoef(y_true_array, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost": calculate_business_cost(fn, fp, fn_cost, fp_cost),
        "n_samples": int(len(y_true_array)),
        "positive_support": int(np.sum(y_true_array == 1)),
    }


def search_optimal_threshold(
    y_true: Any,
    y_score: Any,
    thresholds: list[float],
    fn_cost: int = 10,
    fp_cost: int = 1,
) -> tuple[float, pd.DataFrame]:
    """Search validation thresholds using cost, recall, precision, then closeness to 0.5."""

    search_df = pd.DataFrame(
        [evaluate_threshold(y_true, y_score, threshold, fn_cost, fp_cost) for threshold in thresholds]
    )
    ranked = search_df.assign(distance_to_0_5=(search_df["threshold"] - 0.5).abs()).sort_values(
        ["cost", "recall", "precision", "distance_to_0_5", "threshold"],
        ascending=[True, False, False, True, True],
    )
    selected_threshold = float(ranked.iloc[0]["threshold"])
    return selected_threshold, search_df


def save_threshold_search_results(df: pd.DataFrame, output_path: Path) -> Path:
    """Save validation threshold search results."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_selected_threshold(summary: dict[str, Any], output_path: Path) -> Path:
    """Save selected threshold summary JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_threshold_analysis(
    config: dict[str, Any],
    data_dir: Path,
    metadata_path: Path,
    feature_groups_path: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """Run Stage 3 validation threshold selection and fixed-threshold test comparison."""

    _validate_required_inputs(data_dir, metadata_path, feature_groups_path)
    frozen = freeze_stage2_selection()
    metadata = load_metadata(metadata_path)
    if metadata.get("stage") != "stage2_formal":
        raise ValueError("Metadata stage must be stage2_formal for Stage 3.")

    train_df = pd.read_parquet(data_dir / "train.parquet")
    valid_df = pd.read_parquet(data_dir / "valid.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    _validate_split_rows(metadata, train_df, valid_df, test_df)

    feature_groups = load_feature_groups(feature_groups_path)
    target = metadata.get("target_column", "isFraud")
    id_column = metadata.get("id_column", "TransactionID")
    time_column = metadata.get("time_column", "TransactionDT")
    selected_group = str(frozen["selected_feature_group"])
    actual_model_name = str(frozen["actual_model_name"])
    features = validate_requested_features(
        train_df, feature_groups, selected_group, target, id_column, time_column
    )
    validate_requested_features(valid_df, feature_groups, selected_group, target, id_column, time_column)
    validate_requested_features(test_df, feature_groups, selected_group, target, id_column, time_column)

    threshold_config = config["threshold_optimization"]
    thresholds = generate_threshold_grid(
        threshold_config["minimum_threshold"],
        threshold_config["maximum_threshold"],
        threshold_config["threshold_step"],
    )
    fn_cost = int(threshold_config["false_negative_cost"])
    fp_cost = int(threshold_config["false_positive_cost"])

    valid_pipeline, _ = _fit_pipeline(
        config, train_df, metadata, features, actual_model_name, target
    )
    X_valid, y_valid = split_features_target(valid_df, features, target)
    valid_score = get_probability_scores(valid_pipeline, X_valid)
    selected_threshold, search_df = search_optimal_threshold(
        y_valid, valid_score, thresholds, fn_cost=fn_cost, fp_cost=fp_cost
    )
    selected_row = search_df.loc[search_df["threshold"] == selected_threshold].iloc[0]

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"
    save_threshold_search_results(search_df, tables_dir / "stage3_threshold_search_valid.csv")
    selected_summary = _build_selected_threshold_summary(
        selected_row, selected_threshold, frozen, fn_cost, fp_cost
    )
    save_selected_threshold(selected_summary, tables_dir / "stage3_selected_threshold.json")
    _save_prediction_file(
        valid_df,
        target,
        valid_score,
        selected_threshold,
        predictions_dir / "stage3_validation_predictions.parquet",
    )

    final_train = pd.concat([train_df, valid_df], ignore_index=True)
    final_pipeline, _ = _fit_pipeline(
        config, final_train, metadata, features, actual_model_name, target
    )
    X_test, y_test = split_features_target(test_df, features, target)
    test_score = get_probability_scores(final_pipeline, X_test)
    comparison_df = _build_test_comparison(
        y_test,
        test_score,
        float(config.get("default_threshold", 0.5)),
        selected_threshold,
        actual_model_name,
        selected_group,
        fn_cost,
        fp_cost,
    )
    comparison_path = tables_dir / "stage3_threshold_comparison_test.csv"
    comparison_df.to_csv(comparison_path, index=False)
    _save_prediction_file(
        test_df,
        target,
        test_score,
        selected_threshold,
        predictions_dir / "stage3_test_predictions.parquet",
    )
    _plot_stage3_figures(search_df, comparison_df, selected_threshold, figures_dir)

    return {
        "frozen": frozen,
        "selected_threshold": selected_threshold,
        "selected_summary": selected_summary,
        "comparison": comparison_df,
        "threshold_count": len(search_df),
        "paths": {
            "threshold_search": tables_dir / "stage3_threshold_search_valid.csv",
            "selected_threshold": tables_dir / "stage3_selected_threshold.json",
            "test_comparison": comparison_path,
            "validation_predictions": predictions_dir / "stage3_validation_predictions.parquet",
            "test_predictions": predictions_dir / "stage3_test_predictions.parquet",
        },
    }


def freeze_stage2_selection() -> dict[str, Any]:
    """Freeze Stage 2 model and feature-group selection from result tables."""

    if not STAGE2_FINAL_TEST_PATH.exists():
        raise FileNotFoundError(f"Stage 2 final test table not found: {STAGE2_FINAL_TEST_PATH}")
    final_test = pd.read_csv(STAGE2_FINAL_TEST_PATH)
    if len(final_test) != 1:
        raise ValueError("stage2_final_test.csv must contain exactly one final selection row.")
    row = final_test.iloc[0]
    selected_group = _unique_or_missing(final_test, "selected_feature_group")
    actual_model_name = _unique_or_missing(final_test, "actual_model_name")
    if selected_group is None or actual_model_name is None:
        selected_group, actual_model_name = _freeze_from_ablation()

    return {
        "selected_feature_group": selected_group,
        "actual_model_name": actual_model_name,
        "validation_selection_score": float(row["validation_selection_score"]),
        "random_seed": int(row["random_seed"]),
    }


def _unique_or_missing(table: pd.DataFrame, column: str) -> Any | None:
    if column not in table.columns:
        return None
    values = table[column].dropna().unique()
    if len(values) == 1:
        return values[0]
    return None


def _freeze_from_ablation() -> tuple[str, str]:
    if not STAGE2_ABLATION_VALID_PATH.exists() or not STAGE2_MODEL_COMPARISON_VALID_PATH.exists():
        raise ValueError("Cannot uniquely determine Stage 2 selection from available result tables.")
    ablation = pd.read_csv(STAGE2_ABLATION_VALID_PATH)
    best_score = ablation["pr_auc"].max()
    best_rows = ablation.loc[ablation["pr_auc"] == best_score]
    if len(best_rows) != 1:
        raise ValueError("Cannot uniquely determine selected feature group from valid PR-AUC.")
    model_comparison = pd.read_csv(STAGE2_MODEL_COMPARISON_VALID_PATH)
    best_model_score = model_comparison["pr_auc"].max()
    best_model_rows = model_comparison.loc[model_comparison["pr_auc"] == best_model_score]
    if len(best_model_rows) != 1:
        raise ValueError("Cannot uniquely determine selected model from valid PR-AUC.")
    return str(best_rows.iloc[0]["feature_group"]), str(best_model_rows.iloc[0]["actual_model_name"])


def _validate_required_inputs(data_dir: Path, metadata_path: Path, feature_groups_path: Path) -> None:
    for stage2_path in (
        STAGE2_MODEL_COMPARISON_VALID_PATH,
        STAGE2_ABLATION_VALID_PATH,
        STAGE2_FINAL_TEST_PATH,
    ):
        if not stage2_path.exists():
            raise FileNotFoundError(f"Required Stage 2 result table not found: {stage2_path}")
    for split in ("train", "valid", "test"):
        split_path = data_dir / f"{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Required Stage 2 {split} parquet not found: {split_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not feature_groups_path.exists():
        raise FileNotFoundError(f"Feature groups file not found: {feature_groups_path}")


def _validate_split_rows(
    metadata: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    expected_rows = {
        "train": int(metadata["train_rows"]),
        "valid": int(metadata["valid_rows"]),
        "test": int(metadata["test_rows"]),
    }
    actual_rows = {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}
    mismatches = [
        f"{split}: expected {expected_rows[split]}, got {actual_rows[split]}"
        for split in expected_rows
        if expected_rows[split] != actual_rows[split]
    ]
    if mismatches:
        raise ValueError("Split row count mismatch. " + "; ".join(mismatches))


def _fit_pipeline(
    config: dict[str, Any],
    dataframe: pd.DataFrame,
    metadata: dict[str, Any],
    features: list[str],
    actual_model_name: str,
    target: str,
) -> tuple[object, float]:
    numeric = [column for column in metadata["numeric_columns"] if column in features]
    categorical = [column for column in metadata["categorical_columns"] if column in features]
    model_params = dict(config["models"].get(actual_model_name, {}) or {})
    model_params.pop("enabled", None)
    estimator = build_model(actual_model_name, int(config["random_seed"]), model_params)
    preprocessor = (
        build_tree_preprocessor(numeric, categorical, config["preprocessing"])
        if actual_model_name in TREE_MODELS
        else build_preprocessor(numeric, categorical, config["preprocessing"])
    )
    pipeline = build_training_pipeline(preprocessor, estimator)
    X_train, y_train = split_features_target(dataframe, features, target)
    return measure_training_time(pipeline, X_train, y_train)


def _build_selected_threshold_summary(
    selected_row: pd.Series,
    selected_threshold: float,
    frozen: dict[str, Any],
    fn_cost: int,
    fp_cost: int,
) -> dict[str, Any]:
    return {
        "selection_split": "validation",
        "cost_formula": f"{fn_cost} * FN + {fp_cost} * FP",
        "false_negative_cost": fn_cost,
        "false_positive_cost": fp_cost,
        "selected_threshold": selected_threshold,
        "validation_cost": int(selected_row["cost"]),
        "validation_precision": float(selected_row["precision"]),
        "validation_recall": float(selected_row["recall"]),
        "validation_f1": float(selected_row["f1"]),
        "validation_tn": int(selected_row["tn"]),
        "validation_fp": int(selected_row["fp"]),
        "validation_fn": int(selected_row["fn"]),
        "validation_tp": int(selected_row["tp"]),
        "selected_model": frozen["actual_model_name"],
        "selected_feature_group": frozen["selected_feature_group"],
        "validation_selection_score": frozen["validation_selection_score"],
    }


def _build_test_comparison(
    y_true: Any,
    y_score: Any,
    default_threshold: float,
    selected_threshold: float,
    model_name: str,
    feature_group: str,
    fn_cost: int,
    fp_cost: int,
) -> pd.DataFrame:
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    rows: list[dict[str, Any]] = []
    for threshold_type, threshold in (
        ("default_0_5", default_threshold),
        ("cost_sensitive", selected_threshold),
    ):
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


def _save_prediction_file(
    dataframe: pd.DataFrame,
    target: str,
    y_score: Any,
    selected_threshold: float,
    output_path: Path,
) -> Path:
    pred_default = apply_fixed_threshold(y_score, 0.5)
    pred_selected = apply_fixed_threshold(y_score, selected_threshold)
    keep_columns = [column for column in PREDICTION_CONTEXT_COLUMNS if column in dataframe.columns]
    predictions = dataframe.loc[:, keep_columns].copy()
    if target not in predictions.columns and target in dataframe.columns:
        predictions[target] = dataframe[target]
    predictions["prediction_score"] = np.asarray(y_score)
    predictions["prediction_at_0_5"] = pred_default
    predictions["prediction_at_selected_threshold"] = pred_selected
    predictions["error_type_at_0_5"] = assign_error_type(dataframe[target], pred_default)
    predictions["error_type_at_selected_threshold"] = assign_error_type(dataframe[target], pred_selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    return output_path


def _plot_stage3_figures(
    search_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    selected_threshold: float,
    figures_dir: Path,
) -> None:
    _plot_threshold_metric(
        search_df,
        "cost",
        selected_threshold,
        figures_dir / "stage3_threshold_cost_valid.png",
        "Validation Cost by Threshold",
    )
    _plot_threshold_metric(
        search_df,
        "precision",
        selected_threshold,
        figures_dir / "stage3_threshold_precision_valid.png",
        "Validation Precision by Threshold",
    )
    _plot_threshold_metric(
        search_df,
        "recall",
        selected_threshold,
        figures_dir / "stage3_threshold_recall_valid.png",
        "Validation Recall by Threshold",
    )
    default_row = comparison_df.loc[comparison_df["threshold_type"] == "default_0_5"].iloc[0]
    optimized_row = comparison_df.loc[comparison_df["threshold_type"] == "cost_sensitive"].iloc[0]
    plot_confusion_matrix(
        int(default_row["tn"]),
        int(default_row["fp"]),
        int(default_row["fn"]),
        int(default_row["tp"]),
        figures_dir / "stage3_confusion_matrix_default_test.png",
        title="Test Confusion Matrix: threshold 0.5",
    )
    plot_confusion_matrix(
        int(optimized_row["tn"]),
        int(optimized_row["fp"]),
        int(optimized_row["fn"]),
        int(optimized_row["tp"]),
        figures_dir / "stage3_confusion_matrix_cost_sensitive_test.png",
        title="Test Confusion Matrix: cost-sensitive threshold",
    )
    _plot_test_comparison(comparison_df, figures_dir / "stage3_threshold_comparison_test.png")


def _plot_threshold_metric(
    search_df: pd.DataFrame,
    metric: str,
    selected_threshold: float,
    output_path: Path,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(8, 4))
    axis.plot(search_df["threshold"], search_df[metric])
    axis.axvline(selected_threshold, color="red", linestyle="--", label="selected threshold")
    axis.set_xlabel("threshold")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _plot_test_comparison(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["precision", "recall", "f1", "cost"]
    x = np.arange(len(metrics))
    width = 0.35
    default_values = comparison_df.loc[comparison_df["threshold_type"] == "default_0_5", metrics].iloc[0]
    optimized_values = comparison_df.loc[
        comparison_df["threshold_type"] == "cost_sensitive", metrics
    ].iloc[0]
    _, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - width / 2, default_values, width, label="default_0_5")
    axis.bar(x + width / 2, optimized_values, width, label="cost_sensitive")
    axis.set_xticks(x, labels=metrics)
    axis.set_title("Test Threshold Comparison")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
