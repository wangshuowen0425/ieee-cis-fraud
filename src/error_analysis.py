"""Stage 3-A error analysis based on saved Stage 3 prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 设置为非交互后端

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "isFraud",
    "prediction_score",
    "prediction_at_0_5",
    "prediction_at_selected_threshold",
    "error_type_at_0_5",
    "error_type_at_selected_threshold",
)
ERROR_TYPES: tuple[str, ...] = ("TP", "TN", "FP", "FN")
CATEGORY_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "P_emaildomain",
    "R_emaildomain",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run Stage 3-A FP/FN error analysis.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/predictions/stage3_test_predictions.parquet"),
    )
    parser.add_argument(
        "--threshold-results",
        type=Path,
        default=Path("reports/tables/stage3_threshold_comparison_test.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/stage2_formal/metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def assign_error_type(y_true: Any, y_pred: Any) -> pd.Series:
    """Assign TP, TN, FP, or FN labels for each binary prediction."""

    true_values = pd.Series(y_true).astype(int)
    pred_values = pd.Series(y_pred).astype(int)
    labels = pd.Series(index=true_values.index, dtype="object")
    labels[(true_values == 1) & (pred_values == 1)] = "TP"
    labels[(true_values == 0) & (pred_values == 0)] = "TN"
    labels[(true_values == 0) & (pred_values == 1)] = "FP"
    labels[(true_values == 1) & (pred_values == 0)] = "FN"
    return labels


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata JSON."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata file must contain a JSON object.")
    return metadata


def validate_inputs(
    predictions: pd.DataFrame,
    threshold_results: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate predictions against metadata and threshold comparison results."""

    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing_columns:
        raise ValueError(f"Prediction file missing required columns: {', '.join(missing_columns)}.")

    expected_rows = metadata.get("test_rows")
    if expected_rows is not None and len(predictions) != int(expected_rows):
        raise ValueError(f"Prediction row count {len(predictions)} does not match metadata test_rows {expected_rows}.")

    if predictions["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique in prediction file.")
    if not set(predictions["isFraud"].dropna().unique()).issubset({0, 1}):
        raise ValueError("isFraud must contain only 0 and 1.")
    if not predictions["prediction_score"].between(0, 1).all():
        raise ValueError("prediction_score must be between 0 and 1.")

    for column in ("error_type_at_0_5", "error_type_at_selected_threshold"):
        invalid_values = set(predictions[column].dropna().unique()) - set(ERROR_TYPES)
        if invalid_values:
            raise ValueError(f"{column} contains invalid error type(s): {sorted(invalid_values)}.")

    recomputed_default = assign_error_type(predictions["isFraud"], predictions["prediction_at_0_5"])
    recomputed_selected = assign_error_type(
        predictions["isFraud"], predictions["prediction_at_selected_threshold"]
    )
    if not recomputed_default.reset_index(drop=True).equals(
        predictions["error_type_at_0_5"].reset_index(drop=True)
    ):
        raise ValueError("error_type_at_0_5 does not match isFraud and prediction_at_0_5.")
    if not recomputed_selected.reset_index(drop=True).equals(
        predictions["error_type_at_selected_threshold"].reset_index(drop=True)
    ):
        raise ValueError(
            "error_type_at_selected_threshold does not match isFraud and prediction_at_selected_threshold."
        )

    _validate_against_threshold_results(predictions, threshold_results)
    return {
        "row_count": len(predictions),
        "unique_transaction_ids": int(predictions["TransactionID"].nunique()),
    }


def _validate_against_threshold_results(
    predictions: pd.DataFrame,
    threshold_results: pd.DataFrame,
) -> None:
    required_columns = {"threshold_type", "tn", "fp", "fn", "tp"}
    if not required_columns.issubset(threshold_results.columns):
        raise ValueError("Threshold comparison table missing confusion matrix columns.")

    mapping = {
        "default_0_5": "error_type_at_0_5",
        "cost_sensitive": "error_type_at_selected_threshold",
    }
    for threshold_type, error_column in mapping.items():
        rows = threshold_results.loc[threshold_results["threshold_type"] == threshold_type]
        if len(rows) != 1:
            raise ValueError(f"Threshold comparison table must contain one row for {threshold_type}.")
        row = rows.iloc[0]
        counts = predictions[error_column].value_counts()
        actual = {
            "tn": int(counts.get("TN", 0)),
            "fp": int(counts.get("FP", 0)),
            "fn": int(counts.get("FN", 0)),
            "tp": int(counts.get("TP", 0)),
        }
        expected = {key: int(row[key]) for key in actual}
        if actual != expected:
            raise ValueError(
                f"Prediction error counts do not match threshold table for {threshold_type}: "
                f"actual={actual}, expected={expected}."
            )


def build_error_count_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare TP/TN/FP/FN counts between default and selected thresholds."""

    default_counts = predictions["error_type_at_0_5"].value_counts()
    selected_counts = predictions["error_type_at_selected_threshold"].value_counts()
    rows: list[dict[str, Any]] = []
    for error_type in ERROR_TYPES:
        default_count = int(default_counts.get(error_type, 0))
        selected_count = int(selected_counts.get(error_type, 0))
        absolute_change = selected_count - default_count
        relative_change = np.nan if default_count == 0 else absolute_change / default_count
        rows.append(
            {
                "error_type": error_type,
                "threshold_0_5_count": default_count,
                "selected_threshold_count": selected_count,
                "absolute_change": absolute_change,
                "relative_change": relative_change,
            }
        )
    return pd.DataFrame(rows)


def build_amount_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize TransactionAmt by optimized-threshold error type."""

    columns = ["error_type", "count", "mean", "median", "q25", "q75", "maximum"]
    if "TransactionAmt" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    grouped = predictions.groupby("error_type_at_selected_threshold")["TransactionAmt"]
    summary = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
        maximum="max",
    ).reset_index(names="error_type")
    return summary.loc[:, columns]


def build_category_summary(predictions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Build top category summaries by optimized-threshold error type."""

    rows: list[pd.DataFrame] = []
    for feature in CATEGORY_COLUMNS:
        if feature not in predictions.columns:
            continue
        frame = predictions.loc[:, ["error_type_at_selected_threshold", feature]].copy()
        frame[feature] = frame[feature].fillna("MISSING").astype(str).replace({"": "MISSING"})
        counts = (
            frame.groupby(["error_type_at_selected_threshold", feature])
            .size()
            .reset_index(name="count")
            .rename(columns={"error_type_at_selected_threshold": "error_type", feature: "category"})
        )
        totals = counts.groupby("error_type")["count"].transform("sum")
        counts["share_within_error_type"] = counts["count"] / totals
        counts["feature"] = feature
        rows.append(
            counts.sort_values(["error_type", "count", "category"], ascending=[True, False, True])
            .groupby("error_type", group_keys=False)
            .head(top_n)
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "error_type", "category", "count", "share_within_error_type"])
    return pd.concat(rows, ignore_index=True).loc[
        :, ["feature", "error_type", "category", "count", "share_within_error_type"]
    ]


def select_representative_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select deterministic representative FP/FN cases from optimized-threshold errors."""

    output_columns = [
        "TransactionID",
        "isFraud",
        "prediction_score",
        "error_type",
        "TransactionAmt",
        "ProductCD",
        "DeviceType",
        "missing_count",
        "selection_reason",
    ]
    frames: list[pd.DataFrame] = []
    errors = predictions.copy()
    errors["error_type"] = errors["error_type_at_selected_threshold"]
    fp = errors.loc[errors["error_type"] == "FP"].copy()
    fn = errors.loc[errors["error_type"] == "FN"].copy()
    frames.append(_with_reason(fp.nlargest(5, "prediction_score"), "highest_score_false_positive"))
    frames.append(_with_reason(fn.nsmallest(5, "prediction_score"), "lowest_score_false_negative"))
    if "TransactionAmt" in errors.columns:
        frames.append(_with_reason(fp.nlargest(5, "TransactionAmt"), "highest_amount_false_positive"))
        frames.append(_with_reason(fn.nlargest(5, "TransactionAmt"), "highest_amount_false_negative"))
    representative = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if representative.empty:
        return pd.DataFrame(columns=[column for column in output_columns if column in predictions.columns or column in {"error_type", "selection_reason"}])
    representative = representative.drop_duplicates(subset=["TransactionID"], keep="first")
    available_columns = [
        column
        for column in output_columns
        if column in representative.columns or column in {"error_type", "selection_reason"}
    ]
    return representative.loc[:, available_columns]


def _with_reason(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    result = frame.copy()
    result["selection_reason"] = reason
    return result


def save_table(table: pd.DataFrame, output_path: Path) -> Path:
    """Save a CSV table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def plot_fp_fn_comparison(error_counts: pd.DataFrame, output_path: Path) -> Path:
    """Plot FP and FN counts at default and selected thresholds."""

    subset = error_counts.loc[error_counts["error_type"].isin(["FP", "FN"])]
    x = np.arange(len(subset))
    width = 0.35
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - width / 2, subset["threshold_0_5_count"], width, label="threshold 0.5")
    axis.bar(x + width / 2, subset["selected_threshold_count"], width, label="selected threshold")
    axis.set_xticks(x, labels=subset["error_type"])
    axis.set_ylabel("count")
    axis.set_title("FP/FN Count Comparison")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_amount_boxplot(predictions: pd.DataFrame, output_path: Path) -> Path | None:
    """Plot log1p(TransactionAmt) by optimized-threshold error type."""

    if "TransactionAmt" not in predictions.columns:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        np.log1p(predictions.loc[predictions["error_type_at_selected_threshold"] == error_type, "TransactionAmt"].dropna())
        for error_type in ERROR_TYPES
    ]
    _, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot(data, labels=ERROR_TYPES)
    axis.set_ylabel("log1p(TransactionAmt)")
    axis.set_title("Transaction Amount by Error Type (log1p)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_missing_count_by_error_type(predictions: pd.DataFrame, output_path: Path) -> Path | None:
    """Plot missing_count distribution by optimized-threshold error type."""

    if "missing_count" not in predictions.columns:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        predictions.loc[predictions["error_type_at_selected_threshold"] == error_type, "missing_count"].dropna()
        for error_type in ERROR_TYPES
    ]
    _, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot(data, labels=ERROR_TYPES)
    axis.set_ylabel("missing_count")
    axis.set_title("Missing Count by Error Type")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_error_by_product_code(predictions: pd.DataFrame, output_path: Path) -> Path | None:
    """Plot optimized-threshold error type counts by ProductCD."""

    if "ProductCD" not in predictions.columns:
        return None
    frame = predictions.loc[:, ["ProductCD", "error_type_at_selected_threshold"]].copy()
    frame["ProductCD"] = frame["ProductCD"].fillna("MISSING").astype(str)
    if frame["ProductCD"].nunique() < 2:
        return None
    pivot = pd.crosstab(frame["ProductCD"], frame["error_type_at_selected_threshold"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis = pivot.loc[:, [col for col in ERROR_TYPES if col in pivot.columns]].plot(kind="bar", stacked=True, figsize=(8, 4))
    axis.set_xlabel("ProductCD")
    axis.set_ylabel("count")
    axis.set_title("Error Type Distribution by ProductCD")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def run_error_analysis(
    predictions_path: Path,
    threshold_results_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate inputs, generate Stage 3-A tables, and render figures."""

    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")
    if not threshold_results_path.exists():
        raise FileNotFoundError(f"Threshold comparison table not found: {threshold_results_path}")

    predictions = pd.read_parquet(predictions_path)
    threshold_results = pd.read_csv(threshold_results_path)
    metadata = load_metadata(metadata_path)
    validation = validate_inputs(predictions, threshold_results, metadata)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    error_counts = build_error_count_comparison(predictions)
    amount_summary = build_amount_summary(predictions)
    category_summary = build_category_summary(predictions)
    representative = select_representative_errors(predictions)

    save_table(error_counts, tables_dir / "stage3_error_count_comparison.csv")
    save_table(amount_summary, tables_dir / "stage3_error_amount_summary.csv")
    save_table(category_summary, tables_dir / "stage3_error_category_summary.csv")
    save_table(representative, tables_dir / "stage3_representative_errors.csv")

    generated_figures = [
        plot_fp_fn_comparison(error_counts, figures_dir / "stage3_fp_fn_comparison.png"),
        plot_amount_boxplot(predictions, figures_dir / "stage3_error_amount_boxplot.png"),
        plot_missing_count_by_error_type(predictions, figures_dir / "stage3_missing_count_by_error_type.png"),
        plot_error_by_product_code(predictions, figures_dir / "stage3_error_by_product_code.png"),
    ]

    return {
        "validation": validation,
        "error_counts": error_counts,
        "amount_summary": amount_summary,
        "category_summary": category_summary,
        "representative": representative,
        "figures": [path for path in generated_figures if path is not None],
    }


def main(argv: list[str] | None = None) -> int:
    """Run Stage 3-A error analysis from the command line."""

    args = parse_args(argv)
    try:
        result = run_error_analysis(
            predictions_path=args.predictions,
            threshold_results_path=args.threshold_results,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    counts = result["error_counts"].set_index("error_type")
    fp_change = int(counts.loc["FP", "absolute_change"])
    fn_change = int(counts.loc["FN", "absolute_change"])
    print("Stage 3-A error analysis complete")
    print(f"Rows validated: {result['validation']['row_count']}")
    print(f"FP change: {fp_change}")
    print(f"FN change: {fn_change}")
    print(f"Amount summary generated: {not result['amount_summary'].empty}")
    print(f"Category summary generated: {not result['category_summary'].empty}")
    print(f"Representative cases: {len(result['representative'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
