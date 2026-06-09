"""Tests for Stage 3-A error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.error_analysis import (
    assign_error_type,
    build_amount_summary,
    build_category_summary,
    build_error_count_comparison,
    run_error_analysis,
    select_representative_errors,
    validate_inputs,
)


def _predictions(include_optional: bool = True) -> pd.DataFrame:
    data = {
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "isFraud": [0, 1, 0, 1, 0, 1],
        "prediction_score": [0.9, 0.8, 0.2, 0.1, 0.7, 0.4],
        "prediction_at_0_5": [1, 1, 0, 0, 1, 0],
        "prediction_at_selected_threshold": [1, 1, 0, 0, 1, 1],
    }
    predictions = pd.DataFrame(data)
    predictions["error_type_at_0_5"] = assign_error_type(
        predictions["isFraud"], predictions["prediction_at_0_5"]
    )
    predictions["error_type_at_selected_threshold"] = assign_error_type(
        predictions["isFraud"], predictions["prediction_at_selected_threshold"]
    )
    if include_optional:
        predictions["TransactionAmt"] = [100.0, 20.0, 5.0, 500.0, 300.0, 40.0]
        predictions["ProductCD"] = ["W", "W", None, "C", "W", "C"]
        predictions["card4"] = ["visa", "visa", "mc", None, "visa", "mc"]
        predictions["DeviceType"] = ["desktop", None, "mobile", "desktop", "mobile", "desktop"]
        predictions["missing_count"] = [1, 2, 0, 5, 3, 4]
    return predictions


def _threshold_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "threshold_type": "default_0_5",
                "tn": 1,
                "fp": 2,
                "fn": 2,
                "tp": 1,
            },
            {
                "threshold_type": "cost_sensitive",
                "tn": 1,
                "fp": 2,
                "fn": 1,
                "tp": 2,
            },
        ]
    )


def test_error_type_recalculation_is_correct() -> None:
    labels = assign_error_type([1, 0, 0, 1], [1, 0, 1, 0])

    assert labels.tolist() == ["TP", "TN", "FP", "FN"]


def test_invalid_error_type_is_detected() -> None:
    predictions = _predictions()
    predictions.loc[0, "error_type_at_selected_threshold"] = "WRONG"

    try:
        validate_inputs(predictions, _threshold_results(), {"test_rows": 6})
    except ValueError as error:
        assert "invalid error type" in str(error)
    else:
        raise AssertionError("Expected invalid error_type to fail validation.")


def test_error_count_changes_and_zero_denominator() -> None:
    predictions = _predictions()
    comparison = build_error_count_comparison(predictions)

    fp_row = comparison.loc[comparison["error_type"] == "FP"].iloc[0]
    fn_row = comparison.loc[comparison["error_type"] == "FN"].iloc[0]
    assert fp_row["absolute_change"] == 0
    assert fn_row["absolute_change"] == -1

    no_tp = predictions.loc[predictions["error_type_at_0_5"] != "TP"].copy()
    zero_comparison = build_error_count_comparison(no_tp)
    tp_relative = zero_comparison.loc[zero_comparison["error_type"] == "TP", "relative_change"].iloc[0]
    assert np.isnan(tp_relative)


def test_amount_quantiles_are_correct() -> None:
    summary = build_amount_summary(_predictions())
    fp_summary = summary.loc[summary["error_type"] == "FP"].iloc[0]

    assert fp_summary["median"] == 200.0
    assert fp_summary["q25"] == 150.0
    assert fp_summary["q75"] == 250.0


def test_missing_categories_and_top_n() -> None:
    predictions = _predictions()
    summary = build_category_summary(predictions, top_n=1)

    assert "MISSING" in set(summary["category"])
    assert summary.groupby(["feature", "error_type"]).size().max() == 1


def test_representative_selection_and_deduplication() -> None:
    representatives = select_representative_errors(_predictions())

    assert "highest_score_false_positive" in set(representatives["selection_reason"])
    assert "lowest_score_false_negative" in set(representatives["selection_reason"])
    assert representatives["TransactionID"].is_unique


def test_optional_fields_can_be_absent() -> None:
    predictions = _predictions(include_optional=False)

    assert build_amount_summary(predictions).empty
    assert build_category_summary(predictions).empty
    representatives = select_representative_errors(predictions)
    assert "TransactionAmt" not in representatives.columns


def test_run_does_not_modify_prediction_file_and_generates_outputs(tmp_path: Path) -> None:
    predictions = _predictions()
    predictions_path = tmp_path / "stage3_test_predictions.parquet"
    threshold_path = tmp_path / "stage3_threshold_comparison_test.csv"
    metadata_path = tmp_path / "metadata.json"
    output_dir = tmp_path / "reports"
    predictions.to_parquet(predictions_path, index=False)
    threshold_results = _threshold_results()
    threshold_results["threshold"] = [0.5, 0.1]
    threshold_results.to_csv(threshold_path, index=False)
    metadata_path.write_text(json.dumps({"test_rows": 6}), encoding="utf-8")
    before_bytes = predictions_path.read_bytes()

    result = run_error_analysis(predictions_path, threshold_path, metadata_path, output_dir)

    assert predictions_path.read_bytes() == before_bytes
    assert len(result["representative"]) > 0
    assert (output_dir / "tables" / "stage3_error_count_comparison.csv").exists()
    assert (output_dir / "tables" / "stage3_error_amount_summary.csv").exists()
    assert (output_dir / "tables" / "stage3_error_category_summary.csv").exists()
    assert (output_dir / "tables" / "stage3_representative_errors.csv").exists()
    assert (output_dir / "figures" / "stage3_fp_fn_comparison.png").exists()
    assert (output_dir / "figures" / "stage3_error_amount_boxplot.png").exists()
    assert (output_dir / "figures" / "stage3_missing_count_by_error_type.png").exists()
    assert (output_dir / "figures" / "stage3_error_by_product_code.png").exists()
