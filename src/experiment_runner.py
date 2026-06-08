"""Stage 2 formal model experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    save_results_table,
)
from src.models import build_model, build_training_pipeline, resolve_tree_model
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
TREE_MODEL_ALIAS = "resolved_tree_model"


def read_formal_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read stage 2 train, valid, and test parquet splits."""

    return {
        split: pd.read_parquet(data_dir / f"{split}.parquet")
        for split in ("train", "valid", "test")
    }


def _model_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models_config = config.get("models", {})
    params = dict(models_config.get(model_name, {}) or {})
    params.pop("enabled", None)
    return params


def _columns_for_group(
    metadata: dict[str, Any],
    features: list[str],
) -> tuple[list[str], list[str]]:
    numeric = [column for column in metadata["numeric_columns"] if column in features]
    categorical = [column for column in metadata["categorical_columns"] if column in features]
    return numeric, categorical


def _build_pipeline(
    model_name: str,
    actual_model_name: str,
    classifier: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
    preprocessing_config: dict[str, Any],
) -> object:
    if model_name == TREE_MODEL_ALIAS or actual_model_name in {
        "lightgbm",
        "hist_gradient_boosting",
        "random_forest",
    }:
        preprocessor = build_tree_preprocessor(numeric_columns, categorical_columns, preprocessing_config)
    else:
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, preprocessing_config)
    return build_training_pipeline(preprocessor, classifier)


def _with_run_fields(
    record: dict[str, Any],
    run_name: str,
    threshold: float,
    random_seed: int,
) -> dict[str, Any]:
    """Add run-level audit fields to a result record."""

    return {
        "run_name": run_name,
        "threshold": threshold,
        "random_seed": random_seed,
        **record,
    }


def _fit_and_evaluate_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    target_column: str,
    feature_group: str,
    model_name: str,
    actual_model_name: str,
    classifier: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: dict[str, Any],
    run_name: str,
    fallback_reason: str | None = None,
) -> tuple[dict[str, Any], object, Any]:
    pipeline = _build_pipeline(
        model_name=model_name,
        actual_model_name=actual_model_name,
        classifier=classifier,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocessing_config=config["preprocessing"],
    )
    X_train, y_train = split_features_target(train_df, features, target_column)
    trained_pipeline, training_time = measure_training_time(pipeline, X_train, y_train)
    X_valid, y_valid = split_features_target(valid_df, features, target_column)
    threshold = float(config.get("default_threshold", 0.5))
    record, y_score = evaluate_model(
        trained_pipeline,
        X_valid,
        y_valid,
        threshold=threshold,
        model_name=model_name,
        feature_group=feature_group,
        split_name="valid",
        training_time=training_time,
    )
    record.update(
        {
            "actual_model_name": actual_model_name,
            "fallback_reason": fallback_reason or "",
            "feature_count": len(features),
        }
    )
    record = _with_run_fields(record, run_name, threshold, int(config["random_seed"]))
    return record, trained_pipeline, y_score


def _build_model_for_stage2(
    requested_model_name: str,
    config: dict[str, Any],
) -> tuple[str, object, str | None]:
    random_seed = int(config["random_seed"])
    if requested_model_name == TREE_MODEL_ALIAS:
        return resolve_tree_model(config, random_seed=random_seed)
    estimator = build_model(
        requested_model_name,
        random_seed=random_seed,
        params=_model_params(config, requested_model_name),
    )
    return requested_model_name, estimator, None


def run_model_comparison(
    config: dict[str, Any],
    data_dir: Path,
    metadata_path: Path,
    feature_groups_path: Path,
    output_dir: Path,
    run_name: str,
) -> pd.DataFrame:
    """Compare dummy, logistic regression, and the resolved tree model on transaction_basic."""

    splits = read_formal_splits(data_dir)
    metadata = load_metadata(metadata_path)
    feature_groups = load_feature_groups(feature_groups_path)
    target = metadata.get("target_column", "isFraud")
    id_column = metadata.get("id_column", "TransactionID")
    time_column = metadata.get("time_column", "TransactionDT")
    features = validate_requested_features(
        splits["train"],
        feature_groups,
        "transaction_basic",
        target,
        id_column,
        time_column,
    )
    numeric, categorical = _columns_for_group(metadata, features)
    records: list[dict[str, Any]] = []
    for model_name in ("dummy", "logistic_regression", TREE_MODEL_ALIAS):
        actual_model_name, estimator, fallback_reason = _build_model_for_stage2(model_name, config)
        record, _, _ = _fit_and_evaluate_valid(
            splits["train"],
            splits["valid"],
            features,
            target,
            "transaction_basic",
            model_name,
            actual_model_name,
            estimator,
            numeric,
            categorical,
            config,
            run_name=run_name,
            fallback_reason=fallback_reason,
        )
        records.append(record)

    table_path = output_dir / "tables" / f"{run_name}_model_comparison_valid.csv"
    save_results_table(records, table_path)
    _plot_metric_bar(
        pd.DataFrame(records),
        x_column="model_name",
        y_column="pr_auc",
        output_path=output_dir / "figures" / f"{run_name}_model_comparison_pr_auc.png",
        title="Stage 2 Model Comparison PR-AUC",
    )
    return pd.DataFrame(records)


def _select_best_feature_group(ablation: pd.DataFrame, tolerance: float = 0.002) -> tuple[str, str]:
    best_pr_auc = float(ablation["pr_auc"].max())
    contenders = ablation.loc[ablation["pr_auc"] >= best_pr_auc - tolerance].copy()
    if len(contenders) == 1:
        row = contenders.iloc[0]
        return str(row["feature_group"]), "highest valid PR-AUC"
    max_recall = contenders["recall"].max()
    recall_contenders = contenders.loc[contenders["recall"] == max_recall].copy()
    if len(recall_contenders) == 1:
        row = recall_contenders.iloc[0]
        return str(row["feature_group"]), "valid PR-AUC within 0.002; higher valid recall"
    row = recall_contenders.sort_values(["feature_count", "feature_group"], ascending=[True, True]).iloc[0]
    return str(row["feature_group"]), "valid PR-AUC within 0.002 and recall tied; fewer features"


def run_ablation_and_final_test(
    config: dict[str, Any],
    data_dir: Path,
    metadata_path: Path,
    feature_groups_path: Path,
    output_dir: Path,
    run_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Run feature-group ablation on valid and one final test evaluation."""

    splits = read_formal_splits(data_dir)
    metadata = load_metadata(metadata_path)
    feature_groups = load_feature_groups(feature_groups_path)
    target = metadata.get("target_column", "isFraud")
    id_column = metadata.get("id_column", "TransactionID")
    time_column = metadata.get("time_column", "TransactionDT")
    actual_tree_name, _, tree_fallback_reason = _build_model_for_stage2(TREE_MODEL_ALIAS, config)

    ablation_records: list[dict[str, Any]] = []
    for group_name in ("transaction_basic", "transaction_identity", "transaction_identity_missing"):
        features = validate_requested_features(
            splits["train"],
            feature_groups,
            group_name,
            target,
            id_column,
            time_column,
        )
        numeric, categorical = _columns_for_group(metadata, features)
        _, estimator, _ = _build_model_for_stage2(TREE_MODEL_ALIAS, config)
        record, _, _ = _fit_and_evaluate_valid(
            splits["train"],
            splits["valid"],
            features,
            target,
            group_name,
            TREE_MODEL_ALIAS,
            actual_tree_name,
            estimator,
            numeric,
            categorical,
            config,
            run_name=run_name,
            fallback_reason=tree_fallback_reason,
        )
        ablation_records.append(record)

    ablation = pd.DataFrame(ablation_records)
    ablation_path = output_dir / "tables" / f"{run_name}_ablation_valid.csv"
    save_results_table(ablation_records, ablation_path)
    _plot_metric_bar(
        ablation,
        x_column="feature_group",
        y_column="pr_auc",
        output_path=output_dir / "figures" / f"{run_name}_ablation_pr_auc.png",
        title="Stage 2 Feature Ablation PR-AUC",
    )
    _plot_metric_bar(
        ablation,
        x_column="feature_group",
        y_column="recall",
        output_path=output_dir / "figures" / f"{run_name}_ablation_recall.png",
        title="Stage 2 Feature Ablation Recall",
    )

    selected_group, selection_reason = _select_best_feature_group(ablation)
    selection_details = build_selection_details(ablation, selected_group, selection_reason)
    final_record, final_scores, final_pipeline = _fit_final_model(
        config,
        splits,
        metadata,
        feature_groups,
        selected_group,
        actual_tree_name,
        tree_fallback_reason,
        run_name=run_name,
    )
    final_record.update(selection_details)
    final_test = pd.DataFrame([final_record])
    save_results_table([final_record], output_dir / "tables" / f"{run_name}_final_test.csv")

    _, y_test = split_features_target(
        splits["test"],
        validate_requested_features(splits["test"], feature_groups, selected_group, target, id_column, time_column),
        target,
    )
    plot_precision_recall_curve(
        y_test,
        final_scores,
        output_dir / "figures" / f"{run_name}_best_model_pr_curve_test.png",
        title=f"Stage 2 Best Model PR Curve ({selected_group})",
    )
    plot_confusion_matrix(
        int(final_record["tn"]),
        int(final_record["fp"]),
        int(final_record["fn"]),
        int(final_record["tp"]),
        output_dir / "figures" / f"{run_name}_best_model_confusion_matrix_test.png",
        title=f"Stage 2 Best Model Confusion Matrix ({selected_group})",
    )
    importance_path = output_dir / "figures" / f"{run_name}_feature_importance_top20.png"
    if plot_tree_feature_importance_top20(final_pipeline, importance_path) is None:
        LOGGER.warning("Skipped feature importance plot because names/importances could not be mapped reliably.")
    return ablation, final_test, selected_group, selection_reason


def build_selection_details(
    ablation: pd.DataFrame,
    selected_group: str,
    selection_reason: str,
) -> dict[str, Any]:
    """Build final-test selection audit fields from validation ablation results only."""

    selected_row = ablation.loc[ablation["feature_group"] == selected_group].iloc[0]
    runner_up = (
        ablation.loc[ablation["feature_group"] != selected_group]
        .sort_values(["pr_auc", "recall", "feature_count"], ascending=[False, False, True])
        .iloc[0]
    )
    selected_score = float(selected_row["pr_auc"])
    runner_up_score = float(runner_up["pr_auc"])
    return {
        "selected_feature_group": selected_group,
        "selection_metric": "valid PR-AUC",
        "validation_selection_score": selected_score,
        "runner_up_feature_group": str(runner_up["feature_group"]),
        "runner_up_validation_score": runner_up_score,
        "score_difference": selected_score - runner_up_score,
        "selection_reason": selection_reason,
    }


def _fit_final_model(
    config: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    feature_groups: dict[str, list[str]],
    selected_group: str,
    actual_tree_name: str,
    fallback_reason: str | None,
    run_name: str,
) -> tuple[dict[str, Any], Any, object]:
    target = metadata.get("target_column", "isFraud")
    id_column = metadata.get("id_column", "TransactionID")
    time_column = metadata.get("time_column", "TransactionDT")
    train_valid = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    features = validate_requested_features(train_valid, feature_groups, selected_group, target, id_column, time_column)
    numeric, categorical = _columns_for_group(metadata, features)
    _, estimator, _ = _build_model_for_stage2(TREE_MODEL_ALIAS, config)
    pipeline = _build_pipeline(
        model_name=TREE_MODEL_ALIAS,
        actual_model_name=actual_tree_name,
        classifier=estimator,
        numeric_columns=numeric,
        categorical_columns=categorical,
        preprocessing_config=config["preprocessing"],
    )
    X_train_valid, y_train_valid = split_features_target(train_valid, features, target)
    trained_pipeline, training_time = measure_training_time(pipeline, X_train_valid, y_train_valid)
    X_test, y_test = split_features_target(splits["test"], features, target)
    threshold = float(config.get("default_threshold", 0.5))
    record, y_score = evaluate_model(
        trained_pipeline,
        X_test,
        y_test,
        threshold=threshold,
        model_name=TREE_MODEL_ALIAS,
        feature_group=selected_group,
        split_name="test",
        training_time=training_time,
    )
    record.update(
        {
            "actual_model_name": actual_tree_name,
            "fallback_reason": fallback_reason or "",
            "feature_count": len(features),
            "train_valid_rows": len(train_valid),
            "test_rows": len(splits["test"]),
        }
    )
    record = _with_run_fields(record, run_name, threshold, int(config["random_seed"]))
    return record, y_score, trained_pipeline


def extract_tree_feature_importance(pipeline: object) -> pd.DataFrame | None:
    """Return mapped tree feature importances, or None when mapping is unsafe."""

    if not hasattr(pipeline, "named_steps"):
        return None
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    if preprocessor is None or classifier is None or not hasattr(classifier, "feature_importances_"):
        return None
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return None
    importances = np.asarray(classifier.feature_importances_)
    if len(feature_names) != len(importances):
        return None
    cleaned_names = [name.split("__", 1)[1] if "__" in name else name for name in feature_names]
    return (
        pd.DataFrame({"feature": cleaned_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_tree_feature_importance_top20(pipeline: object, output_path: Path) -> Path | None:
    """Plot top-20 tree feature importances when feature-name mapping is reliable."""

    importance = extract_tree_feature_importance(pipeline)
    if importance is None:
        return None
    top = importance.head(20).iloc[::-1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature"], top["importance"])
    axis.set_xlabel("Feature importance")
    axis.set_ylabel("Feature")
    axis.set_title("Stage 2 LightGBM Feature Importance Top 20")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _plot_metric_bar(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(8, 4))
    axis.bar(table[x_column].astype(str), table[y_column].astype(float))
    axis.set_ylabel(y_column)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
