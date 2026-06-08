"""Command-line entry point for stage 0 checks and stage 1 smoke experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    save_results_table,
)
from src.metrics import validate_metric_names
from src.models import build_model, build_training_pipeline, get_available_models, is_lightgbm_available
from src.preprocessing import (
    build_preprocessor,
    load_feature_groups,
    load_metadata,
    split_features_target,
    validate_requested_features,
)
from src.train import measure_training_time


REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "random_seed",
    "models",
    "primary_metric",
    "metrics",
    "positive_label",
    "feature_groups",
    "fallback_model",
    "preprocessing",
    "default_threshold",
)

TRAINING_OPTION_FLAGS: frozenset[str] = frozenset(
    {
        "--data-dir",
        "--metadata",
        "--feature-groups",
        "--feature-group",
        "--models",
        "--output-dir",
        "--run-name",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run stage 0 config checks or stage 1 model smoke experiments."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.json"))
    parser.add_argument(
        "--feature-groups",
        type=Path,
        default=Path("reports/tables/feature_groups.json"),
    )
    parser.add_argument("--feature-group", default="transaction_basic")
    parser.add_argument("--models", nargs="+", default=["dummy", "logistic_regression"])
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--run-name", default="stage1_smoke")
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return config


def _configured_model_names(models_config: Any) -> list[str]:
    """Return model names from stage 0 list or stage 1 mapping config."""

    if isinstance(models_config, dict):
        return list(models_config.keys())
    if isinstance(models_config, list) and all(isinstance(name, str) for name in models_config):
        return list(models_config)
    raise ValueError("models must be a list of names or a mapping of model names to params.")


def _configured_metrics(config: dict[str, Any]) -> list[str]:
    """Return configured metrics while preserving stage 0 compatibility."""

    if "metrics" in config:
        metrics = config["metrics"]
    else:
        metrics = [config["primary_metric"], *config.get("additional_metrics", [])]
    if not isinstance(metrics, list) or not all(isinstance(name, str) for name in metrics):
        raise ValueError("metrics must be a list of metric names.")
    return metrics


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration without loading data or fitting models."""

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config key(s): {', '.join(missing_keys)}.")

    if config["random_seed"] != 42:
        raise ValueError("random_seed must be 42 for this project.")
    if config["positive_label"] != 1:
        raise ValueError("positive_label must be 1 for isFraud.")

    registered_models = set(get_available_models())
    configured_models = _configured_model_names(config["models"])
    unsupported_models = sorted(set(configured_models) - registered_models)
    if unsupported_models:
        supported = ", ".join(sorted(registered_models))
        invalid = ", ".join(unsupported_models)
        raise ValueError(f"Unsupported model name(s): {invalid}. Supported models: {supported}.")

    fallback_model = config["fallback_model"]
    if not isinstance(fallback_model, str) or fallback_model not in registered_models:
        supported = ", ".join(sorted(registered_models))
        raise ValueError(f"fallback_model must be one of: {supported}.")

    feature_groups = config["feature_groups"]
    if not isinstance(feature_groups, list) or not all(
        isinstance(name, str) for name in feature_groups
    ):
        raise ValueError("feature_groups must be a list of feature group names.")

    validate_metric_names(_configured_metrics(config))


def print_registered_models() -> None:
    """Print model registry details."""

    print("Registered models:")
    for model_name in get_available_models():
        print(f"- {model_name}")
    print(f"LightGBM available: {is_lightgbm_available()}")


def _training_mode_requested(raw_args: list[str]) -> bool:
    """Return whether the user explicitly requested stage 1 training mode."""

    return any(arg in TRAINING_OPTION_FLAGS for arg in raw_args)


def _require_file(path: Path, description: str) -> None:
    """Raise a clear error if a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def _read_split(data_dir: Path, split_name: str) -> pd.DataFrame:
    """Read one processed parquet split."""

    split_path = data_dir / f"{split_name}.parquet"
    _require_file(split_path, f"{split_name} parquet file")
    return pd.read_parquet(split_path)


def _model_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Get estimator params for one configured model."""

    models_config = config["models"]
    if not isinstance(models_config, dict):
        return {}
    params = dict(models_config.get(model_name, {}) or {})
    params.pop("enabled", None)
    return params


def _result_table_path(output_dir: Path, run_name: str) -> Path:
    """Return the result table path for a run."""

    filename = "smoke_model_results.csv" if run_name == "stage1_smoke" else f"{run_name}_model_results.csv"
    return output_dir / "tables" / filename


def _figure_paths(output_dir: Path, run_name: str) -> tuple[Path, Path]:
    """Return PR curve and confusion matrix output paths."""

    if run_name == "stage1_smoke":
        return (
            output_dir / "figures" / "smoke_pr_curve.png",
            output_dir / "figures" / "smoke_confusion_matrix_logistic.png",
        )
    return (
        output_dir / "figures" / f"{run_name}_pr_curve.png",
        output_dir / "figures" / f"{run_name}_confusion_matrix_logistic.png",
    )


def _configure_logging(run_name: str) -> None:
    """Configure file logging for a stage 1 run."""

    log_path = Path("logs") / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def run_stage1(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Run the stage 1 model smoke experiment on existing processed files."""

    _configure_logging(args.run_name)
    _require_file(args.metadata, "metadata JSON")
    _require_file(args.feature_groups, "feature groups JSON")

    train_df = _read_split(args.data_dir, "train")
    valid_df = _read_split(args.data_dir, "valid")
    test_df = _read_split(args.data_dir, "test")
    metadata = load_metadata(args.metadata)
    feature_groups = load_feature_groups(args.feature_groups)

    target_column = metadata.get("target_column", "isFraud")
    id_column = metadata.get("id_column", "TransactionID")
    time_column = metadata.get("time_column", "TransactionDT")

    features = validate_requested_features(
        train_df,
        feature_groups,
        args.feature_group,
        target_column,
        id_column,
        time_column,
    )
    for split_name, dataframe in {"valid": valid_df, "test": test_df}.items():
        validate_requested_features(
            dataframe,
            feature_groups,
            args.feature_group,
            target_column,
            id_column,
            time_column,
        )

    numeric_columns = [column for column in metadata["numeric_columns"] if column in features]
    categorical_columns = [column for column in metadata["categorical_columns"] if column in features]
    threshold = float(config.get("default_threshold", 0.5))
    random_seed = int(config["random_seed"])

    records: list[dict[str, Any]] = []
    logistic_test_record: dict[str, Any] | None = None
    logistic_test_score = None
    _, y_test = split_features_target(test_df, features, target_column)

    for model_name in args.models:
        if model_name not in {"dummy", "logistic_regression"}:
            raise ValueError("Stage 1 smoke training supports only dummy and logistic_regression.")

        classifier = build_model(model_name, random_seed, _model_params(config, model_name))
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, config["preprocessing"])
        pipeline = build_training_pipeline(preprocessor, classifier)

        X_train, y_train = split_features_target(train_df, features, target_column)
        trained_pipeline, training_time = measure_training_time(pipeline, X_train, y_train)

        for split_name, dataframe in {"valid": valid_df, "test": test_df}.items():
            X_split, y_split = split_features_target(dataframe, features, target_column)
            record, y_score = evaluate_model(
                trained_pipeline,
                X_split,
                y_split,
                threshold=threshold,
                model_name=model_name,
                feature_group=args.feature_group,
                split_name=split_name,
                training_time=training_time,
            )
            records.append(record)
            if model_name == "logistic_regression" and split_name == "test":
                logistic_test_record = record
                logistic_test_score = y_score

    result_path = save_results_table(records, _result_table_path(args.output_dir, args.run_name))
    pr_curve_path = None
    confusion_path = None
    if logistic_test_record is not None and logistic_test_score is not None:
        pr_curve_path, confusion_path = _figure_paths(args.output_dir, args.run_name)
        plot_precision_recall_curve(
            y_test,
            logistic_test_score,
            pr_curve_path,
            title="Logistic Regression Precision-Recall",
        )
        plot_confusion_matrix(
            int(logistic_test_record["tn"]),
            int(logistic_test_record["fp"]),
            int(logistic_test_record["fn"]),
            int(logistic_test_record["tp"]),
            confusion_path,
            title="Logistic Regression Confusion Matrix",
        )

    print(f"Stage 1 smoke run complete: {args.run_name}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Feature group: {args.feature_group}")
    print(f"Results: {result_path}")
    if pr_curve_path and confusion_path:
        print(f"PR curve: {pr_curve_path}")
        print(f"Confusion matrix: {confusion_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the experiment CLI."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_args)

    try:
        config = load_config(args.config)
        validate_config(config)

        if args.list_models:
            print_registered_models()

        if args.dry_run:
            print("Dry run passed: configuration structure, models, and metrics are valid.")

        if _training_mode_requested(raw_args):
            return run_stage1(args, config)

        if not args.list_models and not args.dry_run:
            print("No action requested. Use --help for available commands.")

    except (FileNotFoundError, ValueError, ImportError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
