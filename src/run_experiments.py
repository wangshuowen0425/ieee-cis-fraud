"""Stage 0 command-line entry point for experiment configuration checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.metrics import validate_metric_names
from src.models import get_available_models, is_lightgbm_available


REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "random_seed",
    "models",
    "primary_metric",
    "additional_metrics",
    "positive_label",
    "feature_groups",
    "fallback_model",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the stage 0 experiment CLI."""

    parser = argparse.ArgumentParser(
        description="Validate stage 0 model experiment configuration without reading data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List registered model names and LightGBM availability.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration structure, model names, and metric names only.",
    )
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


def validate_config(config: dict[str, Any]) -> None:
    """Validate stage 0 model configuration without loading data or fitting models."""

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config key(s): {', '.join(missing_keys)}.")

    if config["random_seed"] != 42:
        raise ValueError("random_seed must be 42 for this project.")

    if config["positive_label"] != 1:
        raise ValueError("positive_label must be 1 for isFraud.")

    models = config["models"]
    if not isinstance(models, list) or not all(isinstance(name, str) for name in models):
        raise ValueError("models must be a list of model names.")

    registered_models = set(get_available_models())
    unsupported_models = sorted(set(models) - registered_models)
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

    primary_metric = config["primary_metric"]
    additional_metrics = config["additional_metrics"]
    if not isinstance(primary_metric, str):
        raise ValueError("primary_metric must be a metric name.")
    if not isinstance(additional_metrics, list) or not all(
        isinstance(name, str) for name in additional_metrics
    ):
        raise ValueError("additional_metrics must be a list of metric names.")

    validate_metric_names([primary_metric, *additional_metrics])


def print_registered_models() -> None:
    """Print model registry details for CLI users."""

    print("Registered models:")
    for model_name in get_available_models():
        print(f"- {model_name}")
    print(f"LightGBM available: {is_lightgbm_available()}")


def main(argv: list[str] | None = None) -> int:
    """Run the stage 0 experiment CLI."""

    args = parse_args(argv)

    try:
        if args.list_models:
            print_registered_models()

        if args.dry_run:
            config = load_config(args.config)
            validate_config(config)
            print("Dry run passed: configuration structure, models, and metrics are valid.")

        if not args.list_models and not args.dry_run:
            print("No action requested. Use --help for available commands.")

    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
