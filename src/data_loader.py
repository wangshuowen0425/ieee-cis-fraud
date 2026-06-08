"""Raw data entry checks for the IEEE-CIS fraud project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and attach its project root."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    config["_config_path"] = path
    config["_project_root"] = path.parent.parent
    LOGGER.info("Loaded data config from %s", path)
    return config


def resolve_raw_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve raw transaction and identity CSV paths from configuration."""
    project_root = Path(config.get("_project_root", Path.cwd())).resolve()
    raw_dir = Path(config["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    return {
        "transaction": raw_dir / str(config["transaction_file"]),
        "identity": raw_dir / str(config["identity_file"]),
    }


def validate_raw_files(config: dict[str, Any]) -> dict[str, Path]:
    """Validate raw CSV files exist, are regular files, and are non-empty."""
    paths = resolve_raw_paths(config)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} raw file: {path}")
        if not path.is_file():
            raise ValueError(f"{name} raw path is not a regular file: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"{name} raw file is empty: {path}")
        LOGGER.info("%s raw file is accessible: %s", name, path)
    return paths


def inspect_csv_schema(path: str | Path, nrows: int = 5) -> dict[str, str]:
    """Inspect CSV columns and inferred dtypes using only a small row sample."""
    csv_path = Path(path)
    sample = pd.read_csv(csv_path, nrows=nrows)
    schema = {column: str(dtype) for column, dtype in sample.dtypes.items()}
    LOGGER.info("Inspected %s columns from %s using nrows=%s", len(schema), csv_path, nrows)
    return schema


def select_existing_columns(
    available_columns: list[str] | set[str],
    candidate_columns: list[str],
    context: str = "columns",
) -> tuple[list[str], list[str]]:
    """Return existing and missing candidate columns while preserving candidate order."""
    available = set(available_columns)
    existing = [column for column in candidate_columns if column in available]
    missing = [column for column in candidate_columns if column not in available]
    if missing:
        LOGGER.warning("Missing %s columns skipped: %s", context, missing)
    return existing, missing


def _dedupe_preserve_order(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for column in columns:
        if column not in seen:
            unique.append(column)
            seen.add(column)
    return unique


def load_selected_transaction_data(
    config: dict[str, Any],
    selected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load only required transaction columns and selected available feature columns."""
    paths = validate_raw_files(config)
    transaction_path = paths["transaction"]
    schema = inspect_csv_schema(transaction_path)
    required = [config["id_column"], config["target_column"], config["time_column"]]
    candidates = selected_columns if selected_columns is not None else list(config.get("transaction_columns", []))
    existing_features, missing_features = select_existing_columns(
        list(schema),
        candidates,
        context="transaction feature",
    )
    missing_required = [column for column in required if column not in schema]
    if missing_required:
        raise ValueError(f"Missing required transaction columns: {missing_required}")

    usecols = _dedupe_preserve_order(required + existing_features)
    LOGGER.info("Loading transaction data with %s selected columns", len(usecols))
    return pd.read_csv(transaction_path, usecols=usecols), missing_features


def load_selected_identity_data(
    config: dict[str, Any],
    selected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load only TransactionID and selected available identity columns."""
    paths = validate_raw_files(config)
    identity_path = paths["identity"]
    schema = inspect_csv_schema(identity_path)
    required = [config["id_column"]]
    candidates = selected_columns if selected_columns is not None else list(config.get("identity_columns", []))
    existing_features, missing_features = select_existing_columns(
        list(schema),
        candidates,
        context="identity feature",
    )
    missing_required = [column for column in required if column not in schema]
    if missing_required:
        raise ValueError(f"Missing required identity columns: {missing_required}")

    usecols = _dedupe_preserve_order(required + existing_features)
    LOGGER.info("Loading identity data with %s selected columns", len(usecols))
    return pd.read_csv(identity_path, usecols=usecols), missing_features


def describe_raw_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return raw file names, existence flags, and sizes without reading CSVs."""
    descriptions: list[dict[str, Any]] = []
    for name, path in resolve_raw_paths(config).items():
        exists = path.exists()
        descriptions.append(
            {
                "key": name,
                "filename": path.name,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else None,
            }
        )
    return descriptions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check IEEE-CIS raw data files.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate raw file availability without reading CSV contents.",
    )
    return parser


def main() -> int:
    """Run the raw data file checker CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()

    if not args.check_only:
        LOGGER.error("Only --check-only mode is supported in stage 0.")
        return 2

    try:
        config = load_yaml_config(args.config)
        validate_raw_files(config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("Raw data check failed: %s", exc)
        for item in describe_raw_files(config) if "config" in locals() else []:
            LOGGER.error(
                "%s: filename=%s exists=%s size_bytes=%s",
                item["key"],
                item["filename"],
                item["exists"],
                item["size_bytes"],
            )
        return 1

    for item in describe_raw_files(config):
        LOGGER.info(
            "%s: filename=%s exists=%s size_bytes=%s",
            item["key"],
            item["filename"],
            item["exists"],
            item["size_bytes"],
        )
    LOGGER.info("Raw data check passed: transaction and identity files are accessible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
