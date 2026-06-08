"""CLI tests for the stage 0 experiment entry point."""

from pathlib import Path
from subprocess import run
import sys

import yaml


BASE_COMMAND: tuple[str, ...] = (sys.executable, "-m", "src.run_experiments")


def test_help_succeeds() -> None:
    """The CLI help command should run without project data."""

    result = run([*BASE_COMMAND, "--help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "--dry-run" in result.stdout


def test_list_models_succeeds() -> None:
    """Listing registered models should run without project data."""

    result = run(
        [*BASE_COMMAND, "--config", "configs/model_config.yaml", "--list-models"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dummy" in result.stdout
    assert "LightGBM available:" in result.stdout


def test_dry_run_valid_config_succeeds() -> None:
    """Dry-run should validate the checked-in model configuration."""

    result = run(
        [*BASE_COMMAND, "--config", "configs/model_config.yaml", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Dry run passed" in result.stdout


def test_invalid_model_name_fails(tmp_path: Path) -> None:
    """Dry-run should reject an unsupported model name without reading data."""

    with Path("configs/model_config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["models"] = ["dummy", "not_a_model"]

    invalid_config_path = tmp_path / "invalid_model_config.yaml"
    with invalid_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    result = run(
        [*BASE_COMMAND, "--config", str(invalid_config_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unsupported model name" in result.stderr
