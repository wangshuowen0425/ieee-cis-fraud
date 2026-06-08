"""Model registry for the stage 0 experiment entry point.

This module only builds untrained estimator instances. It does not read data,
fit models, or silently replace one registered model with another.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


AVAILABLE_MODELS: tuple[str, ...] = (
    "dummy",
    "logistic_regression",
    "lightgbm",
    "hist_gradient_boosting",
)


def get_available_models() -> tuple[str, ...]:
    """Return the names registered in the local model factory."""

    return AVAILABLE_MODELS


def is_lightgbm_available() -> bool:
    """Return whether the optional LightGBM package can be imported."""

    return find_spec("lightgbm") is not None


def build_model(
    model_name: str,
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> object:
    """Create an untrained model instance from the registry.

    Parameters
    ----------
    model_name:
        Registered model name.
    random_seed:
        Deterministic seed passed to estimators that support it.
    params:
        Optional estimator parameters from configuration.

    Raises
    ------
    ValueError
        If the model name is not registered.
    ImportError
        If ``lightgbm`` is requested but the optional dependency is missing.
    """

    model_params = dict(params or {})

    if model_name == "dummy":
        return DummyClassifier(**model_params)

    if model_name == "logistic_regression":
        model_params.setdefault("random_state", random_seed)
        return LogisticRegression(**model_params)

    if model_name == "lightgbm":
        if not is_lightgbm_available():
            raise ImportError(
                "LightGBM is not installed. Install requirements-optional.txt "
                "or choose the configured fallback model explicitly."
            )
        from lightgbm import LGBMClassifier

        model_params.setdefault("random_state", random_seed)
        return LGBMClassifier(**model_params)

    if model_name == "hist_gradient_boosting":
        model_params.setdefault("random_state", random_seed)
        return HistGradientBoostingClassifier(**model_params)

    supported = ", ".join(AVAILABLE_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")
