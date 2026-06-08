"""Model registry and sklearn pipeline builders."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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


def build_dummy_model(params: dict[str, Any] | None = None) -> DummyClassifier:
    """Build an untrained ``DummyClassifier``."""

    model_params = {"strategy": "prior"}
    model_params.update(params or {})
    return DummyClassifier(**model_params)


def build_logistic_regression_model(
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> LogisticRegression:
    """Build an untrained ``LogisticRegression`` model."""

    model_params: dict[str, Any] = {
        "class_weight": "balanced",
        "solver": "saga",
        "max_iter": 300,
        "random_state": random_seed,
        "n_jobs": -1,
    }
    model_params.update(params or {})
    model_params.setdefault("random_state", random_seed)
    return LogisticRegression(**model_params)


def build_training_pipeline(preprocessor: object, classifier: object) -> Pipeline:
    """Build a sklearn training pipeline from preprocessor and classifier."""

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_model(
    model_name: str,
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> object:
    """Create an untrained model instance from the registry."""

    if model_name == "dummy":
        return build_dummy_model(params)

    if model_name == "logistic_regression":
        return build_logistic_regression_model(random_seed=random_seed, params=params)

    if model_name == "lightgbm":
        if not is_lightgbm_available():
            raise ImportError(
                "LightGBM is not installed. Install requirements-optional.txt "
                "or choose the configured fallback model explicitly."
            )
        from lightgbm import LGBMClassifier

        model_params = dict(params or {})
        model_params.setdefault("random_state", random_seed)
        return LGBMClassifier(**model_params)

    if model_name == "hist_gradient_boosting":
        model_params = dict(params or {})
        model_params.setdefault("random_state", random_seed)
        return HistGradientBoostingClassifier(**model_params)

    supported = ", ".join(AVAILABLE_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")
