"""Model registry and sklearn pipeline builders."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


AVAILABLE_MODELS: tuple[str, ...] = (
    "dummy",
    "logistic_regression",
    "lightgbm",
    "hist_gradient_boosting",
    "random_forest",
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


def build_hist_gradient_boosting_model(
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> HistGradientBoostingClassifier:
    """Build an untrained ``HistGradientBoostingClassifier``."""

    model_params = dict(params or {})
    model_params.setdefault("random_state", random_seed)
    return HistGradientBoostingClassifier(**model_params)


def build_random_forest_model(
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Build an untrained ``RandomForestClassifier``."""

    model_params = dict(params or {})
    model_params.setdefault("random_state", random_seed)
    return RandomForestClassifier(**model_params)


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
        return build_hist_gradient_boosting_model(random_seed=random_seed, params=params)

    if model_name == "random_forest":
        return build_random_forest_model(random_seed=random_seed, params=params)

    supported = ", ".join(AVAILABLE_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")


def resolve_tree_model(
    config: dict[str, Any],
    random_seed: int,
) -> tuple[str, object, str | None]:
    """Resolve LightGBM with configured fallbacks and return actual name, estimator, and reason."""

    models_config = config.get("models", {})
    fallback_order = config.get(
        "tree_model_fallback_order",
        ["lightgbm", "hist_gradient_boosting", "random_forest"],
    )
    failure_reasons: list[str] = []
    for model_name in fallback_order:
        params = dict(models_config.get(model_name, {}) or {})
        enabled = bool(params.pop("enabled", True))
        if not enabled:
            failure_reasons.append(f"{model_name} disabled in config")
            continue
        try:
            estimator = build_model(model_name, random_seed=random_seed, params=params)
            reason = "; ".join(failure_reasons) if failure_reasons else None
            return model_name, estimator, reason
        except Exception as exc:
            failure_reasons.append(f"{model_name} unavailable: {exc}")
    raise ImportError("No configured tree model is available. " + "; ".join(failure_reasons))
