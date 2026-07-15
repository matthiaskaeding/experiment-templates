"""Run a single pyfixest estimation inside an MLflow-tracked experiment."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import mlflow
import pandas as pd
import pyfixest as pf
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.parse import Formula

_MULTI_MODEL_ERROR = (
    "run_experiment only supports single-model results; the formula produced "
    "multiple models (e.g. via sw()/csw() or multiple dependent variables)."
)


def _extract_metrics(fit: Any, model_fn: Callable[..., Any]) -> dict[str, float]:
    """Read the metrics relevant to model_fn off fit via direct access.

    Metrics are pyfixest internals without a stable public getter, and not every
    attribute applies to every model type (e.g. fepois has no F-statistic), so a
    missing attribute is skipped rather than treated as an error.
    """
    if model_fn is pf.fepois:
        attrs = (
            ("nobs", "_N"),
            ("pseudo_r2", "_pseudo_r2"),
            ("deviance", "deviance"),
        )
    else:
        attrs = (
            ("nobs", "_N"),
            ("r2", "_r2"),
            ("adj_r2", "_adj_r2"),
            ("f_statistic", "_f_statistic"),
            ("rmse", "_rmse"),
        )

    metrics = {}
    for name, attr in attrs:
        try:
            metrics[name] = float(getattr(fit, attr))
        except AttributeError:
            pass
    return metrics


def _bind_args(
    model_fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Map positional/keyword call args to model_fn's parameter names."""
    bound = inspect.signature(model_fn).bind_partial(*args, **kwargs)
    return bound.arguments


def run_experiment(
    *args: Any,
    model_fn: Callable[..., Any] | str = pf.feols,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call a pyfixest modeling function inside a tracked MLflow run.

    ``model_fn`` (default ``pyfixest.feols``) is either a pyfixest modeling function
    (e.g. ``pyfixest.fepois``) or its name as a string (e.g. ``"fepois"``), resolved
    via ``getattr(pyfixest, model_fn)``. It is called as ``model_fn(*args, **kwargs)``.

    Only single-model results are supported: formulas that produce several models
    (e.g. via ``sw()``/``csw()`` or multiple dependent variables) raise a
    ``ValueError``. This is checked upfront by parsing the formula, before
    ``model_fn`` runs, and again on the returned object as a backstop.

    Which metrics get logged depends on the model type (e.g. ``fepois`` has no R2):
    ``_extract_metrics`` picks the relevant (metric_name, attribute) pairs based on
    ``model_fn``. Metrics are logged to MLflow together with the coefficient table.
    The object returned by ``model_fn`` is returned unchanged.

    Only key parameters are logged: the formula, the data's shape, and vcov.
    """
    model_fn = _resolve_model_fn(model_fn)

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_param("model_fn", getattr(model_fn, "__name__", str(model_fn)))

        bound_args = _bind_args(model_fn, args, kwargs)

        fml = bound_args.get("fml")
        if fml is not None:
            mlflow.log_param("fml", fml)
            if len(Formula.parse(fml)) > 1:
                raise ValueError(_MULTI_MODEL_ERROR)

        data = bound_args.get("data")
        if isinstance(data, pd.DataFrame):
            mlflow.log_param("data_shape", str(data.shape))

        vcov = bound_args.get("vcov")
        if vcov is not None:
            mlflow.log_param("vcov", str(vcov))

        fit = model_fn(*args, **kwargs)

        if isinstance(fit, FixestMulti):
            raise ValueError(_MULTI_MODEL_ERROR)

        mlflow.log_metrics(_extract_metrics(fit, model_fn))

        coef_table = fit.tidy().reset_index()
        mlflow.log_table(coef_table, artifact_file="coefficients.json")

    return fit


def _resolve_model_fn(model_fn: Callable[..., Any] | str) -> Callable[..., Any]:
    if not isinstance(model_fn, str):
        return model_fn
    resolved = getattr(pf, model_fn, None)
    if not callable(resolved):
        raise ValueError(f"Unknown pyfixest model function: {model_fn!r}")
    return resolved
