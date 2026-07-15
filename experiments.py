"""Run pyfixest estimations inside an MLflow-tracked experiment."""

from __future__ import annotations

from typing import Any, Callable

import mlflow
import pandas as pd

_METRIC_ATTRS = {
    "f_statistic": "_f_statistic",
    "r2": "_r2",
    "adj_r2": "_adj_r2",
    "rmse": "_rmse",
    "nobs": "_N",
}


def run_experiment(
    model_fn: Callable[..., Any],
    *args: Any,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call a pyfixest modeling function inside a tracked MLflow run.

    ``model_fn`` (e.g. ``pyfixest.feols``, ``pyfixest.fepois``, ``pyfixest.feiv``) is
    called as ``model_fn(*args, **kwargs)``. The F-statistic, R2, adjusted R2, RMSE,
    number of observations, and the coefficient table are logged to MLflow for the
    resulting model(s). The object returned by ``model_fn`` is returned unchanged.
    """
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_param("model_fn", getattr(model_fn, "__name__", str(model_fn)))
        for i, value in enumerate(args):
            _log_param(f"arg_{i}", value)
        for key, value in kwargs.items():
            _log_param(key, value)

        result = model_fn(*args, **kwargs)

        fits = result.to_list() if hasattr(result, "to_list") else [result]
        multiple = len(fits) > 1
        for i, fit in enumerate(fits):
            _log_fit(fit, prefix=f"model{i}_" if multiple else "")

    return result


def _log_param(key: str, value: Any) -> None:
    if isinstance(value, pd.DataFrame):
        mlflow.log_param(f"{key}_shape", str(value.shape))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        mlflow.log_param(key, value)
    else:
        mlflow.log_param(key, repr(value))


def _log_fit(fit: Any, prefix: str) -> None:
    for metric_name, attr in _METRIC_ATTRS.items():
        value = getattr(fit, attr, None)
        if value is not None:
            mlflow.log_metric(f"{prefix}{metric_name}", float(value))

    coef_table = fit.tidy().reset_index()
    mlflow.log_table(coef_table, artifact_file=f"{prefix}coefficients.json")
