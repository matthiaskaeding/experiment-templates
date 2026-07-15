"""Run a single pyfixest estimation inside an MLflow-tracked experiment."""

from __future__ import annotations

from typing import Any, Callable

import mlflow
import pandas as pd
import pyfixest as pf


def _feols_metrics(fit: Any) -> dict[str, float]:
    metrics = {}
    for name, attr in (
        ("nobs", "_N"),
        ("r2", "_r2"),
        ("adj_r2", "_adj_r2"),
        ("f_statistic", "_f_statistic"),
        ("rmse", "_rmse"),
    ):
        try:
            metrics[name] = float(getattr(fit, attr))
        except AttributeError:
            pass
    return metrics


def _fepois_metrics(fit: Any) -> dict[str, float]:
    metrics = {}
    for name, attr in (
        ("nobs", "_N"),
        ("pseudo_r2", "_pseudo_r2"),
        ("deviance", "deviance"),
    ):
        try:
            metrics[name] = float(getattr(fit, attr))
        except AttributeError:
            pass
    return metrics


_METRIC_FNS: dict[Callable[..., Any], Callable[[Any], dict[str, float]]] = {
    pf.feols: _feols_metrics,
    pf.fepois: _fepois_metrics,
}


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
    ``ValueError``.

    Metrics are looked up via ``_METRIC_FNS[model_fn]``, which picks the metrics
    relevant to that model type (e.g. ``fepois`` has no R2), and are logged to MLflow
    together with the coefficient table. The object returned by ``model_fn`` is
    returned unchanged.
    """
    model_fn = _resolve_model_fn(model_fn)

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_param("model_fn", getattr(model_fn, "__name__", str(model_fn)))
        for i, value in enumerate(args):
            _log_param(f"arg_{i}", value)
        for key, value in kwargs.items():
            _log_param(key, value)

        fit = model_fn(*args, **kwargs)

        if hasattr(fit, "to_list"):
            raise ValueError(
                "run_experiment only supports single-model results; the formula "
                "produced multiple models (e.g. via sw()/csw() or multiple "
                "dependent variables)."
            )

        metrics_fn = _METRIC_FNS.get(model_fn, _feols_metrics)
        mlflow.log_metrics(metrics_fn(fit))

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


def _log_param(key: str, value: Any) -> None:
    if isinstance(value, pd.DataFrame):
        mlflow.log_param(f"{key}_shape", str(value.shape))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        mlflow.log_param(key, value)
    else:
        mlflow.log_param(key, repr(value))
