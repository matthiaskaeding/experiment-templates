"""Run a single pyfixest estimation inside an MLflow-tracked experiment."""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable

import mlflow
import pandas as pd
import pyfixest as pf
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.models.feglm_ import Feglm
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.quantreg.quantreg_ import Quantreg

_MULTI_MODEL_ERROR = (
    "run_experiment only supports single-model results; the formula produced "
    "multiple models (e.g. via sw()/csw() or multiple dependent variables)."
)


def _extract_metrics(fit: Any) -> dict[str, float]:
    """Read the metrics relevant to ``fit`` off it via direct access.

    Dispatch is on the *type* of the fitted result, not on the modeling function,
    so a user-supplied wrapper around a pyfixest estimator still gets the right
    metrics. ``Fepois``, ``Feglm`` (logit/probit), and ``Quantreg`` are all
    subclasses of ``Feols``, so the ``isinstance`` checks run most specific first
    and fall back to the ``Feols``-style metrics.

    Each entry is ``(metric_name, attribute, required)``. These metrics are
    pyfixest internals without a stable public getter, so a *required* attribute
    that is missing (or non-numeric) is skipped with a ``warnings.warn`` -- if
    pyfixest renames one, the warning surfaces it. *Optional* attributes are
    legitimately absent for some specifications (e.g. ``_f_statistic`` is unset for
    IV or fixed-effects-only feols, where there is nothing to test) and are skipped
    silently, so normal runs don't emit spurious warnings.
    """
    if isinstance(fit, Quantreg):
        attrs = (("nobs", "_N", True),)
    elif isinstance(fit, Fepois):
        attrs = (
            ("nobs", "_N", True),
            ("pseudo_r2", "_pseudo_r2", True),
            ("deviance", "deviance", True),
        )
    elif isinstance(fit, Feglm):
        attrs = (
            ("nobs", "_N", True),
            ("deviance", "deviance", True),
        )
    else:
        attrs = (
            ("nobs", "_N", True),
            ("r2", "_r2", True),
            ("adj_r2", "_adj_r2", True),
            ("f_statistic", "_f_statistic", False),
            ("rmse", "_rmse", True),
        )

    metrics = {}
    for name, attr, required in attrs:
        try:
            metrics[name] = float(getattr(fit, attr))
        except (AttributeError, TypeError, ValueError) as exc:
            if required:
                warnings.warn(
                    f"Could not extract metric {name!r} from "
                    f"{type(fit).__name__} (attribute {attr!r}): {exc!r}; skipping.",
                    stacklevel=2,
                )
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
    (e.g. ``pyfixest.fepois``, ``pyfixest.feglm``, ``pyfixest.quantreg``) or its name
    as a string (e.g. ``"fepois"``), resolved via ``getattr(pyfixest, model_fn)``. It
    is called as ``model_fn(*args, **kwargs)``.

    All input validation happens before the MLflow run is opened, so a bad input
    never leaves a FAILED run behind: binding the call arguments (a signature
    mismatch raises ``TypeError``) and parsing the formula (a malformed formula
    raises ``FormulaSyntaxError``) both run first. In particular, only single-model
    results are supported: formulas that produce several models (e.g. via
    ``sw()``/``csw()`` or multiple dependent variables) raise a ``ValueError``
    before any run is opened; the returned object is also checked as a backstop.

    Which metrics get logged depends on the model type (e.g. ``fepois`` has no R2):
    ``_extract_metrics`` picks the relevant (metric_name, attribute) pairs based on
    the type of the fitted result. Metrics are logged to MLflow together with the
    coefficient table and, when the model type supports it, a human-readable
    regression table (pyfixest ``etable``) as a ``summary.html`` artifact. The
    object returned by ``model_fn`` is returned unchanged.

    Only key parameters are logged: the formula, the data's shape, and vcov.
    """
    model_fn = _resolve_model_fn(model_fn)

    # Validate inputs before opening any MLflow run, so a bad formula raises
    # without leaving a FAILED run polluting the experiment history.
    bound_args = _bind_args(model_fn, args, kwargs)
    fml = bound_args.get("fml")
    data = bound_args.get("data")
    vcov = bound_args.get("vcov")

    if fml is not None and len(Formula.parse(fml)) > 1:
        raise ValueError(_MULTI_MODEL_ERROR)

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_param("model_fn", getattr(model_fn, "__name__", str(model_fn)))
        if fml is not None:
            mlflow.log_param("fml", fml)
        if isinstance(data, pd.DataFrame):
            mlflow.log_param("data_shape", str(data.shape))
        if vcov is not None:
            mlflow.log_param("vcov", str(vcov))

        fit = model_fn(*args, **kwargs)

        if isinstance(fit, FixestMulti):
            raise ValueError(_MULTI_MODEL_ERROR)

        mlflow.log_metrics(_extract_metrics(fit))

        coef_table = fit.tidy().reset_index()
        mlflow.log_table(coef_table, artifact_file="coefficients.json")

        # A human-readable regression table, alongside the tidy coefficients, to
        # eyeball runs in the MLflow UI. The summary is a nice-to-have, not the
        # point of the run, so the whole block is failure-safe: both etable
        # generation (not every model type is guaranteed to be supported) and the
        # log_text upload are caught, so neither can fail the run or lose the fit.
        # (The metric and coefficient-table logging above is deliberately not
        # wrapped -- those failing should surface.)
        try:
            summary_html = pf.etable([fit], type="html")
            mlflow.log_text(summary_html, "summary.html")
        except Exception as exc:
            warnings.warn(f"Could not log etable summary: {exc}", stacklevel=2)

    return fit


def _resolve_model_fn(model_fn: Callable[..., Any] | str) -> Callable[..., Any]:
    if not isinstance(model_fn, str):
        return model_fn
    resolved = getattr(pf, model_fn, None)
    if not callable(resolved):
        raise ValueError(f"Unknown pyfixest model function: {model_fn!r}")
    return resolved
