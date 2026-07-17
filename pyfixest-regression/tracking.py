"""Run a single pyfixest estimation inside an MLflow-tracked experiment."""

from __future__ import annotations

import inspect
import re
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

from hashing import compute_experiment_hash

_MULTI_MODEL_ERROR = (
    "regress only supports single-model results; the formula produced "
    "multiple models (e.g. via sw()/csw() or multiple dependent variables)."
)

# MLflow always auto-creates a "Default" experiment with this reserved id, and
# silently falls back to it when no experiment is active. regress warns when a
# run lands there (see below) so the fallback never goes unnoticed.
_DEFAULT_EXPERIMENT_ID = "0"
_NO_EXPERIMENT_WARNING = (
    "No MLflow experiment is set; this run is being logged to the implicit "
    "'Default' experiment. Pass experiment_name=... (or experiment_id=...) to "
    "regress, or call mlflow.set_experiment(...) once at the top of your script."
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


def _log_coefficient_metrics(fit: Any, coefficients: str | list[str]) -> None:
    """Log selected coefficients as first-class, searchable MLflow metrics.

    For each requested coefficient present in ``fit.tidy()``, logs
    ``coef.<name>.estimate``, ``coef.<name>.std_error``, and ``coef.<name>.pvalue``.
    Coefficient names may contain characters MLflow disallows in metric keys
    (e.g. ``C(f1)[T.1.0]``), so those are replaced with ``_`` in the key; a
    requested name that is not in the model is skipped with a warning.
    """
    names = [coefficients] if isinstance(coefficients, str) else list(coefficients)
    tidy = fit.tidy()
    for coef_name in names:
        if coef_name not in tidy.index:
            warnings.warn(
                f"log_coefficients: {coef_name!r} is not a coefficient of the "
                f"fitted model; skipping.",
                stacklevel=3,
            )
            continue
        row = tidy.loc[coef_name]
        key = re.sub(r"[^\w\-. /]", "_", coef_name)
        mlflow.log_metrics(
            {
                f"coef.{key}.estimate": float(row["Estimate"]),
                f"coef.{key}.std_error": float(row["Std. Error"]),
                f"coef.{key}.pvalue": float(row["Pr(>|t|)"]),
            }
        )


def _already_logged(experiment_hash: str) -> bool:
    """Whether a run with this experiment_hash exists in the active experiment.

    ``mlflow.search_runs`` with no experiment argument searches only the currently
    active experiment, so deduplication is scoped to that experiment: the same
    inputs logged under a different experiment are not considered duplicates.
    """
    runs = mlflow.search_runs(
        filter_string=f"params.experiment_hash = '{experiment_hash}'",
        max_results=1,
    )
    return not runs.empty


def regress(
    *args: Any,
    model_fn: Callable[..., Any] | str = pf.feols,
    name: str | None = None,
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    tags: dict[str, str] | None = None,
    global_version: str = "0",
    log_coefficients: str | list[str] | None = None,
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

    ``name`` is an optional human-readable descriptor of the regression, used as
    the MLflow *run name*. It does not select or override the experiment, and it is
    fine to omit: the run is already identified by its content (formula + data +
    settings, via the experiment hash below).

    Experiment selection: pass ``experiment_name`` or ``experiment_id`` (mutually
    exclusive) to have MLflow use that experiment. If neither is given, the run
    uses whatever experiment is already active (e.g. set once via
    ``mlflow.set_experiment(...)`` at the top of a script). If nothing was set at
    all -- the run would land in MLflow's implicit "Default" experiment -- a
    ``UserWarning`` is issued and logging proceeds there.

    ``log_coefficients`` (a coefficient name or list of names) additionally logs
    those coefficients as first-class, searchable metrics --
    ``coef.<name>.estimate`` / ``.std_error`` / ``.pvalue`` -- so they can be
    filtered and sorted in the MLflow store/UI (e.g.
    ``search_runs(filter_string='metrics.`coef.X1.estimate` > 0')``). It is
    deliberately opt-in and scoped: models can have hundreds of dummy or
    fixed-effect coefficients, and all of them always remain available via the
    ``coefficients.json`` artifact regardless.

    Deduplication: when ``data`` is a DataFrame, a content hash of (data, model
    params including ``model_fn``, ``global_version``) is computed via
    ``compute_experiment_hash`` and logged as the ``experiment_hash`` param. Before
    logging, the active experiment is checked for a run with that same hash; if one
    exists, this call skips logging entirely (no duplicate run is created). The
    model is *always* re-fitted and returned either way -- only the MLflow logging
    is skipped -- since MLflow stores metrics/artifacts, not the live fit object.
    Note that logging configuration (``log_coefficients``) is not part of the
    hash: an experiment already logged without coefficient metrics will be
    skipped as a duplicate; bump ``global_version`` to force a re-log.
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

    if experiment_name is not None and experiment_id is not None:
        raise ValueError("Pass either experiment_name or experiment_id, not both.")
    if experiment_name is not None or experiment_id is not None:
        mlflow.set_experiment(
            experiment_name=experiment_name, experiment_id=experiment_id
        )

    experiment_hash = None
    if isinstance(data, pd.DataFrame):
        model_params = {k: v for k, v in bound_args.items() if k != "data"}
        model_params["model_fn"] = getattr(model_fn, "__name__", str(model_fn))
        experiment_hash = compute_experiment_hash(data, model_params, global_version)

    fit = model_fn(*args, **kwargs)

    if isinstance(fit, FixestMulti):
        raise ValueError(_MULTI_MODEL_ERROR)

    # Skip logging (no new run) if an identical experiment was already logged in
    # the active experiment; the freshly fitted model is still returned.
    if experiment_hash is not None and _already_logged(experiment_hash):
        return fit

    with mlflow.start_run(run_name=name, tags=tags) as run:
        # If no experiment was selected and none was set beforehand, the run lands
        # in MLflow's implicit "Default" experiment. That is allowed but almost
        # never intended, so surface it instead of letting it pass silently.
        if (
            experiment_name is None
            and experiment_id is None
            and run.info.experiment_id == _DEFAULT_EXPERIMENT_ID
        ):
            warnings.warn(_NO_EXPERIMENT_WARNING, stacklevel=2)

        mlflow.log_param("model_fn", getattr(model_fn, "__name__", str(model_fn)))
        if experiment_hash is not None:
            mlflow.log_param("experiment_hash", experiment_hash)
        if fml is not None:
            mlflow.log_param("fml", fml)
        if isinstance(data, pd.DataFrame):
            mlflow.log_param("data_shape", str(data.shape))
        if vcov is not None:
            mlflow.log_param("vcov", str(vcov))

        mlflow.log_metrics(_extract_metrics(fit))

        if log_coefficients is not None:
            _log_coefficient_metrics(fit, log_coefficients)

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


def results_table(experiment_name: str | None = None) -> pd.DataFrame:
    """Return a tidy one-row-per-run comparison table of logged runs.

    A thin, readable wrapper over ``mlflow.search_runs`` so you don't hand-write
    the query and column selection each time you want to compare runs. Keeps
    ``run_id`` plus the logged params and metrics (with their ``params.``/
    ``metrics.`` prefixes stripped, params before metrics), and drops MLflow
    bookkeeping columns (status, timings, artifact_uri, tags).

    With no argument it reads the active experiment (set via
    ``mlflow.set_experiment(...)``); pass ``experiment_name`` to read a specific
    one. Returns an empty DataFrame if the experiment has no runs.
    """
    if experiment_name is None:
        runs = mlflow.search_runs()
    else:
        runs = mlflow.search_runs(experiment_names=[experiment_name])

    if runs.empty:
        return runs

    params = [c for c in runs.columns if c.startswith("params.")]
    metrics = [c for c in runs.columns if c.startswith("metrics.")]
    columns = ["run_id", *params, *metrics]
    renamed = {c: c.split(".", 1)[1] for c in params + metrics}
    return runs[columns].rename(columns=renamed)


def coefficients_table(
    experiment_name: str | None = None,
    coefficients: str | list[str] | None = None,
) -> pd.DataFrame:
    """Return a coefficient-level table across an experiment's runs.

    Reads each run's logged ``coefficients.json`` artifact (via
    ``mlflow.load_table``) and stacks them into one long DataFrame -- one row per
    (run, coefficient), with the coefficient estimate/std-error/etc. columns -- and
    left-joins the run's logged params (``fml``, ``vcov``, ``model_fn``, ...) so
    each row is self-describing. ``run_id`` identifies the run.

    With no argument it reads the active experiment; pass ``experiment_name`` to
    read a specific one. Pass ``coefficients`` (a name or list of names) to keep
    only those coefficients. Returns an empty DataFrame if the experiment has no
    runs.

    Note: coefficients are read back from the per-run artifact, not from
    searchable params/metrics -- so you cannot yet push coefficient filtering into
    the MLflow query (see the coefficient-level-logging issue).
    """
    if experiment_name is None:
        runs = mlflow.search_runs()
    else:
        runs = mlflow.search_runs(experiment_names=[experiment_name])

    if runs.empty:
        return runs

    run_ids = runs["run_id"].tolist()
    table = mlflow.load_table(
        "coefficients.json", run_ids=run_ids, extra_columns=["run_id"]
    )

    param_cols = [c for c in runs.columns if c.startswith("params.")]
    params = runs[["run_id", *param_cols]].rename(
        columns={c: c.split(".", 1)[1] for c in param_cols}
    )
    table = table.merge(params, on="run_id", how="left")

    if coefficients is not None:
        names = [coefficients] if isinstance(coefficients, str) else list(coefficients)
        table = table[table["Coefficient"].isin(names)].reset_index(drop=True)

    return table
