"""Run a single pyfixest estimation inside an MLflow-tracked experiment."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
import warnings
from typing import Any, Callable, cast

import mlflow
import pandas as pd
import pyfixest as pf
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.models.feglm_ import Feglm
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.quantreg.quantreg_ import Quantreg

_MULTI_MODEL_ERROR = (
    "regress produced multiple models from a single fit. Multi-model *formulas* "
    "(csw()/sw() or several dependent variables) are supported and logged as one "
    "run each; this looks like split=/fsplit=, which is not supported yet."
)

# MLflow always auto-creates a "Default" experiment with this reserved id, and
# silently falls back to it when no experiment is active. regress warns when a
# run lands there (see below) so the fallback never goes unnoticed.
# Metrics that are counts, rendered without decimals in the summary table.
_INTEGER_METRICS = {"nobs", "n_coefs", "n_fes"}

_DEFAULT_EXPERIMENT_ID = "0"
_NO_EXPERIMENT_WARNING = (
    "No MLflow experiment is set; this run is being logged to the implicit "
    "'Default' experiment. Pass experiment_name=... (or experiment_id=...) to "
    "regress, or call mlflow.set_experiment(...) once at the top of your script."
)


def regress(
    *args: Any,
    model_fn: Callable[..., Any] | str = pf.feols,
    name: str | None = None,
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    tags: dict[str, str] | None = None,
    global_version: str = "0",
    key_coefs: str | list[str] | None = None,
    n_key_coefs: int = 5,
    steps: list[str | tuple[str, dict[str, Any]]] | None = None,
    dataset_version: str = "v1",
    **kwargs: Any,
) -> Any:
    """Fit a pyfixest model inside a tracked, deduplicated MLflow run.

    Inputs are validated before any run opens (so bad input never leaves a FAILED
    run), then the model is fitted and its params, metrics, coefficient table
    (``coefficients.json``), and a markdown summary (``summary.md``) are logged.
    A multi-model formula (``csw()``/``sw()`` or several dependent variables) is
    expanded and logged as one run per resolved model, returning the list of fits;
    each such run also records ``fml_original`` so the sweep can be regrouped.
    Runs are deduplicated on a content hash, so re-running an identical spec logs
    nothing (the model is still fitted and returned). A step or estimation failure
    is recorded as a FAILED run with an ``error`` tag, then re-raised.

    Args:
        *args: Positional arguments forwarded to ``model_fn`` -- typically the
            formula and ``data`` (dataframe-agnostic: pandas, polars, or anything
            pyfixest accepts via narwhals). Bound by name, so they may also be
            passed as keywords.
        model_fn: A pyfixest estimator (e.g. ``pyfixest.fepois``) or its name as a
            string (e.g. ``"fepois"``). Defaults to ``pyfixest.feols``.
        name: Human-readable descriptor used as the MLflow run name. Does not
            affect experiment selection or the dedup identity.
        experiment_name: Select the target experiment by name. Mutually exclusive
            with ``experiment_id``; if neither is given the active experiment is
            used (a ``UserWarning`` is issued if that is the implicit "Default").
        experiment_id: Select the target experiment by id. Mutually exclusive with
            ``experiment_name``.
        tags: Extra MLflow tags to set on the run.
        global_version: General knob folded into the hash; bump it to force a
            re-log of otherwise-identical runs.
        key_coefs: Coefficient name(s) to log as searchable ``coef.<name>`` /
            ``se.<name>`` / ``pvalue.<name>`` metrics. Must be variables of the
            formula (else ``ValueError``). Defaults to the first ``n_key_coefs``.
        n_key_coefs: How many leading coefficients to log when ``key_coefs`` is not
            given (``0`` logs none). The full table is always in
            ``coefficients.json`` regardless.
        steps: Feature transformations from the ``features`` registry -- names or
            ``(name, params)`` pairs, e.g.
            ``["standardize", ("log", {"columns": ["income"]})]`` -- applied to
            ``data`` in order before fitting. Their ``name@version(params)`` tags
            are folded into the hash, so a transform's version is part of the run's
            identity. ``features`` is imported only when ``steps`` is given.
        dataset_version: Asserts which version of the data the run used; part of
            the dedup hash (the data itself is *not* hashed). Bump it when the
            underlying data changes.
        **kwargs: Keyword arguments forwarded to ``model_fn`` (e.g. ``vcov``).

    Returns:
        The fitted pyfixest model, or a ``list`` of them for a multi-model formula.

    Raises:
        ValueError: Both experiment selectors given, an unknown ``key_coefs``
            name, or an unsupported multi-model source (``split=``/``fsplit=``).
        TypeError: ``steps`` given without a dataframe ``data``, or call arguments
            that do not match ``model_fn``'s signature.
    """
    model_fn = _resolve_model_fn(model_fn)

    # Validate before any run: bad input never pollutes history.
    bound_args = _bind_args(model_fn, args, kwargs)
    fml = bound_args.get("fml")

    # Feature steps: resolve the name@version tags now (validates that each
    # feature is registered and constructible) for the hash and params; the
    # data-dependent transform itself runs later, inside _fit.
    step_tags: list[str] = []
    if steps:
        if _data_shape(bound_args.get("data")) is None:
            raise TypeError("steps require `data` to be a dataframe")
        from features import plan_steps as _plan_steps

        step_tags = _plan_steps(steps)

    # Validation runs once, here, against the formula as written -- a resolved
    # sub-model of a sweep may legitimately lack a key coefficient (csw(X1, X2)
    # resolves to Y ~ X1 first).
    if key_coefs is not None and fml is not None:
        _validate_key_coefs(key_coefs, fml)

    if experiment_name is not None and experiment_id is not None:
        raise ValueError("Pass either experiment_name or experiment_id, not both.")
    if experiment_name is not None or experiment_id is not None:
        mlflow.set_experiment(
            experiment_name=experiment_name, experiment_id=experiment_id
        )
    has_explicit_experiment = experiment_name is not None or experiment_id is not None

    # A multi-model formula (csw()/sw() or several dependent variables) is
    # syntactic sugar: expand it into its resolved single-model formulas and run
    # each through _run_single, the same path a standalone call takes -- so
    # per-model dedup, FAILED-run recording, and estimation_time come for free.
    # Each sub-run records the formula as written (fml_original) so the sweep can
    # be grouped back together, e.g. results_table filtered on fml_original.
    parsed = Formula.parse(fml) if fml is not None else []
    if len(parsed) > 1:
        fits = []
        for formula in parsed:
            resolved_fml = formula.formula
            sub_name = f"{name} [{_abbrev_formula(resolved_fml)}]" if name else None
            fits.append(
                _run_single(
                    {**bound_args, "fml": resolved_fml},
                    model_fn=model_fn,
                    name=sub_name,
                    tags=tags,
                    global_version=global_version,
                    dataset_version=dataset_version,
                    key_coefs=key_coefs,
                    n_key_coefs=n_key_coefs,
                    steps=steps,
                    step_tags=step_tags,
                    has_explicit_experiment=has_explicit_experiment,
                    fml_original=fml,
                )
            )
        return fits

    return _run_single(
        bound_args,
        model_fn=model_fn,
        name=name,
        tags=tags,
        global_version=global_version,
        dataset_version=dataset_version,
        key_coefs=key_coefs,
        n_key_coefs=n_key_coefs,
        steps=steps,
        step_tags=step_tags,
        has_explicit_experiment=has_explicit_experiment,
        fml_original=None,
    )


def _run_single(
    bound_args: dict[str, Any],
    *,
    model_fn: Callable[..., Any],
    name: str | None,
    tags: dict[str, str] | None,
    global_version: str,
    dataset_version: str,
    key_coefs: str | list[str] | None,
    n_key_coefs: int,
    steps: list[str | tuple[str, dict[str, Any]]] | None,
    step_tags: list[str],
    has_explicit_experiment: bool,
    fml_original: str | None,
) -> Any:
    """Fit and log one resolved model: the execution half of ``regress``.

    ``regress`` validates and, for a multi-model formula, expands; every model --
    standalone or sub-model of a sweep -- then comes through here, so dedup,
    FAILED-run recording, and ``estimation_time`` behave identically for both.
    ``fml_original`` is the formula as written when this model came from a sweep
    (logged as the ``fml_original`` param), else ``None``.
    """
    fml = bound_args.get("fml")
    data_shape = _data_shape(bound_args.get("data"))
    vcov = bound_args.get("vcov")

    # --- Decide (whether to log -- never whether or how to fit) --------------
    model_params = {k: v for k, v in bound_args.items() if k != "data"}
    model_params["model_fn"] = getattr(model_fn, "__name__", str(model_fn))
    if step_tags:
        model_params["steps"] = step_tags
    experiment_hash = compute_experiment_hash(
        dataset_version, model_params, global_version
    )

    # Dedup hit: nothing to log, just fit and hand the model back.
    if _already_logged(experiment_hash):
        return _fit(model_fn, bound_args, steps)

    # --- Execute: fit and log ------------------------------------------------
    with mlflow.start_run(run_name=name, tags=tags) as run:
        # If no experiment was selected and none was set beforehand, the run lands
        # in MLflow's implicit "Default" experiment. That is allowed but almost
        # never intended, so surface it instead of letting it pass silently.
        if (
            not has_explicit_experiment
            and run.info.experiment_id == _DEFAULT_EXPERIMENT_ID
        ):
            warnings.warn(_NO_EXPERIMENT_WARNING, stacklevel=3)

        # Identity params first, so a failing fit still records what was
        # attempted (formula, hash, data shape, vcov, steps).
        mlflow.log_param("model_fn", getattr(model_fn, "__name__", str(model_fn)))
        # Log the user-given name as a param too (the run_name/tag MLflow always
        # sets is auto-generated when name is None, so it can't tell a real name
        # from a random one; the param is present only when the user named the run).
        if name is not None:
            mlflow.log_param("name", name)
        mlflow.log_param("dataset_version", dataset_version)
        if step_tags:
            mlflow.log_param("steps", ",".join(step_tags))
        mlflow.log_param("experiment_hash", experiment_hash)
        if fml is not None:
            mlflow.log_param("fml", fml)
        if fml_original is not None and fml_original != fml:
            mlflow.log_param("fml_original", fml_original)
        if data_shape is not None:
            mlflow.log_param("data_shape", str(data_shape))
        if vcov is not None:
            mlflow.log_param("vcov", str(vcov))

        # A step or estimation failure gets an `error` tag, the run is marked
        # FAILED by the context manager, and the exception propagates.
        start = time.perf_counter()
        try:
            fit = _fit(model_fn, bound_args, steps)
        except Exception as exc:
            mlflow.set_tag("error", f"{type(exc).__name__}: {exc}"[:500])
            raise
        estimation_time = time.perf_counter() - start

        metrics = _fit_metrics(fit, estimation_time)
        mlflow.log_metrics(metrics)

        _log_key_coefficients(fit, _select_key_coefs(fit, key_coefs, n_key_coefs))

        mlflow.log_table(_tidy_coefficients(fit), artifact_file="coefficients.json")

        # A human-readable regression table, alongside the tidy coefficients, to
        # eyeball runs in the MLflow UI (or anywhere markdown renders). Built from
        # the same information that is logged anyway, so it works for every model
        # type -- no dependency on pf.etable supporting the fit.
        mlflow.log_text(_summary_markdown(fit, metrics), "summary.md")

    return fit


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
    """Map positional/keyword call args to model_fn's parameter names.

    The returned dict is the single representation of the model call: feature
    steps replace its ``data`` entry, and the fit is ``model_fn(**bound_args)``.
    That requires every bound parameter to be passable by keyword, so
    positional-only parameters are rejected up front with a clear error
    (pyfixest's estimators have none; this only bites exotic user-supplied
    ``model_fn`` callables).

    A ``**kwargs`` catch-all in the signature (user-supplied wrappers like
    ``def wrapper(*args, **kwargs)``) is flattened into the dict, so ``data``
    passed through it is still addressable by name -- and stays out of the
    experiment hash. Arguments captured by a ``*args`` catch-all have no names
    to flatten to; they are kept under the catch-all's own key and passed back
    positionally by ``_call_model_fn``.
    """
    signature = inspect.signature(model_fn)
    positional_only = [
        name
        for name, p in signature.parameters.items()
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:
        raise TypeError(
            f"model_fn {getattr(model_fn, '__name__', model_fn)!r} has "
            f"positional-only parameters {positional_only}, which regress does "
            f"not support (it calls model_fn with keyword arguments only)."
        )
    bound = signature.bind_partial(*args, **kwargs)
    arguments: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        else:
            arguments[name] = value
    return arguments


def _call_model_fn(model_fn: Callable[..., Any], bound_args: dict[str, Any]) -> Any:
    """Call ``model_fn(**bound_args)``.

    The one exception to the pure keyword call: arguments that were captured by
    a ``*args`` catch-all (kept under that parameter's own key by ``_bind_args``)
    are handed back positionally, since they have no keyword names.
    """
    var_positional = next(
        (
            name
            for name, p in inspect.signature(model_fn).parameters.items()
            if p.kind is inspect.Parameter.VAR_POSITIONAL
        ),
        None,
    )
    if var_positional is None or var_positional not in bound_args:
        return model_fn(**bound_args)
    keywords = {k: v for k, v in bound_args.items() if k != var_positional}
    return model_fn(*bound_args[var_positional], **keywords)


def _apply_steps(data: Any, steps: list[Any]) -> pd.DataFrame:
    """Fit and apply the feature ``steps`` to ``data``, returning the transformed
    frame. Steps run in pandas, so a non-pandas frame comes back as pandas."""
    from features import fit_steps as _fit_steps

    frame = _to_pandas(data)
    # regress validates that data is a dataframe before any step runs, so this
    # is never None here; assert to narrow it for the type checker.
    assert frame is not None
    transformed, _states, _tags = _fit_steps(frame, steps)
    return transformed


def _fit(
    model_fn: Callable[..., Any],
    bound_args: dict[str, Any],
    steps: list[Any] | None,
) -> Any:
    """The single place a model is fitted.

    Applies the feature steps (if any) to the bound ``data``, calls the
    estimator, and rejects a fit that unexpectedly fanned out into multiple
    models (split=/fsplit=, which regress does not support). Every caller --
    the dedup early return and the logged run -- goes through here, so the fit
    is identical whether or not it is being logged.
    """
    if steps:
        bound_args["data"] = _apply_steps(bound_args["data"], steps)
    fit = _call_model_fn(model_fn, bound_args)
    if isinstance(fit, FixestMulti):
        raise ValueError(_MULTI_MODEL_ERROR)
    return fit


def _to_pandas(data: Any) -> pd.DataFrame | None:
    """A pandas view of any supported dataframe (pandas, polars, ...), or None.

    pyfixest is dataframe-agnostic (via narwhals), so a fit can be handed a polars
    frame directly. The parts of a run that need pandas -- content hashing,
    ``data_shape``, feature steps -- go through this instead, so those keep working
    regardless of the input backend (and a polars frame hashes the same as the
    equivalent pandas one). Returns None when ``data`` is not a dataframe at all.
    """
    if isinstance(data, pd.DataFrame):
        return data
    try:
        import narwhals as nw

        return nw.from_native(data, eager_only=True).to_pandas()
    except TypeError:
        return None


def _data_shape(data: Any) -> tuple[int, int] | None:
    """The ``(rows, cols)`` shape of any supported dataframe, or None if ``data``
    is not a dataframe. Uses narwhals so it works without converting to pandas."""
    if isinstance(data, pd.DataFrame):
        return data.shape
    try:
        import narwhals as nw

        return nw.from_native(data, eager_only=True).shape
    except TypeError:
        return None


def _to_backend(frame: pd.DataFrame, backend: str) -> Any:
    """Return ``frame`` (built internally as pandas) in the requested backend.

    ``"pandas"`` returns it unchanged; ``"polars"`` converts to a
    ``polars.DataFrame`` (polars has no row index, so any pandas index should be
    materialized as a column by the caller first).
    """
    if backend == "pandas":
        return frame
    if backend == "polars":
        import polars as pl

        return pl.from_pandas(frame)
    raise ValueError(f"backend must be 'pandas' or 'polars', got {backend!r}")


# pyfixest's tidy() uses display-style labels (``Estimate``, ``Std. Error``,
# ``Pr(>|t|)``, ``2.5%`` ...). Log them under plain snake_case names that read
# like a normal DataFrame, and in a presentation order that leads with the
# estimate/SE/p-value/CI and pushes the t (or z) statistic to the right.
_COEF_COLUMN_RENAME = {
    "Coefficient": "coefficient",
    "Estimate": "estimate",
    "Std. Error": "std_error",
    "t value": "t_value",
    "Pr(>|t|)": "p_value",
    "2.5%": "ci_low",
    "97.5%": "ci_high",
}
_COEF_COLUMN_ORDER = (
    "coefficient",
    "estimate",
    "std_error",
    "p_value",
    "ci_low",
    "ci_high",
    "t_value",
)


def _tidy_coefficients(fit: Any) -> pd.DataFrame:
    """The fit's coefficient table with standard column names and order.

    Renames pyfixest's tidy() labels to snake_case and reorders to
    ``_COEF_COLUMN_ORDER`` (t/z statistic last). Any columns not in the map/order
    are kept, appended after the known ones, so unusual estimators still round-trip.
    """
    table = fit.tidy().reset_index().rename(columns=_COEF_COLUMN_RENAME)
    known = [c for c in _COEF_COLUMN_ORDER if c in table.columns]
    rest = [c for c in table.columns if c not in known]
    return table[known + rest]


def _n_fixef(fit: Any) -> int:
    """Number of fixed effects absorbed by the fit (0 if none).

    pyfixest stores them as a ``+``-joined string on ``_fixef`` (e.g.
    ``"firm + year"``), or None/empty when the model has no fixed effects.
    """
    fixef = getattr(fit, "_fixef", None)
    if not fixef:
        return 0
    return len([part for part in fixef.split("+") if part.strip()])


def _fit_metrics(fit: Any, estimation_time: float | None = None) -> dict[str, float]:
    """Model metrics plus a run-level summary of the fit itself: how many
    coefficients (``n_coefs``) and absorbed fixed effects (``n_fes``) it has, and
    -- when timed -- how long it took (``estimation_time``)."""
    metrics = _extract_metrics(fit)
    if estimation_time is not None:
        metrics["estimation_time"] = estimation_time
    metrics["n_coefs"] = float(len(fit.tidy()))
    metrics["n_fes"] = float(_n_fixef(fit))
    return metrics


def _validate_key_coefs(key_coefs: str | list[str], fml: str) -> None:
    """Raise if any name in ``key_coefs`` is not a variable of the formula.

    Reuses the formula-variable extraction from the hashing code
    (``_formula_variables``) so the check understands transforms, interactions,
    fixed effects, and IV parts. A typo'd or absent coefficient name should fail
    loudly here -- before any run is opened -- rather than silently logging
    nothing. If the formula can't be parsed for variables the check is skipped
    (the estimator will raise its own error at fit time).
    """
    variables = _formula_variables(fml)
    if variables is None:
        return
    requested = [key_coefs] if isinstance(key_coefs, str) else list(key_coefs)
    missing = [name for name in requested if name not in variables]
    if missing:
        raise ValueError(
            f"key_coefs {missing} are not variables in the formula {fml!r}; "
            f"available variables are {sorted(variables)}."
        )


def _select_key_coefs(
    fit: Any, key_coefs: str | list[str] | None, n_key_coefs: int
) -> list[str]:
    """The coefficient names to log as metrics.

    When ``key_coefs`` is given (a name or list), those are used (their membership
    in the formula is validated up front by ``regress``, so a typo raises rather
    than silently logging nothing); only names that resolve to an actual model
    coefficient are kept, which drops formula variables that don't map to a single
    coefficient (e.g. a factor's base name). Otherwise it falls back to the first
    ``n_key_coefs`` coefficients in model order. The fallback is capped at
    ``n_key_coefs`` on purpose: selecting by position is a convenience, and a
    dummy- or fixed-effect-heavy spec can have hundreds of coefficients that
    should not all become metrics. Position is not reliable for picking the
    treatment effect (the intercept comes first, ``C()``/``i()`` expansions
    reorder), which is exactly why ``key_coefs`` exists.
    """
    index = list(fit.tidy().index)
    if key_coefs is not None:
        requested = [key_coefs] if isinstance(key_coefs, str) else list(key_coefs)
        return [name for name in requested if name in index]
    return index[: max(n_key_coefs, 0)]


def _log_key_coefficients(fit: Any, coef_names: list[str]) -> None:
    """Log each named coefficient as first-class, searchable MLflow metrics.

    For every coefficient logs three *numeric* metrics -- ``coef.<name>`` (the
    estimate), ``se.<name>`` (standard error), and ``pvalue.<name>`` -- so they
    can be sorted, filtered, and plotted in the MLflow store/UI (e.g.
    ``search_runs(filter_string='metrics.`coef.treat` > 0')``). Only numbers are
    logged: stars and confidence intervals are presentation and are rendered from
    these by ``results_table``/``etable``, and the complete, unsanitized
    coefficient table always remains in the ``coefficients.json`` artifact.
    Coefficient names may contain characters MLflow disallows in metric keys
    (e.g. ``C(f1)[T.1.0]``), so those are replaced with ``_`` in the key.
    """
    if not coef_names:
        return
    tidy = fit.tidy()
    metrics = {}
    for coef_name in coef_names:
        row = tidy.loc[coef_name]
        key = re.sub(r"[^\w\-. /]", "_", coef_name)
        metrics[f"coef.{key}"] = float(row["Estimate"])
        metrics[f"se.{key}"] = float(row["Std. Error"])
        metrics[f"pvalue.{key}"] = float(row["Pr(>|t|)"])
    mlflow.log_metrics(metrics)


def _already_logged(experiment_hash: str) -> bool:
    """Whether a FINISHED run with this experiment_hash exists in the active
    experiment.

    ``mlflow.search_runs`` with no experiment argument searches only the currently
    active experiment, so deduplication is scoped to that experiment: the same
    inputs logged under a different experiment are not considered duplicates.
    Only FINISHED runs count -- a FAILED attempt also logs the hash (so the error
    is recoverable), and it must not suppress logging of a successful retry with
    identical inputs.
    """
    # search_runs returns a DataFrame for the default output_format="pandas"; the
    # stub's list[Run] | DataFrame union is for output_format="list", unused here.
    runs = cast(
        pd.DataFrame,
        mlflow.search_runs(
            filter_string=(
                f"params.experiment_hash = '{experiment_hash}' "
                "and attributes.status = 'FINISHED'"
            ),
            max_results=1,
        ),
    )
    return not runs.empty


def _resolve_model_fn(model_fn: Callable[..., Any] | str) -> Callable[..., Any]:
    if not isinstance(model_fn, str):
        return model_fn
    resolved = getattr(pf, model_fn, None)
    if not callable(resolved):
        raise ValueError(f"Unknown pyfixest model function: {model_fn!r}")
    return resolved


def _stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _md_escape(value: Any) -> str:
    """Escape pipes so values (e.g. formulas like ``Y ~ X | f1``) survive
    markdown table cells."""
    return str(value).replace("|", "\\|")


def _summary_markdown(fit: Any, metrics: dict[str, float]) -> str:
    """Build the per-run regression table (markdown) from the fit's own info.

    Uses the same tidy coefficient table and metrics that are logged anyway, so
    it works for every model type -- unlike ``pf.etable``, which cannot be
    applied after the fact and does not support all fits.
    """
    lines = [
        f"### {_md_escape(getattr(fit, '_fml', type(fit).__name__))}",
        "",
        "| Coefficient | Estimate | Std. Error | p-value |",
        "|:---|---:|---:|---:|",
    ]
    for coef_name, row in fit.tidy().iterrows():
        pvalue = float(row["Pr(>|t|)"])
        lines.append(
            f"| {_md_escape(coef_name)} "
            f"| {float(row['Estimate']):.3f}{_stars(pvalue)} "
            f"| {float(row['Std. Error']):.3f} "
            f"| {pvalue:.3f} |"
        )
    lines += ["", "| Statistic | Value |", "|:---|---:|"]
    for stat, value in metrics.items():
        # estimation_time is runtime, not a property of the estimate -- leave it out
        # of the static summary (it also varies run to run).
        if stat == "estimation_time":
            continue
        formatted = f"{int(value)}" if stat in _INTEGER_METRICS else f"{value:.3f}"
        lines.append(f"| {stat} | {formatted} |")
    lines.append(
        "\nSignificance: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001. "
        "Cells: estimate with stars; standard error and p-value alongside."
    )
    return "\n".join(lines)


def _abbrev_formula(fml: Any) -> str:
    """A short label for a formula: its right-hand side (predictors)."""
    if not isinstance(fml, str):
        return ""
    return fml.split("~", 1)[1].strip() if "~" in fml else fml.strip()


def _column_label(name: Any, fml: Any) -> str:
    """An etable column header: the run's name if it has one, else the formula
    abbreviated to its right-hand side."""
    if isinstance(name, str) and name.strip():
        return name
    return _abbrev_formula(fml)


def _dedupe_labels(labels: list[str]) -> list[str]:
    """Make column labels unique by suffixing repeats with ``(2)``, ``(3)`` ...
    so runs that share a name (or formula) still get distinct columns."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def etable(
    experiment_name: str | None = None,
    coefficients: str | list[str] | None = None,
    drop: str | list[str] | None = None,
    filter_string: str | None = None,
    type: str = "df",
    backend: str = "polars",
) -> Any:
    """Build a cross-run regression table from the logged runs.

    Reconstructs a side-by-side comparison -- one column per run (oldest first),
    coefficient rows as ``estimate<stars> (se)``, followed by spec/stat rows
    (``fml``, ``vcov``, ``nobs``, R2-style metrics) -- entirely from what
    ``regress`` logged (``coefficients.json`` + params + metrics). Unlike
    ``pf.etable`` this works after the fact, across runs, for every model type.
    Fixed effects show up in the ``fml`` row (e.g. ``| f1``).

    Columns are headed by each run's ``name`` when it has one, otherwise by an
    abbreviated formula (the right-hand side); duplicate headers get a ``(2)``,
    ``(3)`` ... suffix so the columns stay distinct.

    ``coefficients`` (a name or list) keeps only those coefficient rows; ``drop``
    (a name or list) removes them (keep first, then drop) -- handy for hiding the
    intercept or a block of controls to focus on the coefficient of interest.
    ``filter_string`` is forwarded to ``mlflow.search_runs`` to restrict which runs
    become columns (e.g. ``"tags.`mlflow.runName` = 'baseline'"``). ``type="df"``
    (default) returns a DataFrame; ``type="md"`` returns a markdown string (with
    formula pipes escaped). ``backend`` (``"polars"`` by default, or ``"pandas"``)
    selects the DataFrame type -- for ``"polars"`` the row labels (coefficients and
    stats) become a leading ``term`` column, since polars has no row index.
    Returns an empty DataFrame/string if nothing matches.
    """
    if type not in ("df", "md"):
        raise ValueError(f"type must be 'df' or 'md', got {type!r}")

    runs = results_table(experiment_name, filter_string=filter_string, backend="pandas")
    if runs.empty:
        return _to_backend(runs, backend) if type == "df" else ""
    coefs = coeftable(
        experiment_name,
        coefficients=coefficients,
        drop=drop,
        filter_string=filter_string,
        backend="pandas",
    )

    stat_rows = ("fml", "vcov", "nobs", "r2", "adj_r2", "pseudo_r2", "deviance")
    # search_runs returns newest first; present oldest first (left to right).
    built: list[tuple[str, dict[str, str]]] = []
    for _, run in runs.iloc[::-1].iterrows():
        column: dict[str, str] = {}
        run_coefs = coefs[coefs["run_id"] == run["run_id"]]
        for _, c in run_coefs.iterrows():
            cell = f"{c['estimate']:.3f}{_stars(c['p_value'])} ({c['std_error']:.3f})"
            column[c["coefficient"]] = cell
        for stat in stat_rows:
            value = run.get(stat)
            if value is None or pd.isna(value):
                continue
            if stat == "nobs":
                column[stat] = f"{int(value)}"
            elif isinstance(value, float):
                column[stat] = f"{value:.3f}"
            else:
                column[stat] = str(value)
        built.append((_column_label(run.get("name"), run.get("fml")), column))

    labels = _dedupe_labels([label for label, _ in built])
    columns: dict[str, dict[str, str]] = {
        label: column for label, (_, column) in zip(labels, built)
    }

    # Row order: coefficients in first-seen order across runs, then the stats.
    coef_order = list(dict.fromkeys(coefs["coefficient"]))
    row_order = coef_order + [
        s for s in stat_rows if any(s in col for col in columns.values())
    ]
    table = pd.DataFrame(columns).reindex(row_order).fillna("")

    if type == "md":
        escaped = table.map(_md_escape)
        escaped.index = [_md_escape(i) for i in escaped.index]
        return escaped.to_markdown()
    if backend == "pandas":
        return table
    # polars has no row index, so move the coefficient/stat labels into a column
    return _to_backend(table.rename_axis("term").reset_index(), backend)


def _search_runs(
    experiment_name: str | None, filter_string: str | None
) -> pd.DataFrame:
    """``mlflow.search_runs`` scoped to an experiment and an optional filter.

    With ``experiment_name=None`` it searches the active experiment. The
    ``filter_string`` is passed straight through to MLflow, so it accepts the full
    query syntax over params, metrics, tags, and attributes -- e.g.
    ``"tags.`mlflow.runName` = 'baseline'"`` to filter by a run's ``name``, or
    ``"metrics.r2 > 0.9"``.
    """
    kwargs: dict[str, Any] = {}
    if experiment_name is not None:
        kwargs["experiment_names"] = [experiment_name]
    if filter_string is not None:
        kwargs["filter_string"] = filter_string
    # DataFrame for the default output_format="pandas" (see _already_logged).
    return cast(pd.DataFrame, mlflow.search_runs(**kwargs))


def results_table(
    experiment_name: str | None = None,
    filter_string: str | None = None,
    backend: str = "polars",
) -> Any:
    """Return a tidy one-row-per-run comparison table of logged runs.

    A thin, readable wrapper over ``mlflow.search_runs`` so you don't hand-write
    the query and column selection each time you want to compare runs. Keeps
    ``run_id`` plus the logged params and metrics (with their ``params.``/
    ``metrics.`` prefixes stripped, params before metrics), and drops MLflow
    bookkeeping columns (status, timings, artifact_uri, tags).

    With no argument it reads the active experiment (set via
    ``mlflow.set_experiment(...)``); pass ``experiment_name`` to read a specific
    one. ``filter_string`` is forwarded to ``mlflow.search_runs`` for arbitrary
    server-side filtering -- e.g. by a run's ``name``
    (``"tags.`mlflow.runName` = 'baseline'"``), a metric (``"metrics.r2 > 0.9"``),
    or a param. ``backend`` (``"polars"`` by default, or ``"pandas"``) selects the
    returned DataFrame type. Returns an empty DataFrame if nothing matches.
    """
    runs = _search_runs(experiment_name, filter_string)

    if runs.empty:
        return _to_backend(runs, backend)

    params = [c for c in runs.columns if c.startswith("params.")]
    metrics = [c for c in runs.columns if c.startswith("metrics.")]
    columns = ["run_id", *params, *metrics]
    renamed = {c: c.split(".", 1)[1] for c in params + metrics}
    return _to_backend(runs[columns].rename(columns=renamed), backend)


def coeftable(
    experiment_name: str | None = None,
    coefficients: str | list[str] | None = None,
    drop: str | list[str] | None = None,
    filter_string: str | None = None,
    backend: str = "polars",
) -> Any:
    """Return a coefficient-level table across an experiment's runs.

    Reads each run's logged ``coefficients.json`` artifact (via
    ``mlflow.load_table``) and stacks them into one long DataFrame -- one row per
    (run, coefficient), with the coefficient estimate/std-error/etc. columns -- and
    left-joins the run's logged params (``fml``, ``vcov``, ``model_fn``, ...) so
    each row is self-describing. ``run_id`` identifies the run.

    With no argument it reads the active experiment; pass ``experiment_name`` to
    read a specific one. ``coefficients`` (a name or list) keeps only those
    coefficient rows; ``drop`` (a name or list) removes them -- with both, the keep
    is applied first, then the drop. ``filter_string`` is forwarded to
    ``mlflow.search_runs`` to restrict which runs are included (e.g.
    ``"tags.`mlflow.runName` = 'baseline'"``). ``backend`` (``"polars"`` by default,
    or ``"pandas"``) selects the returned DataFrame type. Returns an empty DataFrame
    if nothing matches.
    """
    runs = _search_runs(experiment_name, filter_string)

    if runs.empty:
        return _to_backend(runs, backend)

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
        table = table[table["coefficient"].isin(names)]
    if drop is not None:
        names = [drop] if isinstance(drop, str) else list(drop)
        table = table[~table["coefficient"].isin(names)]

    return _to_backend(table.reset_index(drop=True), backend)


# --- Experiment hashing ------------------------------------------------------
# Kept in this module so the template is a single self-contained file to copy:
# no intra-template import to rewrite when it lands in someone else's project.


def compute_experiment_hash(
    dataset_version: str,
    model_params: dict[str, Any],
    global_version: str,
) -> str:
    """Hash a pyfixest experiment from its dataset version, model params, version.

    The hash depends on:
    - ``dataset_version``: a caller-supplied tag asserting which version of the
      data the run used. The data itself is *not* hashed -- you own data identity
      via this tag, so bump it whenever the underlying data changes (default
      ``"v1"`` in ``regress``). This keeps the hash cheap and dataframe-agnostic.
    - ``model_params``: the model call's parameters (e.g. formula, vcov), the
      modeling function's name, and any applied feature ``steps``, hashed via a
      deterministic JSON serialization. The function name is included so that, e.g.,
      ``feols`` and ``quantreg`` on the same formula do not collide.
    - ``global_version``: a general version tag, e.g. to force a re-log across an
      experiment.
    """
    hasher = hashlib.sha256()
    hasher.update(str(global_version).encode())
    hasher.update(str(dataset_version).encode())
    hasher.update(_hash_model_params(model_params))
    return hasher.hexdigest()


def _formula_variables(fml: str) -> set[str] | None:
    """The variables a single-model formula references, or None if undetermined.

    Uses the same parser pyfixest does -- ``Formula.parse`` to split the spec into
    its parts (second stage, IV first stage, fixed effects), then ``formulaic`` to
    list each part's variables -- so transforms, interactions, fixed effects (after
    ``|``) and IV instruments are all covered, unlike a regex over the string. The
    result includes the response and can include non-column tokens (e.g. the ``i``
    of pyfixest's ``i(...)``); callers that need real columns should intersect with
    the frame. Returns None for a multi-model spec or any parse failure.
    """
    try:
        from formulaic import Formula as _FormulaicFormula

        parsed = Formula.parse(fml)
        if len(parsed) != 1:
            return None
        model = parsed[0]
        names: set[str] = set()
        for part in (model.second_stage, model.first_stage, model.fixed_effects):
            if part:
                names |= set(_FormulaicFormula(part).required_variables)
        return names
    except Exception:
        return None


def _hash_model_params(model_params: dict[str, Any]) -> bytes:
    return json.dumps(model_params, sort_keys=True, default=str).encode()
