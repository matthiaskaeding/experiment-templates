"""Run a single pyfixest estimation inside an MLflow-tracked experiment."""

from __future__ import annotations

import hashlib
import inspect
import json
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


def _select_key_coefs(
    fit: Any, key_coefs: str | list[str] | None, n_key_coefs: int
) -> list[str]:
    """The coefficient names to log as metrics.

    When ``key_coefs`` is given (a name or list), those are used -- names not in
    the fitted model are dropped with a warning. Otherwise it falls back to the
    first ``n_key_coefs`` coefficients in model order. The fallback is capped at
    ``n_key_coefs`` on purpose: selecting by position is a convenience, and a
    dummy- or fixed-effect-heavy spec can have hundreds of coefficients that
    should not all become metrics. Position is not reliable for picking the
    treatment effect (the intercept comes first, ``C()``/``i()`` expansions
    reorder), which is exactly why ``key_coefs`` exists.
    """
    index = list(fit.tidy().index)
    if key_coefs is not None:
        requested = [key_coefs] if isinstance(key_coefs, str) else list(key_coefs)
        selected = []
        for name in requested:
            if name in index:
                selected.append(name)
            else:
                warnings.warn(
                    f"key_coefs: {name!r} is not a coefficient of the fitted "
                    f"model; skipping.",
                    stacklevel=3,
                )
        return selected
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
    runs = mlflow.search_runs(
        filter_string=(
            f"params.experiment_hash = '{experiment_hash}' "
            "and attributes.status = 'FINISHED'"
        ),
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
    key_coefs: str | list[str] | None = None,
    n_key_coefs: int = 5,
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

    Estimation errors, by contrast, *are* recorded: the fit runs inside the MLflow
    run, after the parameters are logged. If ``model_fn`` raises, the run remains
    in the store (status FAILED) with the formula/hash/params and an ``error`` tag
    holding the exception, so the failed attempt can be recovered later; the
    exception is then re-raised.

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

    Key coefficients are logged as first-class, searchable metrics by default --
    three numeric metrics each, ``coef.<name>`` (estimate) / ``se.<name>`` /
    ``pvalue.<name>`` -- so they can be filtered, sorted, and plotted in the MLflow
    store/UI (e.g. ``search_runs(filter_string='metrics.`coef.treat` > 0')``).
    ``key_coefs`` (a coefficient name or list) picks which -- use it for the
    coefficient you actually care about, e.g. a treatment effect, since selecting
    by position is unreliable (the intercept comes first, ``C()``/``i()``
    expansions reorder). When ``key_coefs`` is not given it falls back to the first
    ``n_key_coefs`` coefficients (default 5); the cap matters because dummy- or
    fixed-effect-heavy specs can have hundreds of coefficients. Pass
    ``n_key_coefs=0`` (with no ``key_coefs``) to log none. Only numbers are logged
    -- stars and CIs are rendered from them elsewhere -- and the complete
    coefficient table always remains in the ``coefficients.json`` artifact.

    Deduplication: when ``data`` is a DataFrame, a content hash of (data, model
    params including ``model_fn``, ``global_version``) is computed via
    ``compute_experiment_hash`` and logged as the ``experiment_hash`` param. Only
    the columns the model actually reads are hashed -- the formula variables plus
    any cluster/weight/offset/split columns -- so adding or changing unrelated
    columns in ``data`` does not create a spurious new run. Before logging, the
    active experiment is checked for a run with that same hash; if one exists, this
    call skips logging entirely (no duplicate run is created). The model is *always*
    re-fitted and returned either way -- only the MLflow logging is skipped -- since
    MLflow stores metrics/artifacts, not the live fit object. Note that logging
    configuration (``key_coefs`` / ``n_key_coefs``) is not part of the hash: an
    experiment already logged with different key coefficients will still be skipped
    as a duplicate; bump ``global_version`` to force a re-log.
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

    # Dedup hit: skip all logging (no new run), but still fit and return the model.
    if experiment_hash is not None and _already_logged(experiment_hash):
        fit = model_fn(*args, **kwargs)
        if isinstance(fit, FixestMulti):
            raise ValueError(_MULTI_MODEL_ERROR)
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

        # Fit inside the run, after the params are logged: if estimation fails,
        # the run still records what was attempted (formula, hash, data shape,
        # vcov), gets an `error` tag with the exception, is marked FAILED by the
        # context manager, and the exception propagates to the caller.
        try:
            fit = model_fn(*args, **kwargs)
        except Exception as exc:
            mlflow.set_tag("error", f"{type(exc).__name__}: {exc}"[:500])
            raise

        if isinstance(fit, FixestMulti):
            raise ValueError(_MULTI_MODEL_ERROR)

        metrics = _extract_metrics(fit)
        mlflow.log_metrics(metrics)

        _log_key_coefficients(fit, _select_key_coefs(fit, key_coefs, n_key_coefs))

        coef_table = fit.tidy().reset_index()
        mlflow.log_table(coef_table, artifact_file="coefficients.json")

        # A human-readable regression table, alongside the tidy coefficients, to
        # eyeball runs in the MLflow UI (or anywhere markdown renders). Built from
        # the same information that is logged anyway, so it works for every model
        # type -- no dependency on pf.etable supporting the fit.
        mlflow.log_text(_summary_markdown(fit, metrics), "summary.md")

    return fit


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
        formatted = f"{int(value)}" if stat == "nobs" else f"{value:.3f}"
        lines.append(f"| {stat} | {formatted} |")
    lines.append(
        "\nSignificance: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001. "
        "Cells: estimate with stars; standard error and p-value alongside."
    )
    return "\n".join(lines)


def etable(
    experiment_name: str | None = None,
    coefficients: str | list[str] | None = None,
    type: str = "df",
) -> pd.DataFrame | str:
    """Build a cross-run regression table from the logged runs.

    Reconstructs a side-by-side comparison -- one column per run (oldest first,
    labeled ``(1)``, ``(2)``, ...), coefficient rows as ``estimate<stars> (se)``,
    followed by spec/stat rows (``fml``, ``vcov``, ``nobs``, R2-style metrics) --
    entirely from what ``regress`` logged (``coefficients.json`` + params +
    metrics). Unlike ``pf.etable`` this works after the fact, across runs, for
    every model type. Fixed effects show up in the ``fml`` row (e.g. ``| f1``).

    ``coefficients`` (a name or list) keeps only those coefficient rows.
    ``type="df"`` (default) returns a DataFrame; ``type="md"`` returns a
    markdown string (with formula pipes escaped). Returns an empty
    DataFrame/string if the experiment has no runs.
    """
    if type not in ("df", "md"):
        raise ValueError(f"type must be 'df' or 'md', got {type!r}")

    runs = results_table(experiment_name)
    if runs.empty:
        return runs if type == "df" else ""
    coefs = coefficients_table(experiment_name, coefficients)

    stat_rows = ("fml", "vcov", "nobs", "r2", "adj_r2", "pseudo_r2", "deviance")
    columns: dict[str, dict[str, str]] = {}
    # search_runs returns newest first; present oldest first as (1), (2), ...
    for i, (_, run) in enumerate(runs.iloc[::-1].iterrows(), start=1):
        column: dict[str, str] = {}
        run_coefs = coefs[coefs["run_id"] == run["run_id"]]
        for _, c in run_coefs.iterrows():
            cell = f"{c['Estimate']:.3f}{_stars(c['Pr(>|t|)'])} ({c['Std. Error']:.3f})"
            column[c["Coefficient"]] = cell
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
        columns[f"({i})"] = column

    # Row order: coefficients in first-seen order across runs, then the stats.
    coef_order = list(dict.fromkeys(coefs["Coefficient"]))
    row_order = coef_order + [
        s for s in stat_rows if any(s in col for col in columns.values())
    ]
    table = pd.DataFrame(columns).reindex(row_order).fillna("")

    if type == "md":
        escaped = table.map(_md_escape)
        escaped.index = [_md_escape(i) for i in escaped.index]
        return escaped.to_markdown()
    return table


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


# --- Content-based hashing ---------------------------------------------------
# Kept in this module so the template is a single self-contained file to copy:
# no intra-template import to rewrite when it lands in someone else's project.


def compute_experiment_hash(
    data: pd.DataFrame,
    model_params: dict[str, Any],
    global_version: str,
) -> str:
    """Hash a pyfixest experiment from its data, model params, and version.

    The hash depends on:
    - ``data``: hashed by content (via ``pandas.util.hash_pandas_object``), so
      identical values always hash the same regardless of object identity or
      whether the DataFrame was copied. Only the columns the model actually reads
      are hashed (see ``_used_columns``): the formula variables plus the cluster,
      weight, offset and split columns named in ``model_params``. Unrelated columns
      in the frame therefore do not affect the hash, and the used columns are hashed
      in sorted order so column order in the frame does not either. If the used
      columns cannot be determined, the whole frame is hashed instead (conservative:
      a smaller-but-wrong column set could collide two genuinely different runs).
      The row index participates in the hash, so a reindexed or reordered frame
      hashes differently even with identical values; pass a frame with a stable
      index (e.g. ``reset_index(drop=True)``) if you want order-only differences
      ignored.
    - ``model_params``: the model call's parameters (e.g. formula, vcov) *and* the
      modeling function's name, hashed via a deterministic JSON serialization. The
      function name is included so that, e.g., ``feols`` and ``quantreg`` on the
      same data and formula do not collide.
    - ``global_version``: an external version tag (e.g. a pipeline or
      dataset-build version) supplied by the caller.

    Note: narrowing the data hash to the used columns changes the hash for every
    experiment relative to older versions that hashed the whole frame, so each
    previously logged run re-logs once after upgrading. Bump ``global_version`` if
    you want that re-log to be explicit rather than incidental.
    """
    hasher = hashlib.sha256()
    hasher.update(str(global_version).encode())
    hasher.update(_hash_data(data, _used_columns(data, model_params)))
    hasher.update(_hash_model_params(model_params))
    return hasher.hexdigest()


def _hash_data(data: pd.DataFrame, columns: list[str] | None) -> bytes:
    """Hash the given columns of ``data`` (or the whole frame if ``columns`` is
    None), index included."""
    frame = data if columns is None else data[columns]
    col_repr = ",".join(map(str, frame.columns))
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return col_repr.encode() + row_hashes.tobytes()


def _used_columns(data: pd.DataFrame, model_params: dict[str, Any]) -> list[str] | None:
    """The columns the model reads, as a sorted list, or None to hash everything.

    A pyfixest call touches more of the frame than the bare formula variables:
    besides everything in ``fml`` (transforms, interactions, fixed effects after
    ``|``, IV instruments after a second ``|``), it reads the cluster columns named
    in a ``vcov`` dict (e.g. ``{"CRV1": "firm"}``) and the ``weights`` / ``offset``
    / ``split`` / ``fsplit`` columns. Formula variables are pulled with the same
    parser pyfixest uses (``pyfixest``'s ``Formula.parse`` to split the spec, then
    ``formulaic`` to list each part's variables) rather than a regex, so transforms
    and interactions are handled correctly.

    Returns None -- meaning "hash the whole frame" -- if anything about the
    extraction fails or yields nothing usable. That is deliberately conservative:
    over-hashing (including a column the model ignores) at worst logs a duplicate
    run as distinct, whereas under-hashing (missing a column the model reads) would
    let two genuinely different runs collide onto one hash and silently drop the
    second.
    """
    try:
        from formulaic import Formula as _FormulaicFormula

        def _vars(expr: str | None) -> set[str]:
            if not expr:
                return set()
            return set(_FormulaicFormula(expr).required_variables)

        names: set[str] = set()

        fml = model_params.get("fml")
        if fml is not None:
            parsed = Formula.parse(fml)
            # A multi-model spec (e.g. sw()/csw() or several LHS variables) does
            # not have one well-defined column set; bail to the full-frame hash.
            if len(parsed) != 1:
                return None
            model = parsed[0]
            for part in (model.second_stage, model.first_stage, model.fixed_effects):
                names |= _vars(part)

        vcov = model_params.get("vcov")
        if isinstance(vcov, dict):
            for cluster in vcov.values():
                names |= _vars(cluster)

        for key in ("weights", "offset", "split", "fsplit"):
            value = model_params.get(key)
            if isinstance(value, str):
                names.add(value)

        # Keep only real columns of this frame. formulaic reports function tokens
        # as variables too (e.g. `i` from pyfixest's ``i(...)`` interaction), and a
        # caller could name a column absent from the frame; intersecting drops both.
        used = [str(c) for c in data.columns if c in names]
        if not used:
            return None
        return sorted(used)  # sorted -> hash is invariant to frame column order
    except Exception:
        return None


def _hash_model_params(model_params: dict[str, Any]) -> bytes:
    return json.dumps(model_params, sort_keys=True, default=str).encode()
