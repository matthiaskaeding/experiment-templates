# experiment-templates

Copy-paste templates for statistical experiments tracked with MLflow, as a flat
collection of modules at the repository root. The pyfixest regression template
is a single self-contained module, `pyfixest_regression.py` (with
`test_pyfixest_regression.py` and `pyfixest_regression_example.ipynb` alongside),
plus an optional `features.py` for versioned feature transformations. This is not
a Python package; the intended use is copying the module file into your own
project and working from there — one file, no intra-template imports to rewrite
wherever it lands (`features.py` is imported only if you use `steps=`). The flat
layout also keeps the door open for shared building blocks across templates — like
that feature-transformation registry — instead of isolating everything per folder.

## Usage

```python
import mlflow
from pyfixest_regression import regress, results_table

mlflow.set_experiment("my-analysis")   # once per script

fit = regress("Y ~ X1 + X2", data=df, vcov="hetero")

runs = results_table("my-analysis")    # one tidy row per logged run
```

The normal pattern is to call `mlflow.set_experiment(...)` once at the top of
the script (or set the `MLFLOW_EXPERIMENT_NAME` environment variable) and let
runs land in the active experiment; alternatively pass `experiment_name=` or
`experiment_id=` per call. If no experiment is set at all, a warning is issued
and the run lands in MLflow's "Default" experiment. `name` is an optional
descriptor of the regression, used as the MLflow run name. `model_fn` accepts a
pyfixest function or its name as a string, e.g. `model_fn="fepois"`; it defaults
to `feols`.

`data` is dataframe-agnostic — pandas, **polars**, or anything pyfixest accepts
(via narwhals) — and is passed to the fit as given. Runs are identified for
deduplication by a content hash of the model settings plus `dataset_version` (a
string you supply, default `"v1"`) — **the data itself is not hashed**, so you
assert which version of the data a run used and bump `dataset_version` when the
underlying data changes. `global_version` is a separate general knob to force a
re-log.

Multi-model formulas work too: a stepwise `csw()`/`sw()` sweep (or several
dependent variables) is fitted once and logged as one run per resolved model,
and `regress` returns the list of fits. Each run records the resolved `fml` plus
`fml_original` (the formula as written), so `etable("exp")` lines the sweep up
side by side and `results_table` can group it via `fml_original`.

The table helpers below return **polars** by default; pass `backend="pandas"`
for pandas.

`results_table(experiment_name=None, filter_string=None, backend="polars")` pulls
the logged runs back as a tidy DataFrame (one row per run, params and metrics with
prefixes stripped); with no argument it reads the active experiment.
`filter_string` is forwarded to `mlflow.search_runs` for arbitrary server-side
filtering — by a run's name (`"tags.\`mlflow.runName\` = 'baseline'"`), a metric
(`"metrics.r2 > 0.9"`), or a param.

`etable(experiment_name=None, coefficients=None, drop=None, filter_string=None,
type="df", backend="polars")` rebuilds a side-by-side cross-run regression table
(one column per run) from the logged runs; `coefficients` keeps only some rows,
`drop` removes some (e.g. `drop="Intercept"`), `filter_string` restricts which
runs become columns, and `type="md"` returns markdown. In polars the row labels
(coefficients and stats) become a leading `term` column, since polars has no row
index.

`coeftable(experiment_name=None, coefficients=None, drop=None,
filter_string=None, backend="polars")` reads each run's `coefficients.json` into
one long coefficient-level frame (one row per run × coefficient, with the run's
params joined on) — the quick way to get every coefficient across an experiment.
Columns use plain snake_case names (`coefficient`, `estimate`, `std_error`,
`p_value`, `ci_low` / `ci_high` for the 95% CI, `t_value`). `coefficients` /
`drop` keep or remove coefficient rows and `filter_string` restricts which runs
are included.

## What gets logged

Each run logs the key parameters (`model_fn`, `fml`, `data_shape`, `vcov`,
`dataset_version`, `experiment_hash`),
metrics appropriate to the model type (e.g. R² and F-statistic for OLS, pseudo
R² and deviance for Poisson), a short summary of the fit itself (`n_coefs`,
`n_fes` — the number of absorbed fixed effects — and `estimation_time` in
seconds), the fitted coefficient table (`coefficients.json`), and a
human-readable regression table (`summary.md`) built from the run's own logged
info.

Key coefficients are also logged as searchable numeric metrics — `coef.<name>`,
`se.<name>`, `pvalue.<name>` — so you can filter, sort, and plot them in the
MLflow UI (e.g. `search_runs(filter_string="metrics.\`coef.treat\` > 0")`). By
default this covers the first `n_key_coefs=5` coefficients; pass
`key_coefs="treat"` (or a list) to pick the ones you care about — the treatment
effect usually isn't simply "first", since the intercept leads and `C()`/`i()`
expansions reorder. Pass `n_key_coefs=0` to log none. The full coefficient table
is always in `coefficients.json` regardless.

## Feature transformations (`features.py`)

`features.py` is a small registry of **versioned** data transformations. Register
one with the `@feature(name, version)` decorator — it takes a DataFrame and
returns a new one — then apply a pipeline of them with `regress(..., steps=[...])`:

```python
from features import feature
from pyfixest_regression import regress

@feature("winsorize_income", version="1")
def winsorize_income(data):
    out = data.copy()
    lo, hi = out["income"].quantile([0.01, 0.99])
    out["income"] = out["income"].clip(lo, hi)
    return out

regress("y ~ income", data=df, steps=["winsorize_income"])
```

The steps run in order before the fit, and their `name@version` tags are logged
(the `steps` param) and folded into the run's content hash — so the data prep is
part of the run's identity and **bumping a transform's version forces a re-log**.
Two example transforms ship in the module: `standardize` and `add_squares`.
`regress` imports `features.py` only when you pass `steps=`, so grab that file too
if you want this.

## Copying a template

From a clone:

```
cp pyfixest_regression.py path/to/your/project/
```

Or without cloning at all, straight from GitHub into the current directory:

```
curl -sSLO https://raw.githubusercontent.com/matthiaskaeding/experiment-templates/main/pyfixest_regression.py
```

Either way, then install `mlflow`, `pyfixest`, and `polars` in that project
(`narwhals` comes with pyfixest). Grab `test_pyfixest_regression.py` too if you
want the tests; they run with plain `pytest`.

## Development

```
uv sync
uv run pytest
uv run ruff check .
mlflow ui        # inspect logged runs locally
```

By default MLflow logs to `mlflow.db` in the current working directory, so run
`mlflow ui` from the same directory your script ran in (or point both at an
explicit tracking URI via `mlflow.set_tracking_uri(...)` / `--backend-store-uri`).
