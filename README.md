# experiment-templates

Copy-paste templates for statistical experiments tracked with MLflow, as a flat
collection of modules at the repository root. The pyfixest regression template
is a single self-contained module, `pyfixest_regression.py` (with
`test_pyfixest_regression.py` and `pyfixest_regression_example.ipynb` alongside). This is not a
Python package; the intended use is
copying the module file into your own project and working from there — one file,
no intra-template imports to rewrite wherever it lands. The flat layout also
keeps the door open for shared building blocks across templates — for example a
registry that different templates use to register reusable feature
transformations — instead of isolating everything per folder.

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
descriptor of the regression, used as the MLflow run name — runs are identified
by their content (formula + data + settings) either way. `model_fn` accepts a
pyfixest function or its name as a string, e.g. `model_fn="fepois"`; it defaults
to `feols`.

`results_table(experiment_name=None, filter_string=None)` pulls the logged runs
back as a tidy DataFrame (one row per run, params and metrics with prefixes
stripped); with no argument it reads the active experiment. `filter_string` is
forwarded to `mlflow.search_runs` for arbitrary server-side filtering — by a
run's name (`"tags.\`mlflow.runName\` = 'baseline'"`), a metric
(`"metrics.r2 > 0.9"`), or a param.

`etable(experiment_name=None, coefficients=None, drop=None, filter_string=None,
type="df")` rebuilds a side-by-side cross-run regression table (one column per
run) from the logged runs; `coefficients` keeps only some rows, `drop` removes
some (e.g. `drop="Intercept"`), `filter_string` restricts which runs become
columns, and `type="md"` returns markdown.

`coeftable(experiment_name=None, coefficients=None, drop=None,
filter_string=None)` reads each run's `coefficients.json` into one long
coefficient-level frame (one row per run × coefficient, with the run's params
joined on) — the quick way to get every coefficient across an experiment. Columns
use plain snake_case names (`coefficient`, `estimate`, `std_error`, `p_value`,
`ci_low` / `ci_high` for the 95% CI, `t_value`). `coefficients` / `drop` keep or
remove coefficient rows and `filter_string` restricts which runs are included.

## What gets logged

Each run logs the key parameters (`model_fn`, `fml`, `data_shape`, `vcov`),
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

## Copying a template

From a clone:

```
cp pyfixest_regression.py path/to/your/project/
```

Or without cloning at all, straight from GitHub into the current directory:

```
curl -sSLO https://raw.githubusercontent.com/matthiaskaeding/experiment-templates/main/pyfixest_regression.py
```

Either way, then install `mlflow` and `pyfixest` in that project. Grab
`test_pyfixest_regression.py` too if you want the tests; they run with plain
`pytest`.

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
