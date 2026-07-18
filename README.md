# experiment-templates

Copy-paste templates for statistical experiments tracked with MLflow, as a flat
collection of modules at the repository root. The pyfixest regression template
is `tracking.py` + `hashing.py` (with `test_tracking.py` and `example.ipynb`
alongside). This is not a Python package; the intended use is copying the
module files into your own project and working from there. The flat layout also
keeps the door open for shared building blocks across templates — for example a
registry that different templates use to register reusable feature
transformations — instead of isolating everything per folder.

## Usage

```python
import mlflow
from tracking import regress, results_table

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

`results_table(experiment_name=None)` pulls the logged runs back as a tidy
DataFrame (one row per run, params and metrics with prefixes stripped); with no
argument it reads the active experiment.

`etable(experiment_name=None, coefficients=None, type="df")` rebuilds a
side-by-side cross-run regression table (one column per run) from the logged
runs; `type="md"` returns markdown.

`coefficients_table(experiment_name=None, coefficients=None)` reads each run's
`coefficients.json` into one long coefficient-level frame (one row per run ×
coefficient, with the run's params joined on), optionally filtered to specific
coefficient names.

## What gets logged

Each run logs the key parameters (`model_fn`, `fml`, `data_shape`, `vcov`),
metrics appropriate to the model type (e.g. R² and F-statistic for OLS, pseudo
R² and deviance for Poisson), the fitted coefficient table (`coefficients.json`),
and a human-readable regression table (`summary.md`) built from the run's own
logged info. Pass `log_coefficients=["X1", ...]` to additionally log selected
coefficients as searchable metrics (`coef.X1.estimate` / `.std_error` /
`.pvalue`) — opt-in, since models can have hundreds of dummy coefficients.

## Copying a template

From a clone:

```
cp tracking.py hashing.py path/to/your/project/
```

Or without cloning at all, straight from GitHub into the current directory:

```
curl -sSLO https://raw.githubusercontent.com/matthiaskaeding/experiment-templates/main/tracking.py
curl -sSLO https://raw.githubusercontent.com/matthiaskaeding/experiment-templates/main/hashing.py
```

Either way, then install `mlflow` and `pyfixest` in that project. Grab
`test_tracking.py` too if you want the tests; they run with plain `pytest`.

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
