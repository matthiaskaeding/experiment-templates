# experiment-templates

Copy-paste templates for statistical experiments tracked with MLflow. Each
top-level folder is one self-contained template — code plus its tests — named
`library-task`, for example `pyfixest-regression/`. This is not a Python
package; the intended use is copying a folder into your own project and working
from there.

## Usage

```python
import mlflow
from tracking import run_experiment

mlflow.set_experiment("my-analysis")   # once per script

fit = run_experiment("Y ~ X1 + X2", data=df, vcov="hetero")
```

`experiment_name` is an optional per-call override. The normal pattern is to
call `mlflow.set_experiment(...)` once at the top of the script (or set the
`MLFLOW_EXPERIMENT_NAME` environment variable) and let runs land in the active
experiment. `model_fn` accepts a pyfixest function or its name as a string, e.g.
`model_fn="fepois"`; it defaults to `feols`.

## What gets logged

Each run logs the key parameters (`model_fn`, `fml`, `data_shape`, `vcov`),
metrics appropriate to the model type (e.g. R² and F-statistic for OLS, pseudo
R² and deviance for Poisson), the fitted coefficient table (`coefficients.json`),
and a human-readable regression table (`summary.html`) rendered with pyfixest's
`etable`.

## Copying a template

```
cp -r pyfixest-regression/ path/to/your/project/
```

Then install `mlflow` and `pyfixest` in that project. The tests come along with
the folder and run with plain `pytest`.

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
