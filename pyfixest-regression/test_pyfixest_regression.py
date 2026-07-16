import warnings

import mlflow
import pyfixest as pf
import pytest

from tracking import _extract_metrics, run_experiment


def test_run_experiment_logs_single_model(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = run_experiment("Y ~ X1 + X2", data=data, experiment_name="single-model")

    assert fit._r2 is not None

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert metrics["r2"] == fit._r2
    assert metrics["f_statistic"] == fit._f_statistic
    assert metrics["nobs"] == fit._N
    assert run.data.params["model_fn"] == "feols"
    assert run.data.params["fml"] == "Y ~ X1 + X2"
    assert run.data.params["data_shape"] == str(data.shape)


def test_run_experiment_logs_vcov_and_accepts_positional_args(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = run_experiment(
        "Y ~ X1 + X2", data, "hetero", experiment_name="positional-args"
    )

    run = mlflow.last_active_run()
    assert run.data.params["fml"] == "Y ~ X1 + X2"
    assert run.data.params["data_shape"] == str(data.shape)
    assert run.data.params["vcov"] == "hetero"
    assert fit._vcov_type == "hetero"


def test_run_experiment_rejects_multi_model_formula(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    with pytest.raises(ValueError, match="single-model"):
        run_experiment("Y ~ csw(X1, X2)", data=data, experiment_name="csw-formula")


def test_run_experiment_rejects_multi_model_formula_before_fitting(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    # These columns don't exist, so an actual fit would raise a pyfixest formula
    # error instead. Getting our ValueError proves the formula was inspected
    # before model_fn was ever called.
    with pytest.raises(ValueError, match="single-model"):
        run_experiment(
            "Y ~ csw(does_not_exist_1, does_not_exist_2)",
            data=data,
            experiment_name="csw-formula-precheck",
        )

    # The formula is rejected before any MLflow run is opened, so this fresh
    # tracking store must contain no runs (no FAILED run left behind).
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_run_experiment_accepts_explicit_model_fn(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")

    fit = run_experiment(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.fepois,
        experiment_name="explicit-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert run.data.params["model_fn"] == "fepois"
    assert metrics["nobs"] == fit._N
    assert metrics["pseudo_r2"] == fit._pseudo_r2
    assert "r2" not in metrics
    assert "f_statistic" not in metrics


def test_run_experiment_logs_feglm_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")
    data["Y_bin"] = (data["Y"] > data["Y"].median()).astype(int)

    fit = run_experiment(
        "Y_bin ~ X1 + X2",
        data=data,
        model_fn=pf.feglm,
        family="logit",
        experiment_name="feglm-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert run.data.params["model_fn"] == "feglm"
    assert metrics["nobs"] == fit._N
    assert metrics["deviance"] == fit.deviance
    assert "r2" not in metrics
    assert "f_statistic" not in metrics
    assert "pseudo_r2" not in metrics


def test_run_experiment_logs_quantreg_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = run_experiment(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.quantreg,
        experiment_name="quantreg-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert run.data.params["model_fn"] == "quantreg"
    assert metrics == {"nobs": fit._N}


def test_run_experiment_accepts_model_fn_as_string(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")

    fit = run_experiment(
        "Y ~ X1 + X2",
        data=data,
        model_fn="fepois",
        experiment_name="string-model-fn",
    )

    run = mlflow.last_active_run()
    assert run.data.params["model_fn"] == "fepois"
    assert run.data.metrics["nobs"] == fit._N


def test_run_experiment_rejects_unknown_model_fn_string():
    with pytest.raises(ValueError, match="not_a_real_model_fn"):
        run_experiment("Y ~ X1", data=pf.get_data(), model_fn="not_a_real_model_fn")


def test_run_experiment_errors_when_no_experiment_set(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    # MLflow's active experiment is process-global, so point at this store's own
    # Default experiment to be deterministic regardless of what ran before.
    mlflow.set_experiment("Default")

    with pytest.raises(ValueError, match="No MLflow experiment is set"):
        run_experiment("Y ~ X1 + X2", data=pf.get_data())

    # The guard must not itself leave a FAILED run behind in Default.
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_run_experiment_reuses_already_active_experiment(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("already-set")

    fit = run_experiment("Y ~ X1 + X2", data=pf.get_data())

    run = mlflow.last_active_run()
    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name == "already-set"
    assert fit._r2 is not None


def test_run_experiment_logs_etable_summary_artifact(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    run_experiment("Y ~ X1 + X2", data=data, experiment_name="etable-summary")

    run = mlflow.last_active_run()
    artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
    paths = {a.path for a in artifacts}
    assert "summary.html" in paths


def test_run_experiment_completes_when_etable_fails(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    def boom(*args, **kwargs):
        raise RuntimeError("etable exploded")

    monkeypatch.setattr(pf, "etable", boom)

    with pytest.warns(UserWarning, match="Could not log etable summary"):
        fit = run_experiment("Y ~ X1 + X2", data=data, experiment_name="etable-failure")

    run = mlflow.last_active_run()
    assert "nobs" in run.data.metrics
    assert fit._r2 is not None


def test_extract_metrics_warns_on_missing_attribute():
    class FakeFit:
        _N = 123.0

    fit = FakeFit()  # not a pyfixest result -> hits the default (feols) branch
    with pytest.warns(UserWarning):
        metrics = _extract_metrics(fit)

    assert metrics == {"nobs": 123.0}


def test_extract_metrics_no_warning_for_optional_missing_f_statistic():
    # A fixed-effects-only feols legitimately has no _f_statistic. Because that
    # metric is marked optional, extracting metrics must not emit any warning.
    data = pf.get_data()
    fit = pf.feols("Y ~ 1 | f1", data=data)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        metrics = _extract_metrics(fit)

    assert "nobs" in metrics
    assert "f_statistic" not in metrics


def test_run_experiment_dispatches_on_fit_type_not_function_identity(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")

    def fepois_wrapper(*args, **kwargs):
        return pf.fepois(*args, **kwargs)

    fit = run_experiment(
        "Y ~ X1 + X2",
        data=data,
        model_fn=fepois_wrapper,
        experiment_name="wrapper-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert metrics["pseudo_r2"] == fit._pseudo_r2
    assert "r2" not in metrics
    assert "f_statistic" not in metrics
