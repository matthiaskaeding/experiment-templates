import mlflow
import pyfixest as pf
import pytest

from tracking import run_experiment


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
