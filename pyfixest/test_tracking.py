import mlflow
import pyfixest as pf

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
    assert run.data.params["arg_0"] == "Y ~ X1 + X2"
    assert run.data.params["data_shape"] == str(data.shape)


def test_run_experiment_logs_multiple_models(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    result = run_experiment(
        "Y + Y2 ~ X1 + X2", data=data, experiment_name="multi-model"
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert "model0_r2" in metrics
    assert "model1_r2" in metrics
    assert result.to_list()[0]._depvar == "Y"
    assert result.to_list()[1]._depvar == "Y2"


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
    assert run.data.params["model_fn"] == "fepois"
    assert run.data.metrics["nobs"] == fit._N
