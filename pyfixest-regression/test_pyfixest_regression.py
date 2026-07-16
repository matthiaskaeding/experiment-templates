import warnings

import mlflow
import pyfixest as pf
import pytest

from hashing import compute_experiment_hash
from tracking import _extract_metrics, regress


def test_regress_logs_single_model(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress("Y ~ X1 + X2", data=data, name="single-model")

    assert fit._r2 is not None

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert metrics["r2"] == fit._r2
    assert metrics["f_statistic"] == fit._f_statistic
    assert metrics["nobs"] == fit._N
    assert run.data.params["model_fn"] == "feols"
    assert run.data.params["fml"] == "Y ~ X1 + X2"
    assert run.data.params["data_shape"] == str(data.shape)


def test_regress_logs_vcov_and_accepts_positional_args(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress("Y ~ X1 + X2", data, "hetero", name="positional-args")

    run = mlflow.last_active_run()
    assert run.data.params["fml"] == "Y ~ X1 + X2"
    assert run.data.params["data_shape"] == str(data.shape)
    assert run.data.params["vcov"] == "hetero"
    assert fit._vcov_type == "hetero"


def test_regress_rejects_multi_model_formula(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    with pytest.raises(ValueError, match="single-model"):
        regress("Y ~ csw(X1, X2)", data=data, name="csw-formula")


def test_regress_rejects_multi_model_formula_before_fitting(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    # These columns don't exist, so an actual fit would raise a pyfixest formula
    # error instead. Getting our ValueError proves the formula was inspected
    # before model_fn was ever called.
    with pytest.raises(ValueError, match="single-model"):
        regress(
            "Y ~ csw(does_not_exist_1, does_not_exist_2)",
            data=data,
            name="csw-formula-precheck",
        )

    # The formula is rejected before any MLflow run is opened, so this fresh
    # tracking store must contain no runs (no FAILED run left behind).
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_regress_accepts_explicit_model_fn(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")

    fit = regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.fepois,
        name="explicit-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert run.data.params["model_fn"] == "fepois"
    assert metrics["nobs"] == fit._N
    assert metrics["pseudo_r2"] == fit._pseudo_r2
    assert "r2" not in metrics
    assert "f_statistic" not in metrics


def test_regress_logs_feglm_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")
    data["Y_bin"] = (data["Y"] > data["Y"].median()).astype(int)

    fit = regress(
        "Y_bin ~ X1 + X2",
        data=data,
        model_fn=pf.feglm,
        family="logit",
        name="feglm-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert run.data.params["model_fn"] == "feglm"
    assert metrics["nobs"] == fit._N
    assert metrics["deviance"] == fit.deviance
    assert "r2" not in metrics
    assert "f_statistic" not in metrics
    assert "pseudo_r2" not in metrics


def test_regress_logs_quantreg_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.quantreg,
        name="quantreg-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert run.data.params["model_fn"] == "quantreg"
    assert metrics == {"nobs": fit._N}


def test_regress_accepts_model_fn_as_string(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")

    fit = regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn="fepois",
        name="string-model-fn",
    )

    run = mlflow.last_active_run()
    assert run.data.params["model_fn"] == "fepois"
    assert run.data.metrics["nobs"] == fit._N


def test_regress_rejects_unknown_model_fn_string():
    with pytest.raises(ValueError, match="not_a_real_model_fn"):
        regress("Y ~ X1", data=pf.get_data(), model_fn="not_a_real_model_fn")


def test_regress_errors_when_no_experiment_set(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    # MLflow's active experiment is process-global, so point at this store's own
    # Default experiment to be deterministic regardless of what ran before.
    mlflow.set_experiment("Default")

    with pytest.raises(ValueError, match="No MLflow experiment is set"):
        regress("Y ~ X1 + X2", data=pf.get_data())

    # The guard must not itself leave a FAILED run behind in Default.
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_regress_reuses_already_active_experiment(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("already-set")

    fit = regress("Y ~ X1 + X2", data=pf.get_data())

    run = mlflow.last_active_run()
    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name == "already-set"
    assert fit._r2 is not None


def test_regress_logs_etable_summary_artifact(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, name="etable-summary")

    run = mlflow.last_active_run()
    artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
    paths = {a.path for a in artifacts}
    assert "summary.html" in paths


def test_regress_completes_when_etable_fails(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    def boom(*args, **kwargs):
        raise RuntimeError("etable exploded")

    monkeypatch.setattr(pf, "etable", boom)

    with pytest.warns(UserWarning, match="Could not log etable summary"):
        fit = regress("Y ~ X1 + X2", data=data, name="etable-failure")

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


def test_regress_dispatches_on_fit_type_not_function_identity(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data(model="Fepois")

    def fepois_wrapper(*args, **kwargs):
        return pf.fepois(*args, **kwargs)

    fit = regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=fepois_wrapper,
        name="wrapper-model-fn",
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    assert metrics["pseudo_r2"] == fit._pseudo_r2
    assert "r2" not in metrics
    assert "f_statistic" not in metrics


# --- hashing ---


def test_same_inputs_give_same_hash():
    data = pf.get_data()
    params = {"fml": "Y ~ X1 + X2", "vcov": "iid"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    h2 = compute_experiment_hash(data.copy(), dict(params), global_version="v1")

    assert h1 == h2


def test_changed_data_changes_hash():
    data = pf.get_data()
    params = {"fml": "Y ~ X1 + X2", "vcov": "iid"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    changed = data.copy()
    changed["X1"] = changed["X1"] + 1
    h2 = compute_experiment_hash(changed, params, global_version="v1")

    assert h1 != h2


def test_changed_model_params_changes_hash():
    data = pf.get_data()

    h1 = compute_experiment_hash(
        data, {"fml": "Y ~ X1 + X2", "vcov": "iid"}, global_version="v1"
    )
    h2 = compute_experiment_hash(
        data, {"fml": "Y ~ X1", "vcov": "iid"}, global_version="v1"
    )

    assert h1 != h2


def test_changed_model_fn_changes_hash():
    # The modeling function participates in the hash, so the same data + formula
    # under different estimators must not collide (previously they did).
    data = pf.get_data()
    base = {"fml": "Y ~ X1 + X2", "vcov": "iid"}

    h_feols = compute_experiment_hash(
        data, {**base, "model_fn": "feols"}, global_version="v1"
    )
    h_quantreg = compute_experiment_hash(
        data, {**base, "model_fn": "quantreg"}, global_version="v1"
    )

    assert h_feols != h_quantreg


def test_changed_global_version_changes_hash():
    data = pf.get_data()
    params = {"fml": "Y ~ X1 + X2", "vcov": "iid"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    h2 = compute_experiment_hash(data, params, global_version="v2")

    assert h1 != h2


def test_model_params_key_order_does_not_change_hash():
    data = pf.get_data()

    h1 = compute_experiment_hash(
        data, {"fml": "Y ~ X1 + X2", "vcov": "iid"}, global_version="v1"
    )
    h2 = compute_experiment_hash(
        data, {"vcov": "iid", "fml": "Y ~ X1 + X2"}, global_version="v1"
    )

    assert h1 == h2


# --- dedup wiring ---


def test_regress_logs_experiment_hash(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, global_version="v1", name="hash-logging")

    run = mlflow.last_active_run()
    assert "experiment_hash" in run.data.params


def test_regress_skips_duplicate_run_but_returns_model(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit1 = regress("Y ~ X1 + X2", data=data, global_version="v1", name="dedup")
    assert len(mlflow.search_runs(experiment_names=["dedup"])) == 1

    fit2 = regress("Y ~ X1 + X2", data=data, global_version="v1", name="dedup")
    # Second call created no new run, but still returned a valid fitted model.
    assert len(mlflow.search_runs(experiment_names=["dedup"])) == 1
    assert fit2._r2 == fit1._r2


def test_regress_different_global_version_is_not_a_duplicate(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, global_version="v1", name="versions")
    regress("Y ~ X1 + X2", data=data, global_version="v2", name="versions")

    assert len(mlflow.search_runs(experiment_names=["versions"])) == 2


def test_regress_different_model_fn_is_not_a_duplicate(tmp_path):
    # Regression test for the hash-ignores-model_fn bug: same data + formula but
    # different estimators must both be logged, not deduplicated.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, name="model-fn-dedup", global_version="v1")
    regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.quantreg,
        name="model-fn-dedup",
        global_version="v1",
    )

    assert len(mlflow.search_runs(experiment_names=["model-fn-dedup"])) == 2
