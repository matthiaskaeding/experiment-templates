import warnings

import mlflow
import pyfixest as pf
import pytest

from hashing import compute_experiment_hash
from tracking import (
    _extract_metrics,
    coefficients_table,
    regress,
    results_table,
)


def test_regress_logs_single_model(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress("Y ~ X1 + X2", data=data, experiment_name="single-model")

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

    fit = regress("Y ~ X1 + X2", data, "hetero", experiment_name="positional-args")

    run = mlflow.last_active_run()
    assert run.data.params["fml"] == "Y ~ X1 + X2"
    assert run.data.params["data_shape"] == str(data.shape)
    assert run.data.params["vcov"] == "hetero"
    assert fit._vcov_type == "hetero"


def test_regress_rejects_multi_model_formula(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    with pytest.raises(ValueError, match="single-model"):
        regress("Y ~ csw(X1, X2)", data=data, experiment_name="csw-formula")


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
            experiment_name="csw-formula-precheck",
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
        experiment_name="explicit-model-fn",
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


def test_regress_logs_quantreg_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.quantreg,
        experiment_name="quantreg-model-fn",
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
        experiment_name="string-model-fn",
    )

    run = mlflow.last_active_run()
    assert run.data.params["model_fn"] == "fepois"
    assert run.data.metrics["nobs"] == fit._N


def test_regress_rejects_unknown_model_fn_string():
    with pytest.raises(ValueError, match="not_a_real_model_fn"):
        regress("Y ~ X1", data=pf.get_data(), model_fn="not_a_real_model_fn")


def test_regress_warns_when_no_experiment_set(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    # MLflow's active experiment is process-global, so point at this store's own
    # Default experiment to be deterministic regardless of what ran before.
    mlflow.set_experiment("Default")

    with pytest.warns(UserWarning, match="No MLflow experiment is set"):
        fit = regress("Y ~ X1 + X2", data=pf.get_data())

    # Logging proceeds (into Default), and the fit is returned as usual.
    assert fit._r2 is not None
    assert len(mlflow.search_runs(search_all_experiments=True)) == 1


def test_regress_name_sets_the_run_name(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    regress(
        "Y ~ X1 + X2",
        data=pf.get_data(),
        name="baseline iid spec",
        experiment_name="run-name-test",
    )

    run = mlflow.last_active_run()
    assert run.data.tags["mlflow.runName"] == "baseline iid spec"
    # name describes the run; the experiment comes from experiment_name
    assert mlflow.get_experiment(run.info.experiment_id).name == "run-name-test"


def test_regress_rejects_both_experiment_name_and_id(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    with pytest.raises(ValueError, match="not both"):
        regress(
            "Y ~ X1",
            data=pf.get_data(),
            experiment_name="a",
            experiment_id="1",
        )


def test_regress_accepts_experiment_id(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    exp_id = mlflow.create_experiment("by-id")

    fit = regress("Y ~ X1 + X2", data=pf.get_data(), experiment_id=exp_id)

    run = mlflow.last_active_run()
    assert run.info.experiment_id == exp_id
    assert fit._r2 is not None


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

    regress("Y ~ X1 + X2", data=data, experiment_name="etable-summary")

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
        fit = regress("Y ~ X1 + X2", data=data, experiment_name="etable-failure")

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
        experiment_name="wrapper-model-fn",
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

    regress(
        "Y ~ X1 + X2", data=data, global_version="v1", experiment_name="hash-logging"
    )

    run = mlflow.last_active_run()
    assert "experiment_hash" in run.data.params


def test_regress_skips_duplicate_run_but_returns_model(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit1 = regress(
        "Y ~ X1 + X2", data=data, global_version="v1", experiment_name="dedup"
    )
    assert len(mlflow.search_runs(experiment_names=["dedup"])) == 1

    fit2 = regress(
        "Y ~ X1 + X2", data=data, global_version="v1", experiment_name="dedup"
    )
    # Second call created no new run, but still returned a valid fitted model.
    assert len(mlflow.search_runs(experiment_names=["dedup"])) == 1
    assert fit2._r2 == fit1._r2


def test_regress_different_global_version_is_not_a_duplicate(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, global_version="v1", experiment_name="versions")
    regress("Y ~ X1 + X2", data=data, global_version="v2", experiment_name="versions")

    assert len(mlflow.search_runs(experiment_names=["versions"])) == 2


def test_regress_different_model_fn_is_not_a_duplicate(tmp_path):
    # Regression test for the hash-ignores-model_fn bug: same data + formula but
    # different estimators must both be logged, not deduplicated.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress(
        "Y ~ X1 + X2", data=data, experiment_name="model-fn-dedup", global_version="v1"
    )
    regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.quantreg,
        experiment_name="model-fn-dedup",
        global_version="v1",
    )

    assert len(mlflow.search_runs(experiment_names=["model-fn-dedup"])) == 2


# --- results_table ---


def test_results_table_is_tidy_one_row_per_run(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, vcov="iid", experiment_name="rt")
    regress("Y ~ X1 + X2", data=data, vcov="hetero", experiment_name="rt")

    table = results_table("rt")

    assert len(table) == 2
    # params/metrics prefixes are stripped ...
    for col in ("run_id", "fml", "vcov", "model_fn", "r2", "nobs"):
        assert col in table.columns
    # ... and MLflow bookkeeping / prefixed columns are gone
    assert not any(
        c.startswith("params.") or c.startswith("metrics.") for c in table.columns
    )
    assert "status" not in table.columns
    assert set(table["vcov"]) == {"iid", "hetero"}


def test_results_table_reads_active_experiment_by_default(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("active-default")
    regress("Y ~ X1 + X2", data=pf.get_data())

    table = results_table()

    assert len(table) == 1
    assert table["fml"].iloc[0] == "Y ~ X1 + X2"


def test_results_table_empty_experiment_returns_empty(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.create_experiment("no-runs")

    table = results_table("no-runs")

    assert table.empty


# --- coefficients_table ---


def test_coefficients_table_is_long_and_self_describing(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, vcov="iid", experiment_name="ct")
    regress("Y ~ X1 + X2", data=data, vcov="hetero", experiment_name="ct")

    table = coefficients_table("ct")

    # 2 runs x 3 coefficients (Intercept, X1, X2)
    assert len(table) == 6
    assert set(table["Coefficient"]) == {"Intercept", "X1", "X2"}
    # coefficient stats present
    for col in ("Estimate", "Std. Error", "run_id"):
        assert col in table.columns
    # joined run params make each row self-describing
    assert "fml" in table.columns and "vcov" in table.columns
    assert set(table["vcov"]) == {"iid", "hetero"}


def test_coefficients_table_filters_by_coefficient(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, vcov="iid", experiment_name="ct-filter")
    regress("Y ~ X1 + X2", data=data, vcov="hetero", experiment_name="ct-filter")

    only_x1 = coefficients_table("ct-filter", coefficients="X1")
    assert set(only_x1["Coefficient"]) == {"X1"}
    assert len(only_x1) == 2  # one X1 row per run

    x1_x2 = coefficients_table("ct-filter", coefficients=["X1", "X2"])
    assert set(x1_x2["Coefficient"]) == {"X1", "X2"}
    assert len(x1_x2) == 4


def test_coefficients_table_empty_experiment_returns_empty(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.create_experiment("ct-empty")

    assert coefficients_table("ct-empty").empty


# --- log_coefficients ---


def test_regress_logs_selected_coefficients_as_searchable_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress(
        "Y ~ X1 + X2",
        data=data,
        experiment_name="coef-metrics",
        log_coefficients=["X1"],
    )

    run = mlflow.last_active_run()
    metrics = run.data.metrics
    tidy = fit.tidy()
    assert metrics["coef.X1.estimate"] == float(tidy.loc["X1", "Estimate"])
    assert metrics["coef.X1.std_error"] == float(tidy.loc["X1", "Std. Error"])
    assert metrics["coef.X1.pvalue"] == float(tidy.loc["X1", "Pr(>|t|)"])
    # unselected coefficients are not logged as metrics
    assert "coef.X2.estimate" not in metrics

    # the point of first-class logging: filterable in the MLflow store
    hits = mlflow.search_runs(
        experiment_names=["coef-metrics"],
        filter_string="metrics.`coef.X1.estimate` < 0",
    )
    assert len(hits) == 1


def test_regress_log_coefficients_sanitizes_awkward_names(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress(
        "Y ~ X1 + C(f1)",
        data=data,
        experiment_name="coef-sanitize",
        log_coefficients="C(f1)[T.1.0]",
    )

    run = mlflow.last_active_run()
    # parens/brackets are illegal in MLflow metric keys and get replaced by _
    key = "coef.C_f1__T.1.0_.estimate"
    assert key in run.data.metrics
    assert run.data.metrics[key] == float(fit.tidy().loc["C(f1)[T.1.0]", "Estimate"])


def test_regress_log_coefficients_warns_on_unknown_name(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    with pytest.warns(UserWarning, match="not a coefficient"):
        fit = regress(
            "Y ~ X1 + X2",
            data=data,
            experiment_name="coef-unknown",
            log_coefficients=["X1", "not_a_regressor"],
        )

    run = mlflow.last_active_run()
    # the known one is still logged; the run completes normally
    assert "coef.X1.estimate" in run.data.metrics
    assert fit._r2 is not None
