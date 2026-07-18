import warnings

import mlflow
import pyfixest as pf
import pytest
from mlflow.entities import ViewType

from pyfixest_regression import (
    _extract_metrics,
    coefficients_table,
    compute_experiment_hash,
    etable,
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


def test_regress_logs_markdown_summary_artifact(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    fit = regress("Y ~ X1 + X2", data=data, experiment_name="md-summary")

    run = mlflow.last_active_run()
    artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
    assert "summary.md" in {a.path for a in artifacts}
    # the summary is self-built from the fit's own info: coefficient rows,
    # stats rows, and the formula header are all present
    md = mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/summary.md")
    assert "Y ~ X1 + X2" in md
    assert "| X1 " in md and "| nobs " in md
    assert f"{fit._r2:.3f}" in md


def test_regress_markdown_summary_works_for_all_model_types(tmp_path):
    # The old pf.etable-based summary needed a try/except for unsupported model
    # types; the self-built one must work for every estimator we support.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress(
        "Y ~ X1 + X2",
        data=data,
        model_fn=pf.quantreg,
        experiment_name="md-quantreg",
    )

    run = mlflow.last_active_run()
    md = mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/summary.md")
    assert "| X1 " in md


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


def test_unused_column_does_not_change_hash():
    # Only the columns the model reads are hashed, so touching a column the
    # formula never mentions must not change the hash.
    data = pf.get_data()
    params = {"fml": "Y ~ X1 + X2", "model_fn": "feols"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    changed = data.copy()
    changed["f3"] = changed["f3"] + 1  # f3 is not in the formula
    h2 = compute_experiment_hash(changed, params, global_version="v1")

    assert h1 == h2


def test_dropping_unused_column_does_not_change_hash():
    data = pf.get_data()
    params = {"fml": "Y ~ X1 + X2", "model_fn": "feols"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    h2 = compute_experiment_hash(data.drop(columns=["f3"]), params, global_version="v1")

    assert h1 == h2


def test_frame_column_order_does_not_change_hash():
    # Used columns are hashed in sorted order, so reordering the frame's columns
    # leaves the hash unchanged.
    data = pf.get_data()
    params = {"fml": "Y ~ X1 + X2", "model_fn": "feols"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    h2 = compute_experiment_hash(data[data.columns[::-1]], params, global_version="v1")

    assert h1 == h2


def test_fixed_effect_column_is_part_of_hash():
    # Variables after `|` (fixed effects) are used columns.
    data = pf.get_data()
    params = {"fml": "Y ~ X1 | f1", "model_fn": "feols"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    changed = data.copy()
    changed["f1"] = changed["f1"] + 1
    h2 = compute_experiment_hash(changed, params, global_version="v1")

    assert h1 != h2


def test_iv_instrument_column_is_part_of_hash():
    # Instruments after the IV `~` are used columns.
    data = pf.get_data()
    params = {"fml": "Y ~ X2 | X1 ~ Z1", "model_fn": "feols"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    changed = data.copy()
    changed["Z1"] = changed["Z1"] + 1
    h2 = compute_experiment_hash(changed, params, global_version="v1")

    assert h1 != h2


def test_cluster_column_is_part_of_hash_only_when_clustered():
    # A cluster column named in a vcov dict is a used column; the same column is
    # ignored when vcov does not reference it.
    data = pf.get_data()
    changed = data.copy()
    changed["group_id"] = changed["group_id"] + 1

    clustered = {"fml": "Y ~ X1", "vcov": {"CRV1": "group_id"}, "model_fn": "feols"}
    assert compute_experiment_hash(
        data, clustered, global_version="v1"
    ) != compute_experiment_hash(changed, clustered, global_version="v1")

    plain = {"fml": "Y ~ X1", "vcov": "hetero", "model_fn": "feols"}
    assert compute_experiment_hash(
        data, plain, global_version="v1"
    ) == compute_experiment_hash(changed, plain, global_version="v1")


def test_weights_column_is_part_of_hash_only_when_weighted():
    data = pf.get_data()
    changed = data.copy()
    changed["weights"] = changed["weights"] * 2

    weighted = {"fml": "Y ~ X1", "weights": "weights", "model_fn": "feols"}
    assert compute_experiment_hash(
        data, weighted, global_version="v1"
    ) != compute_experiment_hash(changed, weighted, global_version="v1")

    unweighted = {"fml": "Y ~ X1", "model_fn": "feols"}
    assert compute_experiment_hash(
        data, unweighted, global_version="v1"
    ) == compute_experiment_hash(changed, unweighted, global_version="v1")


def test_unextractable_columns_fall_back_to_full_frame_hash():
    # If the used columns can't be determined, hashing falls back to the whole
    # frame -- so even an "unrelated" column change then affects the hash. A
    # formula that parses to multiple models is not a single-model spec, so
    # extraction bails and the conservative full-frame hash is used.
    data = pf.get_data()
    params = {"fml": "Y + Y2 ~ X1", "model_fn": "feols"}

    h1 = compute_experiment_hash(data, params, global_version="v1")
    changed = data.copy()
    changed["f3"] = changed["f3"] + 1
    h2 = compute_experiment_hash(changed, params, global_version="v1")

    assert h1 != h2


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


# --- error capture (#27 / #24) ---


def test_regress_logs_params_and_error_tag_when_fit_fails(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    # Single-model formula (passes the pre-run validation) referencing a column
    # that does not exist, so the estimation itself raises.
    with pytest.raises(Exception, match="does_not_exist"):
        regress("Y ~ does_not_exist", data=data, experiment_name="fit-error")

    runs = mlflow.search_runs(
        experiment_names=["fit-error"], run_view_type=ViewType.ALL
    )
    assert len(runs) == 1
    row = runs.iloc[0]
    # the failed attempt is recoverable: params were logged before the fit ...
    assert row["status"] == "FAILED"
    assert row["params.fml"] == "Y ~ does_not_exist"
    assert row["params.experiment_hash"]
    # ... and the error tag holds the exception
    assert "does_not_exist" in row["tags.error"]


def test_regress_failed_run_is_not_a_dedup_hit(tmp_path):
    # A FAILED attempt logs the experiment hash too, so a retry with *identical*
    # inputs (same hash) must not be treated as a duplicate -- otherwise a
    # transient failure would permanently suppress logging of the successful run.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()
    calls = {"n": 0}

    def flaky_feols(fml, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return pf.feols(fml, data)

    with pytest.raises(RuntimeError, match="transient failure"):
        regress("Y ~ X1 + X2", data=data, model_fn=flaky_feols, experiment_name="flaky")

    fit = regress(
        "Y ~ X1 + X2", data=data, model_fn=flaky_feols, experiment_name="flaky"
    )

    runs = mlflow.search_runs(experiment_names=["flaky"], run_view_type=ViewType.ALL)
    assert set(runs["status"]) == {"FAILED", "FINISHED"}
    assert fit._r2 is not None


# --- cross-run etable (#26) ---


def test_etable_builds_cross_run_table_from_logged_info(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, vcov="iid", experiment_name="xrun")
    regress("Y ~ X1 + X2", data=data, vcov="hetero", experiment_name="xrun")
    regress("Y ~ X1 + X2 | f1", data=data, experiment_name="xrun")

    table = etable("xrun")

    assert list(table.columns) == ["(1)", "(2)", "(3)"]
    # coefficient cells look like "estimate<stars> (se)"
    assert "(" in table.loc["X1", "(1)"] and "*" in table.loc["X1", "(1)"]
    # spec rows make each column self-describing; FEs are visible in fml
    assert table.loc["vcov", "(1)"] == "iid"
    assert table.loc["vcov", "(2)"] == "hetero"
    assert table.loc["fml", "(3)"] == "Y ~ X1 + X2 | f1"
    # the FE model has no Intercept -> empty cell, not NaN
    assert table.loc["Intercept", "(3)"] == ""
    assert table.loc["nobs", "(1)"] == "998"


def test_etable_markdown_output_escapes_formula_pipes(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 | f1", data=data, experiment_name="xrun-md")

    md = etable("xrun-md", type="md")

    assert isinstance(md, str)
    # the formula pipe must be escaped so the markdown table doesn't break
    assert "Y ~ X1 \\| f1" in md


def test_etable_filters_coefficients_and_handles_empty(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    data = pf.get_data()

    regress("Y ~ X1 + X2", data=data, experiment_name="xrun-filter")
    only_x1 = etable("xrun-filter", coefficients="X1")
    assert "X1" in only_x1.index and "X2" not in only_x1.index

    mlflow.create_experiment("xrun-empty")
    assert etable("xrun-empty").empty
    assert etable("xrun-empty", type="md") == ""

    with pytest.raises(ValueError, match="'df' or 'md'"):
        etable("xrun-filter", type="html")
