import numpy as np
import pandas as pd
import pytest

from features import (
    FeatureTransform,
    Log,
    Standardize,
    apply_states,
    feature,
    fit_steps,
    get_feature,
    load_pipeline,
    plan_steps,
    save_pipeline,
)


@feature("_broken_series", version="1")
class _BrokenSeries(FeatureTransform):
    """A deliberately broken transform: stores a non-JSON value (a Series)."""

    def __init__(self, col: str = "x") -> None:
        self.col = col

    def fit(self, train):
        self.state = {"s": train[self.col]}  # a pd.Series -> not JSON-serializable
        return self

    def _transform(self, df):
        return df.copy()


def _frame():
    return pd.DataFrame(
        {"income": [10.0, 20.0, 30.0, 1000.0], "age": [25.0, 40.0, 60.0, 80.0]}
    )


def test_from_state_round_trip():
    train = _frame()
    t = Standardize(columns=["income"]).fit(train)

    # params is derived from the __init__ signature, so this reconstructs t exactly
    rebuilt = type(t).from_state(t.state, **t.params)

    assert t.params == {"columns": ["income"]}
    pd.testing.assert_frame_equal(t.transform(train), rebuilt.transform(train))


def test_save_load_apply_round_trip(tmp_path):
    df = _frame()
    prepped, states, _ = fit_steps(
        df,
        [
            ("winsorize", {"col": "income"}),
            ("log", {"columns": ["income"]}),
            "standardize",
        ],
    )

    path = str(tmp_path / "pipeline.json")
    save_pipeline(states, path)
    loaded = load_pipeline(path)

    pd.testing.assert_frame_equal(apply_states(df, loaded), prepped)


def test_no_leakage_uses_train_statistics():
    train = pd.DataFrame({"x": [0.0, 10.0]})
    test = pd.DataFrame({"x": [5.0, 15.0]})

    t = Standardize(columns=["x"]).fit(train)
    out = t.transform(test)

    mu = 5.0
    sd = train["x"].std()  # sample std (ddof=1)
    expected = (test["x"] - mu) / sd
    assert np.allclose(out["x"].to_numpy(), expected.to_numpy())


def test_transform_before_fit_raises_for_both():
    df = _frame()
    # the guard lives on the base class, inherited by both stateful and stateless
    with pytest.raises(RuntimeError, match="before fit"):
        Standardize(columns=["income"]).transform(df)
    with pytest.raises(RuntimeError, match="before fit"):
        Log(columns=["income"]).transform(df)


def test_stateless_fitted_state_is_empty_dict(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    fitted = Log(columns=["x"]).fit(df)
    assert fitted.state == {}  # fitted, not None

    prepped, states, _ = fit_steps(df, [("log", {"columns": ["x"]})])
    assert states[0]["state"] == {}

    path = str(tmp_path / "pipeline.json")
    save_pipeline(states, path)
    out = apply_states(df, load_pipeline(path))
    pd.testing.assert_frame_equal(out, prepped)


def test_save_pipeline_none_vs_empty_state(tmp_path):
    path = str(tmp_path / "pipeline.json")

    # state None (never fitted) -> refuse to save
    unfitted = [{"name": "standardize", "version": "1", "params": {}, "state": None}]
    with pytest.raises(ValueError, match="never fitted"):
        save_pipeline(unfitted, path)

    # state {} (fitted, stateless) -> fine
    fitted = [
        {"name": "log", "version": "1", "params": {"columns": ["x"]}, "state": {}}
    ]
    save_pipeline(fitted, path)  # does not raise


def test_fit_steps_does_not_mutate_input():
    df = _frame()
    before = df.copy()

    fit_steps(
        df,
        [
            ("winsorize", {"col": "income"}),
            ("log", {"columns": ["income"]}),
            "standardize",
        ],
    )

    pd.testing.assert_frame_equal(df, before)


def test_apply_states_version_mismatch_raises():
    df = _frame()
    _, states, _ = fit_steps(df, ["standardize"])
    states[0]["version"] = "999"  # pretend the registered version moved on

    with pytest.raises(ValueError, match="version mismatch"):
        apply_states(df, states)


def test_registry_errors():
    # duplicate name
    with pytest.raises(ValueError, match="already registered"):

        @feature("standardize", version="1")
        class _Dup(FeatureTransform):
            def _transform(self, df):
                return df

    # unknown name
    with pytest.raises(KeyError, match="unknown feature"):
        get_feature("definitely_not_registered")

    # decorating a non-FeatureTransform
    with pytest.raises(TypeError, match="FeatureTransform"):

        @feature("_not_a_transform", version="1")
        class _NotATransform:
            pass


def test_constant_column_standardizes_to_zero():
    df = pd.DataFrame({"c": [5.0, 5.0, 5.0]})

    out = Standardize(columns=["c"]).fit(df).transform(df)

    assert (out["c"] == 0.0).all()  # zero std guarded -> 0, not NaN/inf


def test_standardize_empty_columns_standardizes_nothing():
    # columns=[] must mean "no columns", not fall through to "all numeric" (the
    # `or` truthiness bug); an explicit empty list is a real, honored choice.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})

    out = Standardize(columns=[]).fit(df).transform(df)

    pd.testing.assert_frame_equal(out, df)


def test_standardize_single_row_fit_raises():
    # a single row has undefined (NaN) std; refuse rather than emit all-NaN
    df = pd.DataFrame({"x": [5.0]})

    with pytest.raises(ValueError, match="undefined"):
        Standardize(columns=["x"]).fit(df)


def test_log_non_positive_raises():
    # np.log of 0/negative is -inf/NaN; fail loudly instead of leaking garbage
    df = pd.DataFrame({"x": [1.0, 0.0, 3.0]})

    with pytest.raises(ValueError, match="non-positive"):
        Log(columns=["x"]).fit(df).transform(df)


def test_save_pipeline_non_json_params_raises(tmp_path):
    # params are validated too, not just state
    states = [
        {
            "name": "winsorize",
            "version": "1",
            "params": {"col": "x", "q": np.int64(1)},  # numpy int -> not JSON
            "state": {"lo": 0.0, "hi": 1.0},
        }
    ]

    with pytest.raises(TypeError, match="non-JSON-serializable params"):
        save_pipeline(states, str(tmp_path / "pipeline.json"))


def test_step_order_matters():
    df = pd.DataFrame({"income": [1.0, 2.0, 3.0, 100.0]})  # an outlier for winsorize
    steps_a = [("winsorize", {"col": "income", "q": 0.25}), "standardize"]
    steps_b = ["standardize", ("winsorize", {"col": "income", "q": 0.25})]

    # clip-then-scale differs from scale-then-clip
    a, _, _ = fit_steps(df, steps_a)
    b, _, _ = fit_steps(df, steps_b)

    assert not a.equals(b)


def test_non_json_state_raises_type_error(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    _, states, _ = fit_steps(df, ["_broken_series"])

    with pytest.raises(TypeError, match="non-JSON-serializable"):
        save_pipeline(states, str(tmp_path / "pipeline.json"))


def test_fit_transform_equals_fit_then_transform():
    df = _frame()

    a = Standardize().fit_transform(df)
    b = Standardize().fit(df).transform(df)

    pd.testing.assert_frame_equal(a, b)


def test_params_gives_helpful_error_on_misnamed_attribute():
    @feature("_misnamed_attr", version="1")
    class _Misnamed(FeatureTransform):
        def __init__(self, columns):
            self.col = columns  # bug: attribute name != constructor arg name

        def _transform(self, df):
            return df

    t = _Misnamed(columns=["x"])
    with pytest.raises(AttributeError, match=r"expected attribute 'columns'"):
        _ = t.params


def test_plan_steps_resolves_tags_without_data():
    # same tags as fit_steps would produce, but no data touched (no fit)
    tags = plan_steps([("winsorize", {"col": "income"}), "standardize"])
    assert tags == ["winsorize@1(col=income,q=0.01)", "standardize@1"]


def test_plan_steps_validates_feature_names():
    with pytest.raises(KeyError, match="unknown feature"):
        plan_steps(["definitely_not_registered"])


def test_tag_format_sorts_param_keys_and_shows_resolved_defaults():
    df = pd.DataFrame({"income": [10.0, 20.0, 30.0]})

    # q is left to its default; the tag still shows it (params is resolved from the
    # instance), with keys sorted
    _, _, tags = fit_steps(df, [("winsorize", {"col": "income"})])
    assert tags == ["winsorize@1(col=income,q=0.01)"]

    # a bare step with only None-valued params (columns=None -> "all") has no suffix
    _, _, bare = fit_steps(df, ["standardize"])
    assert bare == ["standardize@1"]
