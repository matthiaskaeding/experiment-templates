import pandas as pd
import pytest

from features import apply_steps, feature, get_feature, registered_features


def test_builtin_features_are_registered():
    reg = registered_features()
    assert reg["standardize"] == "1"
    assert reg["add_squares"] == "1"


def test_standardize_zscores_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})

    out, applied = apply_steps(df, ["standardize"])

    assert applied == ["standardize@1"]
    assert abs(out["a"].mean()) < 1e-9
    assert abs(out["a"].std() - 1.0) < 1e-9
    # the input frame is not mutated
    assert df["a"].tolist() == [1.0, 2.0, 3.0]


def test_add_squares_adds_squared_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    out, applied = apply_steps(df, ["add_squares"])

    assert applied == ["add_squares@1"]
    assert out["a_sq"].tolist() == [1.0, 4.0, 9.0]


def test_apply_steps_runs_in_order():
    # add_squares first creates a_sq, then standardize z-scores it
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})

    out, applied = apply_steps(df, ["add_squares", "standardize"])

    assert applied == ["add_squares@1", "standardize@1"]
    assert "a_sq" in out.columns
    assert abs(out["a_sq"].mean()) < 1e-9


def test_unknown_feature_raises():
    with pytest.raises(KeyError, match="unknown feature"):
        get_feature("definitely_not_registered")

    with pytest.raises(KeyError):
        apply_steps(pd.DataFrame({"a": [1]}), ["definitely_not_registered"])


def test_duplicate_registration_raises():
    @feature("dup_feature_for_test", version="1")
    def _first(data):
        return data

    with pytest.raises(ValueError, match="already registered"):

        @feature("dup_feature_for_test", version="2")
        def _second(data):
            return data
