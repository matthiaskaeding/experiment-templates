"""A small registry of versioned feature transformations for use with ``regress``.

Register a transform with the ``@feature(name, version)`` decorator; it takes a
DataFrame and returns a new one. ``regress(..., steps=[...])`` then applies the
named transforms, in order, to the data before fitting and records which ran
(``name@version``) as the ``steps`` param -- so a run's data preparation is part
of its recorded, hashable identity, and bumping a transform's ``version`` forces
a re-log.

This module is intentionally standalone: ``pyfixest_regression.py`` imports it
lazily, only when ``steps`` are requested, so the core template still works as a
single file. It is also meant to be reused by other logging helpers later.

Register your own transforms for real work::

    from features import feature

    @feature("winsorize_income", version="1")
    def winsorize_income(data):
        out = data.copy()
        lo, hi = out["income"].quantile([0.01, 0.99])
        out["income"] = out["income"].clip(lo, hi)
        return out

    regress("y ~ income", data=df, steps=["winsorize_income"])

The two transforms below (``standardize``, ``add_squares``) are examples that
operate on every numeric column; they are handy for demos and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

Transform = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class Feature:
    """A registered transform: its ``name``, ``version``, and the function."""

    name: str
    version: str
    fn: Transform


_REGISTRY: dict[str, Feature] = {}


def feature(name: str, version: str) -> Callable[[Transform], Transform]:
    """Register ``fn`` as a feature transformation under ``name`` at ``version``.

    Re-registering an existing name raises, so a typo or a duplicated import
    surfaces instead of silently shadowing a transform. ``version`` is coerced to
    a string and participates in a run's identity, so bump it whenever a
    transform's behavior changes.
    """

    def decorator(fn: Transform) -> Transform:
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} is already registered")
        _REGISTRY[name] = Feature(name, str(version), fn)
        return fn

    return decorator


def get_feature(name: str) -> Feature:
    """Look up a registered feature by name (raising ``KeyError`` if unknown)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown feature {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registered_features() -> dict[str, str]:
    """A mapping of every registered feature name to its version."""
    return {name: feat.version for name, feat in _REGISTRY.items()}


def apply_steps(data: pd.DataFrame, steps: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the named transforms to ``data`` in order.

    Returns the transformed frame and the list of applied ``"name@version"`` tags
    (used by ``regress`` for logging and hashing). The input frame is not mutated:
    each transform is expected to return a new frame, and the first step starts
    from ``data`` unchanged. Steps run left to right, so an unknown name raises
    ``KeyError`` when it is reached.
    """
    out = data
    applied: list[str] = []
    for name in steps:
        feat = get_feature(name)
        out = feat.fn(out)
        applied.append(f"{feat.name}@{feat.version}")
    return out, applied


@feature("standardize", version="1")
def standardize(data: pd.DataFrame) -> pd.DataFrame:
    """Z-score every numeric column (mean 0, standard deviation 1).

    Columns with zero or undefined standard deviation (e.g. constants) are left
    unchanged. Operates on all numeric columns, so scope it to a modeling frame
    where standardizing everything makes sense (or register a narrower transform).
    """
    out = data.copy()
    for col in out.select_dtypes("number").columns:
        std = out[col].std()
        if std and not pd.isna(std):
            out[col] = (out[col] - out[col].mean()) / std
    return out


@feature("add_squares", version="1")
def add_squares(data: pd.DataFrame) -> pd.DataFrame:
    """Add a squared term ``<col>_sq`` for every numeric column.

    A convenient way to make quadratic terms available to a formula (which then
    references e.g. ``age_sq``). Existing columns are left untouched.
    """
    out = data.copy()
    for col in list(out.select_dtypes("number").columns):
        out[f"{col}_sq"] = out[col] ** 2
    return out
