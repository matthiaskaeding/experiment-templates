"""Versioned, JSON-serializable feature transformations.

A tiny framework for feature engineering that can be *fitted* on training data,
*serialized* to JSON, and *replayed* exactly during serving -- so the same data
preparation is shared between experiment scripts and production, and a transform's
behavior is pinned by a version.

Contract
--------
A transform is a ``FeatureTransform`` subclass. It takes its configuration as
normal constructor arguments (column names, quantiles, ...), which must not touch
data. Learned values (means, quantiles, ...) live in ``self.state``:

- ``state is None`` means the transform has never been fitted. Every fitted-check
  uses ``state is None`` (not truthiness), because a fitted *stateless* transform
  legitimately has ``state == {}``.
- Subclasses implement ``_transform`` (required) and override ``fit`` only when
  they learn something from data. They never override ``transform`` or
  ``fit_transform``. State dicts hold only built-in JSON types.

Train/serve workflow
--------------------
::

    prepped, states, tags = fit_steps(
        train_df,
        [("winsorize", {"col": "income"}), ("log", {"columns": ["income"]}), ("standardize", {})],
    )
    booster = lgb.train(params, lgb.Dataset(prepped[features], prepped[target]))
    save_pipeline(states, "pipeline.json")
    # serving:
    states = load_pipeline("pipeline.json")
    preds = booster.predict(apply_states(new_df, states)[features])

Uses only pandas, numpy, and the standard library (no sklearn, no pickle).
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

Step = tuple[str, dict]


class FeatureTransform(ABC):
    """Base class for a versioned, JSON-serializable feature transformation.

    Subclasses take their configuration as normal named constructor arguments and
    set them on ``self`` (they do not need to call ``super().__init__()``: the
    unfitted ``state`` default lives on the class). Only ``_transform`` is
    abstract; ``fit`` (for stateless transforms), ``transform``, ``fit_transform``
    and ``from_state`` are concrete defaults that subclasses inherit and do not
    override (except ``fit``, overridden by stateful transforms).
    """

    state: dict | None = None  # class-attribute default: None = not fitted

    @property
    def params(self) -> dict:
        """The configuration this transform was built with, derived from its
        ``__init__`` signature: every parameter except ``self``, read off the
        instance. Lets ``from_state(t.state, **t.params)`` rebuild a fitted
        transform without a separately stored params dict.

        This requires each constructor argument to be stored on ``self`` under the
        *same* name; if one isn't (e.g. ``__init__(self, columns)`` that does
        ``self.col = columns``), a clear ``AttributeError`` explains the fix instead
        of a bare missing-attribute error.
        """
        names = [
            name
            for name in inspect.signature(type(self).__init__).parameters
            if name != "self"
        ]
        params = {}
        for name in names:
            try:
                params[name] = getattr(self, name)
            except AttributeError:
                raise AttributeError(
                    f"{type(self).__name__}.params expected attribute {name!r} "
                    f"(a parameter of __init__), but it is not set. Store each "
                    f"constructor argument on self under the same name, e.g. "
                    f"`self.{name} = {name}`."
                ) from None
        return params

    def fit(self, train: pd.DataFrame) -> FeatureTransform:
        """Default for stateless transforms: learn nothing, but mark as fitted by
        setting ``state`` to ``{}`` (an instance attribute that shadows the class
        default). Returns ``self``. Stateful subclasses override this and assign
        their learned values into ``self.state`` (a JSON-serializable dict)."""
        self.state = {}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public entry point (not overridable). Guards that the transform was
        fitted, then delegates to the subclass ``_transform`` hook."""
        if self.state is None:
            raise RuntimeError(f"{type(self).__name__} used before fit()")
        return self._transform(df)

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subclass hook: apply the config and ``self.state`` to ``df``. Must not
        mutate ``df``; must return a new frame."""

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Convenience: ``fit(train).transform(train)`` (not overridable)."""
        return self.fit(train).transform(train)

    @classmethod
    def from_state(cls, state: dict, **params) -> FeatureTransform:
        """Reconstruct a fitted instance without refitting: build ``cls(**params)``
        (the same config the transform was created with) and attach ``state``."""
        obj = cls(**params)
        obj.state = state
        return obj


# --- Registry ----------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    """A registered transform: its ``name``, ``version``, and the class."""

    name: str
    version: str
    cls: type[FeatureTransform]


_REGISTRY: dict[str, Feature] = {}


def feature(name: str, version: str):
    """Class decorator registering a ``FeatureTransform`` subclass.

    Raises ``TypeError`` if the decorated object is not a ``FeatureTransform``
    subclass, and ``ValueError`` if ``name`` is already registered. ``version`` is
    coerced to ``str``.
    """

    def decorator(cls: type[FeatureTransform]) -> type[FeatureTransform]:
        if not isinstance(cls, type) or not issubclass(cls, FeatureTransform):
            raise TypeError(
                f"@feature can only decorate FeatureTransform subclasses; got {cls!r}"
            )
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} is already registered")
        _REGISTRY[name] = Feature(name, str(version), cls)
        return cls

    return decorator


def get_feature(name: str) -> Feature:
    """Look up a registered feature by name, or raise ``KeyError`` listing the
    registered names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown feature {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registered_features() -> dict[str, str]:
    """A mapping of every registered feature name to its version."""
    return {name: feat.version for name, feat in _REGISTRY.items()}


# --- Pipeline helpers --------------------------------------------------------


def _tag(name: str, version: str, params: dict) -> str:
    """A compact ``"name@version"`` tag, with ``(k=v,...)`` (sorted keys) appended
    for the params that are set, e.g. ``"winsorize@1(col=income,q=0.01)"``. Params
    left at ``None`` (an unset default, e.g. ``columns=None`` meaning "all") are
    omitted so a bare step reads as ``"standardize@1"``."""
    shown = {k: v for k, v in params.items() if v is not None}
    tag = f"{name}@{version}"
    if shown:
        inner = ",".join(f"{k}={shown[k]}" for k in sorted(shown))
        tag += f"({inner})"
    return tag


def fit_steps(
    train: pd.DataFrame, steps: list[Step]
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Fit and apply a pipeline of steps to ``train``, in order.

    Each step is a ``(name, params)`` pair of a registered name and its
    constructor params -- e.g. ``("log", {"columns": ["income"]})``, or
    ``("standardize", {})`` for a transform with no params. For each step the
    class is looked up, instantiated as ``cls(**params)``, and
    ``fit_transform``-ed on the current frame, carrying the result forward. The stored/tagged params are the transform's
    *resolved* ``.params`` (so defaults are captured too, e.g. ``winsorize``'s
    ``q=0.01``). Returns ``(transformed_train, states, tags)`` where ``states`` is a
    list of ``{"name", "version", "params", "state"}`` dicts (ready for
    ``save_pipeline``) and ``tags`` is the list of ``"name@version(...)"`` tags.
    """
    current = train
    states: list[dict] = []
    tags: list[str] = []
    for step in steps:
        name, params = step
        feat = get_feature(name)
        obj = feat.cls(**params)
        current = obj.fit_transform(current)
        resolved = obj.params
        states.append(
            {
                "name": name,
                "version": feat.version,
                "params": resolved,
                "state": obj.state,
            }
        )
        tags.append(_tag(name, feat.version, resolved))
    return current, states, tags


def plan_steps(steps: list[Step]) -> list[str]:
    """Resolve ``steps`` to the ``"name@version(...)"`` tags ``fit_steps`` would
    produce, without touching any data.

    Instantiates each transform from its params (construction never touches
    data, per the module contract), so an unregistered name or bad params raise
    here -- letting callers validate a pipeline and compute its identity before
    the data-dependent ``fit_steps`` runs. The tags use the transform's
    *resolved* ``.params``, exactly as ``fit_steps`` tags them.
    """
    tags: list[str] = []
    for step in steps:
        name, params = step
        feat = get_feature(name)
        obj = feat.cls(**params)
        tags.append(_tag(name, feat.version, obj.params))
    return tags


def apply_states(df: pd.DataFrame, states: list[dict]) -> pd.DataFrame:
    """Replay a fitted pipeline (from ``fit_steps``/``load_pipeline``) on ``df``.

    Applies each entry in order via ``cls.from_state(state, **params)``. Raises
    ``ValueError`` naming the step if the stored version differs from the currently
    registered version of that feature (a transform changed under a pinned
    pipeline).
    """
    current = df
    for entry in states:
        name = entry["name"]
        feat = get_feature(name)
        if entry["version"] != feat.version:
            raise ValueError(
                f"version mismatch for step {name!r}: pipeline has "
                f"{entry['version']!r}, registry has {feat.version!r}"
            )
        obj = feat.cls.from_state(entry["state"], **entry["params"])
        current = obj.transform(current)
    return current


def save_pipeline(states: list[dict], path: str) -> None:
    """Write a fitted pipeline to ``path`` as JSON (indent 2).

    Validates every entry first (so a bad entry never leaves a partial file):
    raises ``ValueError`` naming a step whose ``state`` is None (never fitted), and
    ``TypeError`` naming a step whose ``params`` or ``state`` is not
    JSON-serializable.
    """
    for entry in states:
        if entry["state"] is None:
            raise ValueError(
                f"step {entry['name']!r} was never fitted (state is None); cannot save"
            )
        for field in ("params", "state"):
            try:
                json.dumps(entry[field])
            except TypeError as exc:
                raise TypeError(
                    f"step {entry['name']!r} has non-JSON-serializable {field}: {exc}"
                ) from exc
    with open(path, "w") as fh:
        json.dump(states, fh, indent=2)


def load_pipeline(path: str) -> list[dict]:
    """Read a pipeline written by ``save_pipeline`` back into a list of states."""
    with open(path) as fh:
        return json.load(fh)


# --- Example transforms ------------------------------------------------------


@feature("standardize", version="1")
class Standardize(FeatureTransform):
    """Stateful: z-score columns using the training mean and standard deviation.

    ``columns`` defaults to None, meaning every numeric column of ``train`` (an
    explicit empty list standardizes nothing). Columns with zero standard deviation
    (constants) are scaled by 1 instead, so the output is all-zero rather than
    NaN/inf. A column whose standard deviation is *undefined* (NaN -- e.g. a
    single-row fit or an all-NaN column) raises, since that is almost certainly a
    caller bug rather than something to paper over.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, train: pd.DataFrame) -> Standardize:
        cols = (
            self.columns
            if self.columns is not None
            else list(train.select_dtypes("number").columns)
        )
        mu = train[cols].mean()
        sd = train[cols].std().replace(0, 1.0)  # constants -> scale by 1 (output 0)
        if sd.isna().any():
            undefined = [c for c in cols if pd.isna(sd[c])]
            raise ValueError(
                f"Standardize: standard deviation is undefined for {undefined} "
                f"(e.g. a single-row fit or an all-NaN column); cannot standardize."
            )
        self.state = {
            "cols": cols,
            "mu": {c: float(mu[c]) for c in cols},
            "sd": {c: float(sd[c]) for c in cols},
        }
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols, mu, sd = self.state["cols"], self.state["mu"], self.state["sd"]
        out = df.copy()
        for c in cols:
            out[c] = (out[c] - mu[c]) / sd[c]
        return out


@feature("log", version="1")
class Log(FeatureTransform):
    """Stateless: add a log-transformed copy of each column.

    For each column ``c`` in ``columns``, adds a new column ``f"{c}{suffix}"``
    (``suffix`` defaults to ``"_log"``) holding the natural log of ``c``. Uses
    ``np.log`` (not ``log1p``), so the columns must be strictly positive:
    non-positive values raise a ``ValueError`` rather than silently emitting
    ``-inf``/``NaN`` into the model. Being stateless, it implements only
    ``_transform``; ``fit`` (which records the empty ``state == {}``), the
    fitted-guard, and serialization are all inherited.
    """

    def __init__(self, columns: list[str], suffix: str = "_log") -> None:
        self.columns = columns
        self.suffix = suffix

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.columns:
            if (df[c] <= 0).any():
                raise ValueError(
                    f"log: column {c!r} has non-positive values, where the natural "
                    f"log is undefined; clip/filter them first (e.g. winsorize)."
                )
            out[f"{c}{self.suffix}"] = np.log(df[c])
        return out


@feature("winsorize", version="1")
class Winsorize(FeatureTransform):
    """Stateful: clip a column to its ``[q, 1 - q]`` training quantiles.

    ``col`` is the column name; ``q`` (default 0.01) is the lower tail probability.
    """

    def __init__(self, col: str, q: float = 0.01) -> None:
        self.col = col
        self.q = q

    def fit(self, train: pd.DataFrame) -> Winsorize:
        self.state = {
            "lo": float(train[self.col].quantile(self.q)),
            "hi": float(train[self.col].quantile(1 - self.q)),
        }
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.col] = out[self.col].clip(self.state["lo"], self.state["hi"])
        return out
