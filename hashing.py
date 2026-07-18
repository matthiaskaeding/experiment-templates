"""Content-based hashing for pyfixest experiment runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def compute_experiment_hash(
    data: pd.DataFrame,
    model_params: dict[str, Any],
    global_version: str,
) -> str:
    """Hash a pyfixest experiment from its data, model params, and version.

    The hash depends on:
    - ``data``: hashed by content (via ``pandas.util.hash_pandas_object``), so
      identical values always hash the same regardless of object identity or
      whether the DataFrame was copied. The row index participates in the hash, so
      a reindexed or reordered frame hashes differently even with identical values;
      pass a frame with a stable index (e.g. ``reset_index(drop=True)``) if you want
      order-only differences ignored.
    - ``model_params``: the model call's parameters (e.g. formula, vcov) *and* the
      modeling function's name, hashed via a deterministic JSON serialization. The
      function name is included so that, e.g., ``feols`` and ``quantreg`` on the
      same data and formula do not collide.
    - ``global_version``: an external version tag (e.g. a pipeline or
      dataset-build version) supplied by the caller.
    """
    hasher = hashlib.sha256()
    hasher.update(str(global_version).encode())
    hasher.update(_hash_data(data))
    hasher.update(_hash_model_params(model_params))
    return hasher.hexdigest()


def _hash_data(data: pd.DataFrame) -> bytes:
    columns = ",".join(map(str, data.columns))
    row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy()
    return columns.encode() + row_hashes.tobytes()


def _hash_model_params(model_params: dict[str, Any]) -> bytes:
    return json.dumps(model_params, sort_keys=True, default=str).encode()
