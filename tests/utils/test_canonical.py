from __future__ import annotations

import math

import numpy as np
import pytest

from bittr_tess_vetter.utils.canonical import canonical_hash, canonical_json


def test_canonical_json_sorts_dict_keys_and_is_deterministic() -> None:
    a = {"b": 2, "a": 1, "c": [3, 2, 1]}
    b = {"c": [3, 2, 1], "a": 1, "b": 2}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_json_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_json({"x": -float("inf")})

    # Numpy NaN/Inf should also be rejected via float handling.
    with pytest.raises(ValueError):
        canonical_json({"x": np.float64(np.nan)})
    with pytest.raises(ValueError):
        canonical_json({"x": np.float64(np.inf)})


def test_canonical_json_rounds_floats_and_handles_numpy_scalars() -> None:
    # Round-trip should be deterministic and JSONable.
    data = {"x": 1.23456789012345, "y": np.float64(2.0), "z": np.int64(3)}
    out = canonical_json(data).decode("utf-8")
    assert '"z":3' in out
    assert '"y":2' in out  # 2.0 becomes 2
    assert "1.2345678901" in out
    assert math.isnan(1.0) is False

