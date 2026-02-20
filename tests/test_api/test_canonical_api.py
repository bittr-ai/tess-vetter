from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from tess_vetter.api.canonical import (
    FLOAT_DECIMAL_PLACES,
    canonical_hash,
    canonical_hash_prefix,
    canonical_json,
)


def test_canonical_json_normalizes_numpy_scalars_and_dates() -> None:
    payload = {
        3: np.int64(7),
        "flag": np.bool_(True),
        "value": np.float64(1.234567890123),
        "date": date(2024, 1, 2),
        "datetime": datetime(2024, 1, 2, 3, 4, 5),
    }

    out = canonical_json(payload).decode("utf-8")

    assert '"3":7' in out
    assert '"flag":true' in out
    assert '"value":1.2345678901' in out
    assert '"date":"2024-01-02"' in out
    assert '"datetime":"2024-01-02T03:04:05"' in out


def test_canonical_json_quantizes_to_int_and_zero() -> None:
    rounded_to_int = canonical_json({"x": 5.0}).decode("utf-8")
    negative_zero = canonical_json({"x": -0.0}).decode("utf-8")
    tiny = canonical_json({"x": 10 ** (-(FLOAT_DECIMAL_PLACES + 2))}).decode("utf-8")

    assert rounded_to_int == '{"x":5}'
    assert negative_zero == '{"x":0}'
    assert tiny == '{"x":0}'


def test_canonical_json_rejects_nan_inf_scalars_and_arrays() -> None:
    with pytest.raises(ValueError, match="NaN is not allowed"):
        canonical_json({"x": float("nan")})

    with pytest.raises(ValueError, match="Inf is not allowed"):
        canonical_json({"x": float("inf")})

    with pytest.raises(ValueError, match="NaN is not allowed"):
        canonical_json({"x": np.array([1.0, np.nan])})

    with pytest.raises(ValueError, match="Inf is not allowed"):
        canonical_json({"x": np.array([1.0, -np.inf])})


def test_canonical_json_set_sorting_mixed_types_is_deterministic() -> None:
    payload = {"items": {1, "2", 3}}

    out = canonical_json(payload)

    assert out == b'{"items":[1,"2",3]}'


def test_canonical_json_rejects_unsupported_type() -> None:
    class NotSerializable:
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json({"x": NotSerializable()})


def test_canonical_hash_and_prefix_behaviors() -> None:
    left = canonical_hash({"b": 2, "a": [1, 2]})
    right = canonical_hash({"a": [1, 2], "b": 2})

    assert left == right
    assert len(left) == 64

    assert canonical_hash_prefix({"a": 1}, length=12) == canonical_hash({"a": 1})[:12]
    assert canonical_hash_prefix({"a": 1}, length="8") == canonical_hash({"a": 1})[:8]

    with pytest.raises(ValueError, match="between 1 and 64"):
        canonical_hash_prefix({"a": 1}, length=0)

    with pytest.raises(ValueError, match="between 1 and 64"):
        canonical_hash_prefix({"a": 1}, length=65)
