from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from bittr_tess_vetter.api.evidence import checks_to_evidence_items
from bittr_tess_vetter.api.types import CheckResult


class _EnumForTest(Enum):
    A = "a"


@dataclass(frozen=True)
class _DataclassForTest:
    x: int


def test_checks_to_evidence_items_jsonable_details_and_metrics_only() -> None:
    checks = [
        CheckResult(
            id="V00",
            name="metrics_only_sentinel",
            passed=None,
            confidence=0.5,
            details={
                "np_scalar": np.float64(1.25),
                "np_array": np.array([1, 2, 3], dtype=np.int64),
                "enum": _EnumForTest.A,
                "dataclass": _DataclassForTest(x=7),
                "mapping_non_str_key": {1: "x"},
                "sequence": (1, 2, 3),
            },
        ),
        CheckResult(
            id="V01",
            name="passed_but_metrics_only_flag",
            passed=True,
            confidence=1.0,
            details={"_metrics_only": True, "depth_ppm": 123.4},
        ),
    ]

    items = checks_to_evidence_items(checks)
    assert isinstance(items, list)
    assert len(items) == 2

    item0 = items[0]
    assert item0["id"] == "V00"
    assert item0["passed"] is None
    assert item0["metrics_only"] is True

    d0 = item0["details"]
    assert isinstance(d0, dict)
    assert isinstance(d0["np_scalar"], float)
    assert d0["np_array"] == [1, 2, 3]
    assert d0["enum"] == "a"
    assert d0["dataclass"] == {"x": 7}
    assert d0["mapping_non_str_key"] == {"1": "x"}
    assert d0["sequence"] == [1, 2, 3]

    item1 = items[1]
    assert item1["id"] == "V01"
    assert item1["passed"] is True
    assert item1["metrics_only"] is True
