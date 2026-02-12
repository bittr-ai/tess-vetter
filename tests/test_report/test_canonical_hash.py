from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from bittr_tess_vetter.report._data import _canonical_sha256


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "report" / "canonical_hash_payload.json"


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _reverse_key_order(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _reverse_key_order(obj[k]) for k in reversed(list(obj.keys()))}
    if isinstance(obj, list):
        return [_reverse_key_order(v) for v in obj]
    return obj


def test_canonical_hash_matches_golden_fixture() -> None:
    fixture = _load_fixture()

    assert _canonical_sha256(fixture["summary"]) == fixture["expected"]["summary_hash"]
    assert _canonical_sha256(fixture["plot_data"]) == fixture["expected"]["plot_data_hash"]


def test_canonical_hash_is_key_order_invariant() -> None:
    fixture = _load_fixture()

    summary_reordered = _reverse_key_order(copy.deepcopy(fixture["summary"]))
    plot_data_reordered = _reverse_key_order(copy.deepcopy(fixture["plot_data"]))

    assert _canonical_sha256(summary_reordered) == fixture["expected"]["summary_hash"]
    assert _canonical_sha256(plot_data_reordered) == fixture["expected"]["plot_data_hash"]


def test_canonical_hash_normalizes_tuples_like_lists() -> None:
    fixture = _load_fixture()

    plot_data_with_tuple = copy.deepcopy(fixture["plot_data"])
    plot_data_with_tuple["phase_folded"]["phase_range"] = tuple(
        plot_data_with_tuple["phase_folded"]["phase_range"]
    )

    assert _canonical_sha256(plot_data_with_tuple) == fixture["expected"]["plot_data_hash"]
