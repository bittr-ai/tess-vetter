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


def test_canonical_hash_alias_collapse_payload_is_order_invariant() -> None:
    summary = {
        "alias_scalar_summary": {
            "best_harmonic": "P",
            "best_ratio_over_p": 1.0,
            "score_p": 0.61,
            "score_p_over_2": 0.25,
            "score_2p": 0.14,
            "depth_ppm_peak": 1000.0,
            "classification": "NONE",
            "phase_shift_event_count": 0,
            "phase_shift_peak_sigma": None,
            "secondary_significance": 0.0,
        },
    }
    plot_data = {
        "alias_summary": {
            "harmonic_labels": ["P", "P/2", "2P"],
            "periods": [3.5, 1.75, 7.0],
            "scores": [0.61, 0.25, 0.14],
            "harmonic_depth_ppm": [1000.0, 350.0, 150.0],
            "best_harmonic": "P",
            "best_ratio_over_p": 1.0,
            "classification": "NONE",
            "phase_shift_event_count": 0,
            "phase_shift_peak_sigma": None,
            "secondary_significance": 0.0,
        },
    }

    summary_hash = _canonical_sha256(summary)
    plot_data_hash = _canonical_sha256(plot_data)

    assert summary_hash == _canonical_sha256(_reverse_key_order(copy.deepcopy(summary)))
    assert plot_data_hash == _canonical_sha256(_reverse_key_order(copy.deepcopy(plot_data)))


def test_canonical_hash_data_gap_summary_block_is_order_invariant_and_hash_sensitive() -> None:
    summary = {
        "data_gap_summary": {
            "missing_frac_max_in_coverage": 0.45,
            "missing_frac_median_in_coverage": 0.2,
            "n_epochs_missing_ge_0p25_in_coverage": 3,
            "n_epochs_excluded_no_coverage": 1,
            "n_epochs_evaluated_in_coverage": 9,
        },
    }

    base_hash = _canonical_sha256(summary)
    assert base_hash == _canonical_sha256(_reverse_key_order(copy.deepcopy(summary)))

    changed = copy.deepcopy(summary)
    changed["data_gap_summary"]["n_epochs_missing_ge_0p25_in_coverage"] = 4
    assert _canonical_sha256(changed) != base_hash
