from __future__ import annotations

from typing import Any

from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.report import build_report, build_vet_lc_summary_blocks


def _make_lightcurve() -> LightCurve:
    return LightCurve(
        time=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flux=[1.0, 0.9995, 1.0002, 1.0001, 0.9994, 1.0003, 1.0, 0.9998, 1.0001],
        flux_err=[0.0002] * 9,
    )


def _make_candidate() -> Candidate:
    return Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=2.0),
        depth_ppm=600.0,
    )


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {k: _drop_none(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


def _assert_non_none_subset(expected: Any, actual: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            if value is None:
                continue
            assert key in actual
            _assert_non_none_subset(value, actual[key])
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=False):
            _assert_non_none_subset(expected_item, actual_item)
        return
    assert expected == actual


def test_build_vet_lc_summary_blocks_matches_summary_semantics() -> None:
    report = build_report(_make_lightcurve(), _make_candidate())
    blocks = build_vet_lc_summary_blocks(report)
    summary = report.to_json()["summary"]

    for key in (
        "lc_summary",
        "noise_summary",
        "variability_summary",
        "lc_robustness_summary",
        "odd_even_summary",
        "alias_scalar_summary",
    ):
        assert key in blocks
        _assert_non_none_subset(_drop_none(blocks[key]), summary.get(key))


def test_build_vet_lc_summary_blocks_handles_optional_blocks_absent() -> None:
    report = build_report(
        _make_lightcurve(),
        _make_candidate(),
        include_additional_plots=False,
        include_lc_robustness=False,
    )
    blocks = build_vet_lc_summary_blocks(report)

    assert blocks["lc_robustness_summary"] is None
    assert blocks["alias_scalar_summary"]["best_harmonic"] is None
