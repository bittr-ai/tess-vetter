from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.api.types import LightCurve
from bittr_tess_vetter.report import AliasHarmonicSummaryData, ReportData
from bittr_tess_vetter.validation.alias_diagnostics import HarmonicScore, classify_alias
from bittr_tess_vetter.validation.report_bridge import compute_alias_diagnostics

_SCORE_ABS_TOL = 0.01
_RATIO_ABS_TOL = 0.01
_SIGNIFICANCE_ABS_TOL = 0.1


def _build_alias_scalar_summary_block(alias_summary: AliasHarmonicSummaryData) -> dict[str, object]:
    report = ReportData(alias_summary=alias_summary, checks_run=[])
    return report.to_json()["summary"]["alias_scalar_summary"]


def _make_alias_probe_lc() -> LightCurve:
    dt_days = 10.0 / (24.0 * 60.0)
    time = np.arange(0.0, 27.0, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 50.0e-6)

    phase = ((time - 0.5) / 3.5) % 1.0
    primary = np.minimum(phase, 1.0 - phase) < 0.015
    secondary = np.abs(phase - 0.5) < 0.015

    flux[primary] *= 1.0 - 0.010
    flux[secondary] *= 1.0 - 0.003
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def _assert_non_negative_alias_scalars(block: dict[str, object]) -> None:
    for key in ("score_p", "score_p_over_2", "score_2p"):
        value = block.get(key)
        if value is not None:
            assert isinstance(value, float)
            assert value >= 0.0

    secondary = block.get("secondary_significance")
    if secondary is not None:
        assert isinstance(secondary, float)
        assert secondary >= 0.0

    phase_shift_peak = block.get("phase_shift_peak_sigma")
    if phase_shift_peak is not None:
        assert isinstance(phase_shift_peak, float)
        assert phase_shift_peak >= 0.0


def test_alias_scalar_summary_physics_invariants_and_determinism() -> None:
    lc = _make_alias_probe_lc()
    diagnostics = compute_alias_diagnostics(
        lc.to_internal(),
        period_days=3.5,
        t0_btjd=0.5,
        duration_hours=2.5,
    )

    alias_summary = AliasHarmonicSummaryData(
        harmonic_labels=diagnostics.harmonic_labels,
        periods=diagnostics.periods,
        scores=diagnostics.scores,
        harmonic_depth_ppm=diagnostics.harmonic_depth_ppm,
        best_harmonic=diagnostics.best_harmonic,
        best_ratio_over_p=diagnostics.best_ratio_over_p,
        classification=diagnostics.classification,
        phase_shift_event_count=diagnostics.phase_shift_event_count,
        phase_shift_peak_sigma=diagnostics.phase_shift_peak_sigma,
        secondary_significance=diagnostics.secondary_significance,
    )

    block_a = _build_alias_scalar_summary_block(alias_summary)
    block_b = _build_alias_scalar_summary_block(alias_summary)

    assert block_a == block_b
    assert block_a["classification"] == diagnostics.classification

    label_to_score = dict(zip(diagnostics.harmonic_labels, diagnostics.scores, strict=False))
    assert block_a["best_ratio_over_p"] == pytest.approx(diagnostics.best_ratio_over_p, abs=_RATIO_ABS_TOL)
    assert block_a["score_p"] == pytest.approx(label_to_score.get("P"), abs=_SCORE_ABS_TOL)
    assert block_a["score_p_over_2"] == pytest.approx(label_to_score.get("P/2"), abs=_SCORE_ABS_TOL)
    assert block_a["score_2p"] == pytest.approx(label_to_score.get("2P"), abs=_SCORE_ABS_TOL)

    phase_shift_peak = block_a.get("phase_shift_peak_sigma")
    if diagnostics.phase_shift_peak_sigma is None:
        assert phase_shift_peak is None
    else:
        assert phase_shift_peak == pytest.approx(
            diagnostics.phase_shift_peak_sigma, abs=_SIGNIFICANCE_ABS_TOL
        )
    assert block_a.get("secondary_significance") == pytest.approx(
        diagnostics.secondary_significance, abs=_SIGNIFICANCE_ABS_TOL
    )

    _assert_non_negative_alias_scalars(block_a)


@pytest.mark.parametrize(
    ("ratio", "expected_classification"),
    [
        (1.09, "NONE"),
        (1.10, "ALIAS_WEAK"),
        (1.49, "ALIAS_WEAK"),
        (1.50, "ALIAS_STRONG"),
    ],
)
def test_alias_scalar_summary_threshold_edges(
    ratio: float,
    expected_classification: str,
) -> None:
    harmonic_scores = [
        HarmonicScore(harmonic="P", period=10.0, score=1.0, depth_ppm=1200.0, duration_hours=2.5),
        HarmonicScore(harmonic="P/2", period=5.0, score=ratio, depth_ppm=900.0, duration_hours=2.5),
        HarmonicScore(harmonic="2P", period=20.0, score=0.2, depth_ppm=300.0, duration_hours=2.5),
    ]
    classification, best_harmonic, best_ratio = classify_alias(harmonic_scores, base_score=1.0)

    alias_summary = AliasHarmonicSummaryData(
        harmonic_labels=["P", "P/2", "2P"],
        periods=[10.0, 5.0, 20.0],
        scores=[1.0, ratio, 0.2],
        harmonic_depth_ppm=[1200.0, 900.0, 300.0],
        best_harmonic=best_harmonic,
        best_ratio_over_p=best_ratio,
        classification=classification,
        phase_shift_event_count=0,
        phase_shift_peak_sigma=0.0,
        secondary_significance=0.0,
    )

    block_a = _build_alias_scalar_summary_block(alias_summary)
    block_b = _build_alias_scalar_summary_block(alias_summary)

    assert block_a == block_b
    assert block_a["classification"] == expected_classification
    assert block_a["classification"] == classification
    assert block_a["best_harmonic"] == best_harmonic
    assert block_a["best_ratio_over_p"] == pytest.approx(best_ratio, abs=_RATIO_ABS_TOL)

    _assert_non_negative_alias_scalars(block_a)
