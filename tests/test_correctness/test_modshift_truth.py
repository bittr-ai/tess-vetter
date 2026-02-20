from __future__ import annotations

import numpy as np

from tess_vetter.domain.detection import TransitCandidate
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.features import FeatureConfig
from tess_vetter.features.builder import build_features
from tess_vetter.validation.exovetter_checks import run_modshift


def _make_synthetic_lc(*, with_secondary: bool) -> LightCurveData:
    period = 5.0
    t0 = 1.0
    duration_days = 2.0 / 24.0
    depth = 0.001  # 1000 ppm

    time = np.arange(0.0, 30.0, 0.02, dtype=np.float64)  # ~30 min cadence
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.ones_like(time, dtype=np.float64) * 2e-4
    quality = np.zeros_like(time, dtype=np.int32)
    valid_mask = np.ones_like(time, dtype=bool)

    def _apply_box(center: float, d: float) -> None:
        in_tr = np.abs(time - center) <= (duration_days / 2.0)
        flux[in_tr] -= d

    # Primary transits
    for k in range(-2, 20):
        _apply_box(t0 + k * period, depth)

    # Secondary eclipse at phase 0.5
    if with_secondary:
        for k in range(-2, 20):
            _apply_box(t0 + (k + 0.5) * period, depth)

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=1,
        sector=1,
        cadence_seconds=1800.0,
    )


def test_exovetter_modshift_secondary_increases_ratio() -> None:
    candidate = TransitCandidate(period=5.0, t0=1.0, duration_hours=2.0, depth=0.001, snr=10.0)

    r0 = run_modshift(candidate=candidate, lightcurve=_make_synthetic_lc(with_secondary=False))
    r1 = run_modshift(candidate=candidate, lightcurve=_make_synthetic_lc(with_secondary=True))

    ratio0 = float(r0.details.get("secondary_primary_ratio") or 0.0)
    ratio1 = float(r1.details.get("secondary_primary_ratio") or 0.0)

    # Correctness contract: adding an eclipse-like secondary should increase the
    # ModShift secondary/primary signal ratio (not necessarily to any exact value).
    assert ratio1 > ratio0


def test_feature_schema_splits_v11_vs_v11b_ratio() -> None:
    raw = {
        "target": {"tic_id": 123},
        "ephemeris": {"period_days": 5.0, "t0_btjd": 1.0, "duration_hours": 2.0, "sectors": [1]},
        "depth_ppm": {"input_depth_ppm": 1000.0},
        "check_results": [
            {"id": "V11", "status": "ok", "metrics": {"secondary_primary_ratio": 0.8, "fred": 12.3}},
            {"id": "V11b", "status": "ok", "metrics": {"sig_pri": 10.0, "sig_sec": 3.0, "fred": 1.2}},
        ],
        "pixel_host_hypotheses": {"skipped": True, "reason": "no_tpf"},
        "localization": {"skipped": True, "reason": "no_tpf"},
        "sector_quality_report": {"skipped": True, "reason": "no_tpf"},
        "candidate_evidence": {"skipped": True, "reason": "no_network"},
        "provenance": {"pipeline_version": "test", "code_hash": "test"},
    }

    row = build_features(raw, FeatureConfig(network_ok=False, bulk_mode=True))

    # Canonical definition: modshift_secondary_primary_ratio means exovetter V11.
    assert row["modshift_secondary_primary_ratio"] == 0.8

    # Separate V11b-derived ratio stored distinctly.
    assert row["v11b_secondary_primary_ratio"] == 3.0 / 10.0


def test_exovetter_modshift_prefers_sigma_fields(monkeypatch) -> None:
    """Correctness: when exovetter provides sigma_* and pri/sec, ratio uses sigma_*."""

    class _FakeModShift:
        def __init__(self, lc_name: str = "flux") -> None:
            self.lc_name = lc_name

        def run(self, tce, lk_obj, plot: bool = False):  # noqa: ANN001
            # pri/sec are not signal strengths; sigma_* are the intended significances.
            return {
                "pri": 1000.0,
                "sec": 10.0,
                "sigma_pri": -10.0,  # negative sign conventions should be handled
                "sigma_sec": 1.0,
                "Fred": 12.0,
                "false_alarm_threshold": 0.0,
            }

    try:
        import exovetter.vetters as vetters
    except Exception:
        # If exovetter isn't installed, the check wrapper will skip; that's OK.
        return

    monkeypatch.setattr(vetters, "ModShift", _FakeModShift, raising=True)

    candidate = TransitCandidate(period=5.0, t0=1.0, duration_hours=2.0, depth=0.001, snr=10.0)
    r = run_modshift(candidate=candidate, lightcurve=_make_synthetic_lc(with_secondary=False))

    ratio = float(r.details.get("secondary_primary_ratio") or 0.0)
    # Expected sigma-based ratio: |1| / |10| = 0.1 (not pri/sec = 0.01)
    assert abs(ratio - 0.1) < 1e-6
