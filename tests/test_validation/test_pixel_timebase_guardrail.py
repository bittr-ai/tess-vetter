from dataclasses import dataclass

import numpy as np

from bittr_tess_vetter.api import Candidate, Ephemeris, LightCurve, vet_candidate


@dataclass(frozen=True)
class DummyTPF:
    time: np.ndarray
    flux: np.ndarray


def test_pixel_checks_flag_disjoint_lc_tpf_time_ranges() -> None:
    # LC time range: [0, 1]
    lc = LightCurve(
        time=np.linspace(0.0, 1.0, 1000),
        flux=np.ones(1000),
        flux_err=np.full(1000, 1e-4),
    )

    # TPF time range: [100, 101] (disjoint from LC)
    tpf_time = np.linspace(100.0, 101.0, 500, dtype=np.float64)
    tpf_flux = np.ones((len(tpf_time), 5, 5), dtype=np.float64)
    tpf = DummyTPF(time=tpf_time, flux=tpf_flux)

    # Put transits within the TPF time range so V08 can run.
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=100.5, duration_hours=2.0),
        depth_ppm=500.0,
    )

    bundle = vet_candidate(lc, candidate, tpf=tpf, network=False, checks=["V08"])
    assert len(bundle.results) == 1
    r = bundle.results[0]
    assert r.id == "V08"

    assert r.status in {"ok", "warn", "skipped", "error"}
    # Guardrail should trigger deterministically for disjoint time ranges.
    assert "LC_TPF_TIME_DISJOINT" in (r.flags or [])
    assert r.metrics is not None
    assert r.metrics.get("lc_tpf_time_overlap_days") == 0.0

