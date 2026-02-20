from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tess_vetter.api.generate_report import generate_report
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.domain.target import StellarParameters, Target
from tess_vetter.report import build_vet_lc_summary_blocks


def _make_lc_data() -> LightCurveData:
    return LightCurveData(
        time=np.linspace(1800.0, 1810.0, 256, dtype=np.float64),
        flux=np.ones(256, dtype=np.float64),
        flux_err=np.full(256, 1e-4, dtype=np.float64),
        quality=np.zeros(256, dtype=np.int32),
        valid_mask=np.ones(256, dtype=bool),
        tic_id=123456789,
        sector=1,
        cadence_seconds=120.0,
    )


def test_public_vet_lc_summary_seam_returns_expected_blocks() -> None:
    client = MagicMock(spec=["download_all_sectors", "get_target_info"])
    client.download_all_sectors.return_value = [_make_lc_data()]
    client.get_target_info.return_value = Target(
        tic_id=123456789,
        stellar=StellarParameters(teff=5800.0, radius=1.0, mass=1.0),
    )
    result = generate_report(
        123456789,
        period_days=3.5,
        t0_btjd=1850.0,
        duration_hours=2.5,
        mast_client=client,
    )

    blocks = build_vet_lc_summary_blocks(result.report)
    assert isinstance(blocks["lc_summary"], dict)
    assert "snr" in blocks["lc_summary"]
    assert isinstance(blocks["noise_summary"], dict)
    assert isinstance(blocks["variability_summary"], dict)
