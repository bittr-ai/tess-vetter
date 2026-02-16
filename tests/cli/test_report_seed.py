from __future__ import annotations

import pytest

from bittr_tess_vetter.cli.common_cli import BtvCliError
from bittr_tess_vetter.cli.report_seed import resolve_candidate_inputs_with_report_seed


def test_resolve_candidate_inputs_with_report_seed_toi_no_network_requires_resolution_inputs() -> None:
    with pytest.raises(BtvCliError) as excinfo:
        resolve_candidate_inputs_with_report_seed(
            network_ok=False,
            toi="TOI-7182.01",
            tic_id=None,
            period_days=None,
            t0_btjd=None,
            duration_hours=None,
            depth_ppm=None,
            report_seed=None,
        )

    assert "--toi requires --network-ok to resolve TIC/ephemeris" in str(excinfo.value)
