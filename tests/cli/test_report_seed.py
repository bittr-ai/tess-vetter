from __future__ import annotations

import pytest

from bittr_tess_vetter.cli.common_cli import BtvCliError
from bittr_tess_vetter.cli.report_seed import ReportSeed, resolve_candidate_inputs_with_report_seed


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


def test_resolve_candidate_inputs_with_report_seed_uses_manual_inputs_without_toi_lookup(monkeypatch) -> None:
    def _should_not_resolve(_toi: str):
        raise AssertionError("TOI lookup should not run when manual candidate inputs are complete")

    monkeypatch.setattr("bittr_tess_vetter.cli.report_seed.resolve_toi_to_tic_ephemeris_depth", _should_not_resolve)

    tic_id, period_days, t0_btjd, duration_hours, depth_ppm, input_resolution = resolve_candidate_inputs_with_report_seed(
        network_ok=False,
        toi="TOI-7182.01",
        tic_id=123456,
        period_days=12.3,
        t0_btjd=2450.5,
        duration_hours=2.1,
        depth_ppm=456.0,
        report_seed=None,
    )

    assert tic_id == 123456
    assert period_days == 12.3
    assert t0_btjd == 2450.5
    assert duration_hours == 2.1
    assert depth_ppm == 456.0
    assert input_resolution["source"] == "cli"
    assert input_resolution["resolved_from"] == "cli"


def test_resolve_candidate_inputs_with_report_seed_uses_report_seed_without_toi_lookup(monkeypatch) -> None:
    def _should_not_resolve(_toi: str):
        raise AssertionError("TOI lookup should not run when report seed already provides candidate inputs")

    monkeypatch.setattr("bittr_tess_vetter.cli.report_seed.resolve_toi_to_tic_ephemeris_depth", _should_not_resolve)

    seed = ReportSeed(
        tic_id=999,
        period_days=9.9,
        t0_btjd=2000.0,
        duration_hours=3.3,
        depth_ppm=210.0,
        sectors_used=[14, 15],
        toi="TOI-999.01",
        source_path="/tmp/report.json",
    )
    tic_id, period_days, t0_btjd, duration_hours, depth_ppm, input_resolution = resolve_candidate_inputs_with_report_seed(
        network_ok=False,
        toi="TOI-999.01",
        tic_id=None,
        period_days=None,
        t0_btjd=None,
        duration_hours=None,
        depth_ppm=None,
        report_seed=seed,
    )

    assert tic_id == 999
    assert period_days == 9.9
    assert t0_btjd == 2000.0
    assert duration_hours == 3.3
    assert depth_ppm == 210.0
    assert input_resolution["source"] == "report_file"
    assert input_resolution["resolved_from"] == "report_file"
