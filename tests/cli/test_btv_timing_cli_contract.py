from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

import bittr_tess_vetter.cli.timing_cli as timing_cli
from bittr_tess_vetter.cli.timing_cli import timing_command
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.transit.result import TransitTime


def _make_lc(*, tic_id: int, sector: int, start: float) -> LightCurveData:
    time = np.linspace(start, start + 1.0, 24, dtype=np.float64)
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.full_like(time, 1e-3, dtype=np.float64)
    quality = np.zeros(time.shape, dtype=np.int32)
    valid_mask = np.ones(time.shape, dtype=np.bool_)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=120.0,
    )


def test_btv_timing_success_contract_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 9.5, 2456.25, 2.75, 640.0, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {
                "tic_id": 123,
                "period_days": 9.5,
                "t0_btjd": 2456.25,
                "duration_hours": 2.75,
                "depth_ppm": 640.0,
            },
        }

    def _fake_load_lightcurves_with_sector_policy(**kwargs: Any):
        seen["download"] = kwargs
        return [
            _make_lc(tic_id=kwargs["tic_id"], sector=14, start=2000.0),
            _make_lc(tic_id=kwargs["tic_id"], sector=15, start=2010.0),
        ], "cache_only_explicit_sectors"

    def _fake_stitch(lightcurves: list[LightCurveData], *, tic_id: int):
        seen["stitch_called"] = True
        seen["stitch_tic_id"] = tic_id
        _ = lightcurves
        return _make_lc(tic_id=tic_id, sector=-1, start=2000.0), object()

    prealigned_transit_times = [
        TransitTime(
            epoch=0,
            tc=2456.25,
            tc_err=0.0003,
            depth_ppm=650.0,
            duration_hours=2.7,
            snr=8.1,
        ),
        TransitTime(
            epoch=1,
            tc=2465.75,
            tc_err=0.0004,
            depth_ppm=630.0,
            duration_hours=2.8,
            snr=7.8,
            is_outlier=True,
            outlier_reason="timing_outlier",
        ),
    ]

    def _fake_prealign_candidate(**kwargs: Any):
        seen["prealign_kwargs"] = kwargs
        return kwargs["candidate"], prealigned_transit_times, {
            "prealign_requested": True,
            "prealign_applied": False,
            "alignment_quality": "unchanged",
            "delta_t0_minutes": 0.0,
            "delta_period_ppm": 0.0,
            "n_transits_pre": 2,
            "n_transits_post": 2,
            "prealign_score_z": 3.1,
            "prealign_error": None,
        }, {
            "pre": {"attempted_epochs": 3, "accepted_epochs": 2, "reject_counts": {"snr_below_threshold": 1}},
            "post": {"attempted_epochs": 3, "accepted_epochs": 2, "reject_counts": {"snr_below_threshold": 1}},
            "selected": {"attempted_epochs": 3, "accepted_epochs": 2, "reject_counts": {"snr_below_threshold": 1}},
        }

    def _fake_analyze_ttvs(**kwargs: Any):
        seen["analyze_kwargs"] = kwargs

        class _Result:
            def to_dict(self) -> dict[str, Any]:
                return {
                    "n_transits": 2,
                    "rms_seconds": 74.2,
                    "periodicity_score": 2.6,
                    "periodicity_sigma": 2.6,
                    "linear_trend": None,
                }

        return _Result()

    def _fake_timing_series(**kwargs: Any):
        seen["series_kwargs"] = kwargs

        class _Series:
            def to_dict(self) -> dict[str, Any]:
                return {
                    "n_points": 2,
                    "rms_seconds": 74.2,
                    "periodicity_score": 2.6,
                    "periodicity_sigma": 2.6,
                    "linear_trend_sec_per_epoch": 0.0,
                    "points": [],
                }

        return _Series()

    monkeypatch.setattr("bittr_tess_vetter.cli.timing_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli.load_lightcurves_with_sector_policy",
        _fake_load_lightcurves_with_sector_policy,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.timing_cli.stitch_lightcurve_data", _fake_stitch)
    monkeypatch.setattr(timing_cli.api.timing, "analyze_ttvs", _fake_analyze_ttvs)
    monkeypatch.setattr(timing_cli.api.timing, "timing_series", _fake_timing_series)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli._prealign_candidate",
        _fake_prealign_candidate,
    )

    out_path = tmp_path / "timing.json"
    runner = CliRunner()
    result = runner.invoke(
        timing_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--flux-type",
            "sap",
            "--min-snr",
            "3.5",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["download"] == {
        "tic_id": 123,
        "flux_type": "sap",
        "sectors": [14, 15],
        "explicit_sectors": True,
        "network_ok": True,
        "cache_dir": None,
    }
    assert seen["stitch_called"] is True
    assert seen["stitch_tic_id"] == 123
    assert seen["prealign_kwargs"]["min_snr"] == 3.5
    assert seen["analyze_kwargs"]["period_days"] == 9.5
    assert seen["analyze_kwargs"]["t0_btjd"] == 2456.25
    assert seen["series_kwargs"]["min_snr"] == 3.5

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.timing.v1"
    assert payload["transit_times"][0]["epoch"] == 0
    assert payload["transit_times"][1]["outlier_reason"] == "timing_outlier"
    assert payload["ttv"]["rms_seconds"] == 74.2
    assert payload["timing_series"]["n_points"] == 2
    assert payload["alignment"]["n_transits_pre"] == 2
    assert payload["alignment"]["n_transits_post"] == 2
    assert payload["diagnostics"]["selected"]["accepted_epochs"] == 2
    assert payload["next_actions"][0]["code"] == "TIMING_MEASURABLE"
    assert payload["result"]["next_actions"][0]["code"] == "TIMING_MEASURABLE"
    assert payload["result"]["transit_times"] == payload["transit_times"]
    assert payload["result"]["ttv"] == payload["ttv"]
    assert payload["result"]["timing_series"] == payload["timing_series"]
    assert payload["result"]["alignment"] == payload["alignment"]
    assert payload["result"]["diagnostics"] == payload["diagnostics"]
    assert payload["result"]["next_actions"] == payload["next_actions"]
    assert "verdict" in payload
    assert "verdict_source" in payload
    assert payload["verdict"] == "TIMING_MEASURABLE"
    assert payload["verdict_source"] == "$.next_actions[0].code"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["inputs_summary"]["input_resolution"]["inputs"]["tic_id"] == 123
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"] == {
        "network_ok": True,
        "sectors": [14, 15],
        "flux_type": "sap",
        "min_snr": 3.5,
        "cache_dir": None,
        "prealign": True,
        "prealign_steps": 25,
        "prealign_lr": 0.05,
        "prealign_window_phase": 0.02,
    }
    assert payload["provenance"]["sector_load_path"] == "cache_only_explicit_sectors"


def test_btv_timing_missing_required_ephemeris_input_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        timing_command,
        [
            "--tic-id",
            "123",
        ],
    )

    assert result.exit_code == 1
    assert "Missing required inputs" in result.output


def test_prealign_candidate_recovers_transits_and_sets_drift_metadata(monkeypatch) -> None:
    lc = timing_cli.LightCurve(
        time=np.linspace(2000.0, 2010.0, 200, dtype=np.float64),
        flux=np.ones(200, dtype=np.float64),
        flux_err=np.full(200, 1e-3, dtype=np.float64),
    )
    candidate = timing_cli.Candidate(
        ephemeris=timing_cli.Ephemeris(
            period_days=1.06,
            t0_btjd=2277.0,
            duration_hours=3.2,
        ),
        depth_ppm=1000.0,
    )

    class _FakeRefinementResult:
        t0_refined_btjd = 2277.01
        duration_refined_hours = 3.1
        score_z = 4.2

    def _fake_refine_one_candidate_numpy(**_kwargs: Any):
        return _FakeRefinementResult()

    call_t0s: list[float] = []

    def _fake_measure_transit_times_with_diagnostics(**kwargs: Any):
        t0 = float(kwargs["candidate"].ephemeris.t0_btjd)
        call_t0s.append(t0)
        if len(call_t0s) == 1:
            return [], {"attempted_epochs": 4, "accepted_epochs": 0, "reject_counts": {"optimizer_failed": 4}}
        return [
            TransitTime(
                epoch=0,
                tc=2277.01,
                tc_err=0.0005,
                depth_ppm=980.0,
                duration_hours=3.1,
                snr=4.5,
            )
        ], {"attempted_epochs": 4, "accepted_epochs": 1, "reject_counts": {"snr_below_threshold": 3}}

    monkeypatch.setattr(
        timing_cli.ephemeris_refinement_api,
        "refine_one_candidate_numpy",
        _fake_refine_one_candidate_numpy,
    )
    monkeypatch.setattr(
        timing_cli.api.timing,
        "measure_transit_times_with_diagnostics",
        _fake_measure_transit_times_with_diagnostics,
    )

    candidate_out, transit_times_out, metadata, diagnostics = timing_cli._prealign_candidate(
        lc=lc,
        candidate=candidate,
        min_snr=2.0,
        prealign_enabled=True,
        prealign_steps=25,
        prealign_lr=0.05,
        prealign_window_phase=0.02,
    )

    assert len(call_t0s) == 2
    assert len(transit_times_out) == 1
    assert metadata["prealign_applied"] is True
    assert metadata["alignment_quality"] == "improved_transit_recovery"
    assert metadata["n_transits_pre"] == 0
    assert metadata["n_transits_post"] == 1
    assert metadata["delta_t0_minutes"] > 0
    assert metadata["delta_period_ppm"] == 0.0
    assert diagnostics["selected"]["accepted_epochs"] == 1
    assert candidate_out.ephemeris.t0_btjd == 2277.01


def test_build_next_actions_noise_limited_branch() -> None:
    actions = timing_cli._build_next_actions(
        alignment_metadata={
            "alignment_quality": "unchanged",
        },
        measurement_diagnostics={
            "selected": {
                "attempted_epochs": 10,
                "accepted_epochs": 0,
                "reject_counts": {"snr_below_threshold": 10},
            }
        },
        min_snr=2.0,
    )
    assert actions[0]["code"] == "NOISE_LIMITED"
    assert "--min-snr 1.5" in actions[0]["guidance"]


def test_build_next_actions_alignment_review_branch() -> None:
    actions = timing_cli._build_next_actions(
        alignment_metadata={
            "alignment_quality": "degraded_rejected",
        },
        measurement_diagnostics={
            "selected": {
                "attempted_epochs": 10,
                "accepted_epochs": 0,
                "reject_counts": {"insufficient_points": 10},
            }
        },
    )
    codes = [a["code"] for a in actions]
    assert "DATA_LIMITED" in codes
    assert "ALIGNMENT_REVIEW" in codes


def test_btv_timing_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, 9.5, 2456.25, 2.75, 640.0, {"source": "toi", "inputs": {"toi": kwargs.get("toi")}}

    monkeypatch.setattr("bittr_tess_vetter.cli.timing_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli.load_lightcurves_with_sector_policy",
        lambda **kwargs: ([_make_lc(tic_id=kwargs["tic_id"], sector=14, start=2000.0)], "mast_discovery"),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli._prealign_candidate",
        lambda **kwargs: (
            kwargs["candidate"],
            [],
            {
                "prealign_requested": False,
                "prealign_applied": False,
                "alignment_quality": "disabled",
                "delta_t0_minutes": 0.0,
                "delta_period_ppm": 0.0,
                "n_transits_pre": 0,
                "n_transits_post": 0,
                "prealign_score_z": None,
                "prealign_error": None,
            },
            {"selected": {"attempted_epochs": 0, "accepted_epochs": 0, "reject_counts": {}}},
        ),
    )
    monkeypatch.setattr(
        timing_cli.api.timing,
        "analyze_ttvs",
        lambda **_kwargs: type("R", (), {"to_dict": lambda self: {"n_transits": 0}})(),
    )
    monkeypatch.setattr(
        timing_cli.api.timing,
        "timing_series",
        lambda **_kwargs: type("S", (), {"to_dict": lambda self: {"n_points": 0}})(),
    )

    out_path = tmp_path / "timing_positional.json"
    runner = CliRunner()
    result = runner.invoke(timing_command, ["TOI-5807.01", "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"


def test_btv_timing_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "timing.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 987,
                        "ephemeris": {
                            "period_days": 3.2,
                            "t0_btjd": 2100.0,
                            "duration_hours": 1.9,
                        },
                        "input_depth_ppm": 220.0,
                    },
                    "provenance": {"sectors_used": [40]},
                }
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli._resolve_candidate_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli.load_lightcurves_with_sector_policy",
        lambda **kwargs: (
            [(_make_lc(tic_id=kwargs["tic_id"], sector=40, start=2000.0))],
            "mast_filtered",
        ),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli._prealign_candidate",
        lambda **kwargs: (
            kwargs["candidate"],
            [],
            {
                "prealign_requested": True,
                "prealign_applied": False,
                "alignment_quality": "unchanged",
                "delta_t0_minutes": 0.0,
                "delta_period_ppm": 0.0,
                "n_transits_pre": 0,
                "n_transits_post": 0,
                "prealign_score_z": None,
                "prealign_error": None,
            },
            {"selected": {"attempted_epochs": 0, "accepted_epochs": 0, "reject_counts": {}}},
        ),
    )
    monkeypatch.setattr(timing_cli.api.timing, "analyze_ttvs", lambda **_kwargs: type("R", (), {"to_dict": lambda self: {}})())
    monkeypatch.setattr(timing_cli.api.timing, "timing_series", lambda **_kwargs: type("S", (), {"to_dict": lambda self: {}})())

    out_path = tmp_path / "timing_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        timing_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-987.01",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())


def test_btv_timing_explicit_sectors_cache_miss_exits_4(monkeypatch) -> None:
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.timing_cli.load_lightcurves_with_sector_policy",
        lambda **_kwargs: (_ for _ in ()).throw(
            timing_cli.BtvCliError(
                "Cache-only sector load failed for TIC 123. Missing cached light curve for sector(s): 14.",
                exit_code=4,
            )
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        timing_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "9.5",
            "--t0-btjd",
            "2456.25",
            "--duration-hours",
            "2.75",
            "--sectors",
            "14",
        ],
    )
    assert result.exit_code == 4
    assert "Cache-only sector load failed for TIC 123" in result.output


def test_btv_timing_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(
        timing_command,
        ["TOI-5807.01", "--toi", "TOI-4510.01"],
    )
    assert result.exit_code == 1
    assert "must match" in result.output
