"""Tests for the generate_report convenience API.

All 12 test cases from the spec (Section 13), with MASTClient mocked
to avoid network calls.
"""

from __future__ import annotations

import inspect
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import bittr_tess_vetter.api.generate_report as generate_report_api
from bittr_tess_vetter.api.generate_report import (
    EnrichmentConfig,
    GenerateReportResult,
    generate_report,
)
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.domain.target import StellarParameters, Target
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError
from bittr_tess_vetter.validation.result_schema import CheckResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPH = {"period_days": 3.5, "t0_btjd": 1850.0, "duration_hours": 2.5}


def _extract_include_v03_state(report_json: dict[str, object]) -> tuple[bool, bool, str | None]:
    summary = report_json["summary"]
    assert isinstance(summary, dict)
    check_execution = summary.get("check_execution")
    assert isinstance(check_execution, dict)
    if "include_v03" in check_execution:
        include_v03 = check_execution["include_v03"]
        assert isinstance(include_v03, dict)
        return (
            bool(include_v03.get("requested")),
            bool(include_v03.get("enabled")),
            include_v03.get("reason"),
        )
    return (
        bool(check_execution.get("v03_requested")),
        bool(check_execution.get("v03_enabled")),
        check_execution.get("v03_disabled_reason"),
    )


def _make_lc_data(sector: int, n: int = 500, tic_id: int = 123456789) -> LightCurveData:
    """Create a minimal but valid LightCurveData for testing."""
    rng = np.random.default_rng(sector)
    time = np.linspace(1800.0 + sector * 30, 1800.0 + (sector + 1) * 30, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0, 1e-4, n)
    flux_err = np.full(n, 1e-4, dtype=np.float64)
    quality = np.zeros(n, dtype=np.int32)
    valid_mask = np.ones(n, dtype=np.bool_)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=tic_id,
        sector=sector,
        cadence_seconds=120.0,
    )


def _mock_client(
    sectors: list[int] | None = None,
    *,
    stellar: StellarParameters | None = None,
    get_target_raises: bool = False,
    download_raises: type[Exception] | None = None,
) -> MagicMock:
    """Build a mock MASTClient with configurable behaviour."""
    client = MagicMock(spec=["download_all_sectors", "get_target_info"])

    if download_raises is not None:
        client.download_all_sectors.side_effect = download_raises("boom")
    elif sectors is not None:
        client.download_all_sectors.return_value = [_make_lc_data(s) for s in sectors]
    else:
        client.download_all_sectors.return_value = []

    if get_target_raises:
        client.get_target_info.side_effect = RuntimeError("TIC unavailable")
    else:
        target = Target(
            tic_id=123456789,
            stellar=stellar or StellarParameters(teff=5800.0, radius=1.0, mass=1.0),
        )
        client.get_target_info.return_value = target

    return client


def _mock_vet_artifact(check_ids: list[str]) -> dict[str, object]:
    return {
        "results": [
            {
                "id": check_id,
                "name": f"check {check_id}",
                "status": "ok",
                "confidence": 0.5,
                "metrics": {},
                "flags": [],
                "notes": [],
                "provenance": {},
                "raw": None,
            }
            for check_id in check_ids
        ],
        "warnings": [],
        "provenance": {"pipeline_version": "test"},
        "inputs_summary": {},
    }


# ---------------------------------------------------------------------------
# 1. Happy path, multi-sector
# ---------------------------------------------------------------------------
def test_happy_path_multi_sector() -> None:
    client = _mock_client(sectors=[1, 2, 3])
    result = generate_report(123456789, **_EPH, mast_client=client)

    assert isinstance(result, GenerateReportResult)
    assert result.sectors_used == [1, 2, 3]
    assert result.stitch_diagnostics is not None
    assert len(result.stitch_diagnostics) > 0
    assert isinstance(result.report_json, dict)
    assert "plot_data" not in result.report_json
    assert isinstance(result.plot_data_json, dict)
    assert "full_lc" in result.plot_data_json


# ---------------------------------------------------------------------------
# 2. Happy path, single sector
# ---------------------------------------------------------------------------
def test_happy_path_single_sector() -> None:
    client = _mock_client(sectors=[5])
    result = generate_report(123456789, **_EPH, mast_client=client)

    assert result.sectors_used == [5]
    assert result.stitch_diagnostics is None
    assert isinstance(result.report_json, dict)


# ---------------------------------------------------------------------------
# 3. No sectors found
# ---------------------------------------------------------------------------
def test_no_sectors_raises() -> None:
    client = _mock_client(download_raises=LightCurveNotFoundError)
    with pytest.raises(LightCurveNotFoundError):
        generate_report(123456789, **_EPH, mast_client=client)


# ---------------------------------------------------------------------------
# 4. Stellar auto-fetch succeeds
# ---------------------------------------------------------------------------
def test_stellar_auto_fetch_succeeds() -> None:
    stellar = StellarParameters(teff=5800.0, radius=1.1, mass=1.05)
    client = _mock_client(sectors=[1], stellar=stellar)

    result = generate_report(123456789, **_EPH, mast_client=client)

    # Verify get_target_info was called (auto-fetch path)
    client.get_target_info.assert_called_once_with(123456789)
    assert result.report is not None


# ---------------------------------------------------------------------------
# 5. Stellar auto-fetch fails
# ---------------------------------------------------------------------------
def test_stellar_auto_fetch_fails_gracefully() -> None:
    client = _mock_client(sectors=[1], get_target_raises=True)
    # Should not raise; proceeds with stellar=None
    result = generate_report(123456789, **_EPH, mast_client=client)
    assert result.report is not None


# ---------------------------------------------------------------------------
# 6. include_html=True
# ---------------------------------------------------------------------------
def test_include_html_true() -> None:
    client = _mock_client(sectors=[1])
    result = generate_report(123456789, **_EPH, mast_client=client, include_html=True)
    assert result.html is not None
    assert isinstance(result.html, str)
    assert len(result.html) > 0


# ---------------------------------------------------------------------------
# 7. include_html=False (default)
# ---------------------------------------------------------------------------
def test_include_html_false() -> None:
    client = _mock_client(sectors=[1])
    result = generate_report(123456789, **_EPH, mast_client=client)
    assert result.html is None


def test_include_v03_execution_state_enabled_with_stellar() -> None:
    client = _mock_client(sectors=[1])
    explicit_stellar = StellarParameters(teff=5800.0, radius=1.1, mass=1.0)

    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_v03=True,
        stellar=explicit_stellar,
    )
    assert _extract_include_v03_state(result.report_json) == (True, True, None)


def test_include_v03_execution_state_disabled_without_stellar() -> None:
    client = _mock_client(sectors=[1], get_target_raises=True)

    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_v03=True,
    )
    requested, enabled, reason = _extract_include_v03_state(result.report_json)
    assert requested is True
    assert enabled is False
    assert isinstance(reason, str) and reason


def test_include_v03_execution_state_default_false() -> None:
    client = _mock_client(sectors=[1])
    result = generate_report(123456789, **_EPH, mast_client=client, include_v03=False)
    requested, enabled, reason = _extract_include_v03_state(result.report_json)
    assert requested is False
    assert enabled is False
    assert reason is None or isinstance(reason, str)


def test_deterministic_budget_kwargs_forwarded_to_build_report(monkeypatch) -> None:
    client = _mock_client(sectors=[1])
    real_build_report = generate_report_api.build_report
    captured: dict[str, object] = {}

    def _spy_build_report(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_build_report(*args, **kwargs)

    monkeypatch.setattr(generate_report_api, "build_report", _spy_build_report)
    generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_additional_plots=False,
        max_transit_windows=12,
        max_points_per_window=123,
        max_timing_points=77,
        include_lc_robustness=False,
        max_lc_robustness_epochs=55,
    )

    assert captured["include_additional_plots"] is False
    assert captured["max_transit_windows"] == 12
    assert captured["max_points_per_window"] == 123
    assert captured["max_timing_points"] == 77
    assert captured["include_lc_robustness"] is False
    assert captured["max_lc_robustness_epochs"] == 55


def test_deterministic_budget_defaults_match_build_report() -> None:
    generate_sig = inspect.signature(generate_report_api.generate_report)
    build_sig = inspect.signature(generate_report_api.build_report)
    params = (
        "include_additional_plots",
        "max_transit_windows",
        "max_points_per_window",
        "max_timing_points",
        "include_lc_robustness",
        "max_lc_robustness_epochs",
    )

    for name in params:
        assert generate_sig.parameters[name].default == build_sig.parameters[name].default


def test_invalid_deterministic_budget_propagates() -> None:
    client = _mock_client(sectors=[1])
    with pytest.raises(ValueError, match="max_transit_windows"):
        generate_report(
            123456789,
            **_EPH,
            mast_client=client,
            max_transit_windows=0,
        )


def test_custom_views_forwarded_to_build_report_when_supported(monkeypatch) -> None:
    client = _mock_client(sectors=[1])
    real_build_report = generate_report_api.build_report
    captured: dict[str, object] = {}

    def _spy_build_report(*args, custom_views=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["custom_views"] = custom_views
        return real_build_report(*args, custom_views=custom_views, **kwargs)

    monkeypatch.setattr(generate_report_api, "build_report", _spy_build_report)
    custom_views = {
        "version": "1",
        "views": [
            {
                "id": "cv1",
                "title": "Custom 1",
                "producer": {"source": "agent"},
                "mode": "deterministic",
                "chart": {
                    "type": "scatter",
                    "series": [
                        {
                            "x": {"path": "/plot_data/full_lc/time"},
                            "y": {"path": "/plot_data/full_lc/flux"},
                        }
                    ],
                    "options": {},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            }
        ],
    }
    generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        custom_views=custom_views,
    )

    assert captured["custom_views"] == custom_views


def test_custom_views_added_to_report_json_and_html() -> None:
    client = _mock_client(sectors=[1])
    custom_views = {
        "version": "1",
        "views": [
            {
                "id": "cv_scatter",
                "title": "Custom Scatter",
                "producer": {"source": "agent"},
                "mode": "deterministic",
                "chart": {
                    "type": "scatter",
                    "series": [
                        {
                            "label": "LC",
                            "x": {"path": "/plot_data/full_lc/time"},
                            "y": {"path": "/plot_data/full_lc/flux"},
                        }
                    ],
                    "options": {"x_label": "Time", "y_label": "Flux"},
                },
                "quality": {"min_points_required": 3, "status": "ok", "flags": []},
            }
        ],
    }

    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_html=True,
        custom_views=custom_views,
    )

    assert result.report_json["custom_views"]["version"] == "1"
    assert len(result.report_json["custom_views"]["views"]) == 1
    assert result.report_json["custom_views"]["views"][0]["id"] == "cv_scatter"
    assert result.html is not None
    assert "Custom Views" in result.html
    assert "Custom Scatter" in result.html


def test_custom_views_default_unchanged_when_not_passed() -> None:
    client = _mock_client(sectors=[1])
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
    )
    assert result.report_json["custom_views"] == {"version": "1", "views": []}


# ---------------------------------------------------------------------------
# 8. mast_client injection
# ---------------------------------------------------------------------------
def test_mast_client_injection() -> None:
    client = _mock_client(sectors=[1])
    generate_report(123456789, **_EPH, mast_client=client)
    # Verify the injected client was used (not a new one)
    client.download_all_sectors.assert_called_once()


# ---------------------------------------------------------------------------
# 9. stellar kwarg provided skips TIC query
# ---------------------------------------------------------------------------
def test_stellar_kwarg_skips_tic_query() -> None:
    client = _mock_client(sectors=[1])
    explicit_stellar = StellarParameters(teff=4000.0, radius=0.5, mass=0.5)

    result = generate_report(
        123456789, **_EPH, mast_client=client, stellar=explicit_stellar
    )

    # get_target_info should NOT have been called
    client.get_target_info.assert_not_called()
    assert result.report is not None


# ---------------------------------------------------------------------------
# 10. flux_type forwarded
# ---------------------------------------------------------------------------
def test_flux_type_forwarded() -> None:
    client = _mock_client(sectors=[1])
    generate_report(123456789, **_EPH, mast_client=client, flux_type="sap")
    call_kwargs = client.download_all_sectors.call_args
    assert call_kwargs[1]["flux_type"] == "sap" or call_kwargs[0][1] == "sap"


# ---------------------------------------------------------------------------
# 11. Duplicate sectors sanitized
# ---------------------------------------------------------------------------
def test_duplicate_sectors_sanitized() -> None:
    client = _mock_client(sectors=[5, 6])
    generate_report(
        123456789, **_EPH, mast_client=client, sectors=[5, 5, 6]
    )
    call_kwargs = client.download_all_sectors.call_args
    # sectors should be deduplicated and sorted
    assert call_kwargs[1]["sectors"] == [5, 6]


# ---------------------------------------------------------------------------
# 12. Invalid flux_type propagates ValueError
# ---------------------------------------------------------------------------
def test_invalid_flux_type_propagates() -> None:
    client = MagicMock(spec=["download_all_sectors", "get_target_info"])
    client.download_all_sectors.side_effect = ValueError(
        "flux_type must be 'pdcsap' or 'sap', got 'invalid'"
    )
    with pytest.raises(ValueError, match="flux_type"):
        generate_report(123456789, **_EPH, mast_client=client, flux_type="invalid")


def test_include_enrichment_false_omits_enrichment_block() -> None:
    """Default behavior should not attach enrichment payload."""
    client = _mock_client(sectors=[1])
    result = generate_report(123456789, **_EPH, mast_client=client)
    assert result.report.enrichment is None
    assert "enrichment" not in result.report_json["summary"]


def test_include_enrichment_true_adds_blocks() -> None:
    """Enrichment-enabled path returns domain blocks with deterministic statuses."""
    client = _mock_client(sectors=[1])
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
    )

    assert result.report.enrichment is not None
    e = result.report_json["summary"]["enrichment"]
    assert e["version"] == "0.1.0"
    assert e["pixel_diagnostics"]["status"] == "skipped"
    assert e["catalog_context"]["status"] == "skipped"
    assert e["followup_context"]["status"] in {"ok", "skipped"}
    assert isinstance(e["pixel_diagnostics"]["flags"], list)


def test_include_enrichment_respects_block_toggles() -> None:
    """Config toggles should omit disabled enrichment blocks."""
    client = _mock_client(sectors=[1])
    cfg = EnrichmentConfig(
        include_pixel_diagnostics=False,
        include_catalog_context=True,
        include_followup_context=False,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )

    e = result.report_json["summary"]["enrichment"]
    assert e["pixel_diagnostics"] is None
    assert e["catalog_context"]["status"] == "skipped"
    assert e["followup_context"] is None


def test_enrichment_budget_knobs_are_passed_through_to_catalog_provenance() -> None:
    """Budget knobs should be surfaced in catalog_context provenance."""
    client = _mock_client(sectors=[1])
    cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=False,
        include_followup_context=False,
        network=False,
        max_catalog_rows=7,
        per_request_timeout_seconds=1.25,
        max_network_seconds=9.5,
        max_concurrent_requests=2,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    catalog = result.report_json["summary"]["enrichment"]["catalog_context"]
    budget = catalog["provenance"]["budget"]
    network_cfg = catalog["provenance"]["network_config"]
    assert budget["max_catalog_rows"] == 7
    assert network_cfg["per_request_timeout_seconds"] == 1.25
    assert network_cfg["max_network_seconds"] == 9.5
    assert network_cfg["max_concurrent_requests"] == 2


def test_report_reuses_complete_vet_artifact_without_incremental_checks(monkeypatch) -> None:
    client = _mock_client(sectors=[1])

    def _no_incremental(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("incremental checks should not run for complete artifact")

    monkeypatch.setattr("bittr_tess_vetter.api.report_vet_reuse.run_lc_checks", _no_incremental)
    artifact = _mock_vet_artifact(["V01", "V02", "V04", "V05", "V13", "V15"])
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        vet_result=artifact,
    )

    assert result.vet_artifact_reuse is not None
    assert result.vet_artifact_reuse.missing_fields == []
    assert result.vet_artifact_reuse.incremental_compute == []
    assert result.vet_artifact_reuse.reused is True


def test_report_partial_vet_artifact_runs_only_missing_checks(monkeypatch) -> None:
    client = _mock_client(sectors=[1])
    captured: dict[str, object] = {}

    def _only_missing(_lightcurve, **kwargs):  # type: ignore[no-untyped-def]
        captured["enabled"] = kwargs["enabled"]
        return [
            CheckResult(
                id="V13",
                name="check V13",
                status="ok",
                confidence=0.7,
                metrics={},
                flags=[],
                notes=[],
                provenance={},
                raw=None,
            )
        ]

    monkeypatch.setattr("bittr_tess_vetter.api.report_vet_reuse.run_lc_checks", _only_missing)
    artifact = _mock_vet_artifact(["V01", "V02", "V04", "V05", "V15"])
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        vet_result=artifact,
    )

    assert captured["enabled"] == {"V13"}
    assert result.vet_artifact_reuse is not None
    assert result.vet_artifact_reuse.missing_fields == ["checks.V13"]
    assert result.vet_artifact_reuse.incremental_compute == ["compute_check.V13"]
    assert "V13" in result.report_json["summary"]["checks"]


def test_enrichment_fail_open_true_recovers_block_errors(monkeypatch) -> None:
    """With fail_open=True, block errors should not abort report generation."""
    client = _mock_client(sectors=[1])

    def _boom(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _boom,
    )
    cfg = EnrichmentConfig(fail_open=True, include_catalog_context=True)
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    e = result.report_json["summary"]["enrichment"]
    assert e["catalog_context"]["status"] == "error"


def test_enrichment_fail_open_false_raises(monkeypatch) -> None:
    """With fail_open=False, block errors should abort generation."""
    client = _mock_client(sectors=[1])

    def _boom(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _boom,
    )
    cfg = EnrichmentConfig(fail_open=False, include_catalog_context=True)
    with pytest.raises(RuntimeError, match="boom"):
        generate_report(
            123456789,
            **_EPH,
            mast_client=client,
            include_enrichment=True,
            enrichment_config=cfg,
        )


def test_select_tpf_best_prefers_sector_with_higher_transit_coverage() -> None:
    """Best strategy should score sectors from per-sector time arrays when provided."""
    lc_api = LightCurve(time=np.array([0.0, 1.0]), flux=np.array([1.0, 1.0]))
    candidate_api = Candidate(
        ephemeris=Ephemeris(period_days=2.0, t0_btjd=10.0, duration_hours=12.0)
    )
    sector_times = {
        1: np.array([11.0, 13.0], dtype=np.float64),  # out-of-transit
        2: np.array([10.0, 12.0], dtype=np.float64),  # in-transit centers
    }

    selected = generate_report_api._select_tpf_sectors(
        strategy="best",
        sectors_used=[1, 2],
        requested=None,
        lc_api=lc_api,
        candidate_api=candidate_api,
        sector_times=sector_times,
    )
    assert selected == [2]


def test_pixel_point_budget_enforced_before_pixel_checks(monkeypatch) -> None:
    """Pixel checks should be skipped when TPF exceeds max_pixel_points budget."""
    client = MagicMock()
    client.download_all_sectors.return_value = [_make_lc_data(1)]
    client.get_target_info.return_value = Target(
        tic_id=123456789,
        stellar=StellarParameters(teff=5800.0, radius=1.0, mass=1.0),
    )
    client.download_tpf_cached.return_value = (
        np.array([1.0, 2.0], dtype=np.float64),
        np.ones((2, 3, 3), dtype=np.float64),  # 18 points
        None,
        None,
        np.ones((3, 3), dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    )

    def _boom(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("pixel check tier should not run when budget exceeded")

    monkeypatch.setattr("bittr_tess_vetter.api.generate_report._tier_bundle", _boom)
    cfg = EnrichmentConfig(
        include_pixel_diagnostics=True,
        include_catalog_context=False,
        include_followup_context=False,
        max_pixel_points=4,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    pixel = result.report_json["summary"]["enrichment"]["pixel_diagnostics"]
    assert pixel["status"] == "skipped"
    assert "PIXEL_POINT_BUDGET_EXCEEDED" in pixel["flags"]
    assert pixel["provenance"]["budget"]["budget_applied"] is True
    assert "sector_scores" in pixel["provenance"]


def test_pixel_point_budget_downsamples_when_feasible(monkeypatch) -> None:
    """Oversized TPF should be downsampled and still execute pixel checks when feasible."""
    client = MagicMock()
    client.download_all_sectors.return_value = [_make_lc_data(1)]
    client.get_target_info.return_value = Target(
        tic_id=123456789,
        stellar=StellarParameters(teff=5800.0, radius=1.0, mass=1.0),
    )
    client.download_tpf_cached.return_value = (
        np.arange(500, dtype=np.float64),
        np.ones((500, 3, 3), dtype=np.float64),  # 4500 points
        None,
        None,
        np.ones((3, 3), dtype=np.int32),
        np.zeros(500, dtype=np.int32),
    )

    captured: dict[str, object] = {}

    def _fake_tier_bundle(**kwargs):  # type: ignore[no-untyped-def]
        captured["tpf_shape"] = kwargs["tpf"].shape
        return generate_report_api.VettingBundleResult.from_checks([])

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._tier_bundle",
        _fake_tier_bundle,
    )
    cfg = EnrichmentConfig(
        include_pixel_diagnostics=True,
        include_catalog_context=False,
        include_followup_context=False,
        max_pixel_points=2_000,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    pixel = result.report_json["summary"]["enrichment"]["pixel_diagnostics"]
    assert "PIXEL_POINT_BUDGET_EXCEEDED" not in pixel["flags"]
    assert pixel["payload"]["downsample_applied"] is True
    assert pixel["payload"]["downsample_stride"] > 1
    assert captured["tpf_shape"] == (167, 3, 3)


def test_enrichment_uses_explicit_stellar_when_tic_lookup_skipped(monkeypatch) -> None:
    """Catalog context should receive explicit stellar kwarg even without target metadata."""
    client = _mock_client(sectors=[1])
    explicit_stellar = StellarParameters(teff=4100.0, radius=0.6, mass=0.6)
    captured: dict[str, object] = {}

    def _fake_catalog_context(**kwargs):  # type: ignore[no-untyped-def]
        captured["stellar"] = kwargs["stellar"]
        return generate_report_api._skipped_enrichment_block("CATALOG_TEST")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _fake_catalog_context,
    )
    cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=False,
        include_followup_context=False,
    )
    generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        stellar=explicit_stellar,
        include_enrichment=True,
        enrichment_config=cfg,
    )

    assert captured["stellar"] is explicit_stellar
    client.get_target_info.assert_not_called()


def test_enrichment_timeout_sets_timeout_flag_when_fail_open_true(monkeypatch) -> None:
    """Timed-out enrichment blocks should return error with ENRICHMENT_TIMEOUT."""
    client = _mock_client(sectors=[1])

    def _slow_catalog(**kwargs):  # type: ignore[no-untyped-def]
        import time as _time

        _time.sleep(0.2)
        return generate_report_api._skipped_enrichment_block("SLOW_CATALOG")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _slow_catalog,
    )
    cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=False,
        include_followup_context=False,
        fail_open=True,
        per_request_timeout_seconds=0.01,
    )
    started = time.monotonic()
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    elapsed = time.monotonic() - started
    catalog = result.report_json["summary"]["enrichment"]["catalog_context"]
    assert catalog["status"] == "error"
    assert "ENRICHMENT_TIMEOUT" in catalog["flags"]
    assert elapsed < 0.1


def test_enrichment_total_budget_exhaustion_skips_remaining_blocks(monkeypatch) -> None:
    """When total enrichment budget is exhausted, later blocks are skipped deterministically."""
    client = _mock_client(sectors=[1])

    def _slow_catalog(**kwargs):  # type: ignore[no-untyped-def]
        import time as _time

        _time.sleep(0.02)
        return generate_report_api._skipped_enrichment_block("CATALOG_DONE")

    def _slow_pixel(**kwargs):  # type: ignore[no-untyped-def]
        import time as _time

        _time.sleep(0.02)
        return generate_report_api._skipped_enrichment_block("PIXEL_DONE")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _slow_catalog,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_pixel_diagnostics",
        _slow_pixel,
    )
    cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=True,
        include_followup_context=True,
        fail_open=True,
        max_network_seconds=0.001,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    enrichment = result.report_json["summary"]["enrichment"]
    assert enrichment["pixel_diagnostics"]["status"] == "skipped"
    assert "NETWORK_BUDGET_EXHAUSTED" in enrichment["pixel_diagnostics"]["flags"]
    assert enrichment["followup_context"]["status"] == "skipped"
    assert "NETWORK_BUDGET_EXHAUSTED" in enrichment["followup_context"]["flags"]


def test_enrichment_honors_max_concurrent_requests(monkeypatch) -> None:
    """Catalog+pixel blocks should run serially/concurrently per max_concurrent_requests."""
    client = _mock_client(sectors=[1])

    def _slow_catalog(**kwargs):  # type: ignore[no-untyped-def]
        import time as _time

        _time.sleep(0.08)
        return generate_report_api._skipped_enrichment_block("CATALOG_DONE")

    def _slow_pixel(**kwargs):  # type: ignore[no-untyped-def]
        import time as _time

        _time.sleep(0.08)
        return generate_report_api._skipped_enrichment_block("PIXEL_DONE")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _slow_catalog,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_pixel_diagnostics",
        _slow_pixel,
    )

    serial_cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=True,
        include_followup_context=False,
        max_network_seconds=1.0,
        per_request_timeout_seconds=1.0,
        max_concurrent_requests=1,
    )
    started = time.monotonic()
    generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=serial_cfg,
    )
    serial_elapsed = time.monotonic() - started

    parallel_cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=True,
        include_followup_context=False,
        max_network_seconds=1.0,
        per_request_timeout_seconds=1.0,
        max_concurrent_requests=2,
    )
    started = time.monotonic()
    generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=parallel_cfg,
    )
    parallel_elapsed = time.monotonic() - started

    assert serial_elapsed > 0.12
    assert parallel_elapsed < serial_elapsed - 0.03


def test_catalog_context_runs_v07_only() -> None:
    """Inline report enrichment should exclude V06 and run V07 only."""
    client = _mock_client(sectors=[1])
    cfg = EnrichmentConfig(
        include_catalog_context=True,
        include_pixel_diagnostics=False,
        include_followup_context=False,
        network=False,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    catalog = result.report_json["summary"]["enrichment"]["catalog_context"]
    check_ids = {c["id"] for c in catalog["payload"]["checks_summary"]}
    assert check_ids == {"V07"}
