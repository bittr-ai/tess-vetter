"""Tests for the generate_report convenience API.

All 12 test cases from the spec (Section 13), with MASTClient mocked
to avoid network calls.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPH = {"period_days": 3.5, "t0_btjd": 1850.0, "duration_hours": 2.5}


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
    assert "enrichment" not in result.report_json


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
    e = result.report_json["enrichment"]
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

    e = result.report_json["enrichment"]
    assert e["pixel_diagnostics"] is None
    assert e["catalog_context"]["status"] == "skipped"
    assert e["followup_context"] is None


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
    e = result.report_json["enrichment"]
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
    pixel = result.report_json["enrichment"]["pixel_diagnostics"]
    assert pixel["status"] == "skipped"
    assert "PIXEL_POINT_BUDGET_EXCEEDED" in pixel["flags"]
    assert pixel["provenance"]["budget"]["budget_applied"] is True
    assert "sector_scores" in pixel["provenance"]


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

        _time.sleep(0.05)
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
        per_request_timeout_seconds=0.001,
    )
    result = generate_report(
        123456789,
        **_EPH,
        mast_client=client,
        include_enrichment=True,
        enrichment_config=cfg,
    )
    catalog = result.report_json["enrichment"]["catalog_context"]
    assert catalog["status"] == "error"
    assert "ENRICHMENT_TIMEOUT" in catalog["flags"]


def test_enrichment_total_budget_exhaustion_skips_remaining_blocks(monkeypatch) -> None:
    """When total enrichment budget is exhausted, later blocks are skipped deterministically."""
    client = _mock_client(sectors=[1])

    def _slow_catalog(**kwargs):  # type: ignore[no-untyped-def]
        import time as _time

        _time.sleep(0.02)
        return generate_report_api._skipped_enrichment_block("CATALOG_DONE")

    monkeypatch.setattr(
        "bittr_tess_vetter.api.generate_report._run_catalog_context",
        _slow_catalog,
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
    enrichment = result.report_json["enrichment"]
    assert enrichment["pixel_diagnostics"]["status"] == "skipped"
    assert "NETWORK_BUDGET_EXHAUSTED" in enrichment["pixel_diagnostics"]["flags"]
    assert enrichment["followup_context"]["status"] == "skipped"
    assert "NETWORK_BUDGET_EXHAUSTED" in enrichment["followup_context"]["flags"]
