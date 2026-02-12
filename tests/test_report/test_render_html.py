"""Tests for the HTML report renderer.

Covers structural sanity checks: the returned string contains expected
markers (Plotly CDN tag, check IDs, TIC ID, section headings, etc.).
Does NOT validate pixel-perfect rendering.
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.types import (
    Candidate,
    Ephemeris,
    LightCurve,
    VettingBundleResult,
    error_result,
    ok_result,
    skipped_result,
)
from bittr_tess_vetter.report import (
    FullLCPlotData,
    LCSummary,
    PhaseFoldedPlotData,
    ReportData,
    build_report,
    render_html,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_box_transit_lc(
    *,
    period_days: float = 3.5,
    t0_btjd: float = 0.5,
    duration_hours: float = 2.5,
    baseline_days: float = 27.0,
    cadence_minutes: float = 10.0,
    depth_frac: float = 0.01,
    noise_ppm: float = 50.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic light curve with box-shaped transits."""
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)

    duration_days = duration_hours / 24.0
    half_phase = (duration_days / period_days) / 2.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    phase_dist = np.minimum(phase, 1.0 - phase)
    in_transit = phase_dist < half_phase

    flux[in_transit] *= 1.0 - depth_frac

    return time, flux, flux_err


def _build_mock_report() -> ReportData:
    """Build a ReportData with all fields populated from mock data."""
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    lc_summary = LCSummary(
        n_points=1000,
        n_valid=990,
        n_transits=7,
        n_in_transit_total=70,
        duration_days=27.0,
        cadence_seconds=120.0,
        flux_std_ppm=100.0,
        flux_mad_ppm=90.0,
        gap_fraction=0.01,
        snr=25.0,
        depth_ppm=10000.0,
        depth_err_ppm=400.0,
    )

    full_lc = FullLCPlotData(
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        flux=[1.0, 0.99, 1.0, 0.99, 1.0],
        transit_mask=[False, True, False, True, False],
    )

    phase_folded = PhaseFoldedPlotData(
        phase=[-0.1, -0.05, 0.0, 0.05, 0.1],
        flux=[1.0, 1.0, 0.99, 1.0, 1.0],
        bin_centers=[-0.05, 0.0, 0.05],
        bin_flux=[1.0, 0.99, 1.0],
        bin_err=[0.001, None, 0.001],
        bin_minutes=30.0,
        transit_duration_phase=0.0298,
        phase_range=(-0.0893, 0.0893),
    )

    check_ok = ok_result(
        id="V01",
        name="odd_even_depth",
        metrics={"rel_diff": 0.05},
        confidence=0.9,
    )
    check_err = error_result(
        id="V02",
        name="transit_shape",
        error="SHAPE_MISMATCH",
        flags=["V_SHAPED"],
    )
    check_skip = skipped_result(
        id="V03",
        name="duration_consistency",
        reason_flag="NO_STELLAR",
        notes=["No stellar parameters provided"],
    )

    all_checks = [check_ok, check_err, check_skip]
    bundle = VettingBundleResult.from_checks(all_checks)

    return ReportData(
        tic_id=12345678,
        toi="TOI-1234.01",
        candidate=candidate,
        lc_summary=lc_summary,
        checks={r.id: r for r in all_checks},
        bundle=bundle,
        full_lc=full_lc,
        phase_folded=phase_folded,
        checks_run=["V01", "V02", "V03"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_html_returns_string() -> None:
    """render_html returns a non-empty string."""
    report = _build_mock_report()
    html = render_html(report)
    assert isinstance(html, str)
    assert len(html) > 0


def test_render_html_contains_plotly_cdn() -> None:
    """Output includes the Plotly.js CDN script tag."""
    report = _build_mock_report()
    html = render_html(report)
    assert "plotly-2.35.2.min.js" in html
    assert "<script src=" in html


def test_render_html_contains_tic_and_toi() -> None:
    """Output includes TIC ID and TOI designation."""
    report = _build_mock_report()
    html = render_html(report)
    assert "12345678" in html
    assert "TOI-1234.01" in html


def test_render_html_contains_ephemeris() -> None:
    """Output includes ephemeris values."""
    report = _build_mock_report()
    html = render_html(report)
    assert "3.500000" in html  # period
    assert "2.500" in html  # duration


def test_render_html_contains_check_ids() -> None:
    """Output includes all check IDs."""
    report = _build_mock_report()
    html = render_html(report)
    assert "V01" in html
    assert "V02" in html
    assert "V03" in html


def test_render_html_contains_check_statuses() -> None:
    """Output shows each check status."""
    report = _build_mock_report()
    html = render_html(report)
    # The ok check
    assert "odd_even_depth" in html
    # The error check
    assert "transit_shape" in html
    assert "SHAPE_MISMATCH" in html
    # The skipped check
    assert "duration_consistency" in html


def test_render_html_contains_lc_summary_values() -> None:
    """Output includes LC summary vital signs."""
    report = _build_mock_report()
    html = render_html(report)
    assert "25.0" in html  # SNR
    assert "10000" in html  # depth
    assert "7" in html  # n_transits


def test_render_html_contains_plot_divs() -> None:
    """Output includes the expected plot container div IDs."""
    report = _build_mock_report()
    html = render_html(report)
    assert 'id="full-lc-plot"' in html
    assert 'id="phase-plot"' in html
    assert 'id="per-transit-plot"' in html
    assert 'id="odd-even-plot"' in html
    assert 'id="secondary-scan-plot"' in html


def test_render_html_embeds_json_data() -> None:
    """Output embeds the report data as JSON in a script block."""
    report = _build_mock_report()
    html = render_html(report)
    assert "var REPORT =" in html
    # The JSON should contain our flux data
    assert '"flux"' in html


def test_render_html_contains_plotly_newplot_calls() -> None:
    """Output includes Plotly.newPlot calls for both charts."""
    report = _build_mock_report()
    html = render_html(report)
    assert "Plotly.newPlot('full-lc-plot'" in html
    assert "Plotly.newPlot('phase-plot'" in html
    assert "Plotly.newPlot('per-transit-plot'" in html
    assert "Plotly.newPlot('odd-even-plot'" in html
    assert "Plotly.newPlot('secondary-scan-plot'" in html


def test_render_html_contains_bundle_summary() -> None:
    """Output shows the bundle summary bar."""
    report = _build_mock_report()
    html = render_html(report)
    assert "1 passed" in html
    assert "1 failed" in html
    assert "1 skipped" in html


def test_render_html_contains_transit_duration_shape() -> None:
    """Output includes transit duration shading in the phase plot JS."""
    report = _build_mock_report()
    html = render_html(report)
    assert "transit_duration_phase" in html


def test_render_html_custom_title() -> None:
    """Custom title is used when provided."""
    report = _build_mock_report()
    html = render_html(report, title="My Custom Report")
    assert "<title>My Custom Report</title>" in html


def test_render_html_default_title_uses_toi() -> None:
    """Default title includes TOI when available."""
    report = _build_mock_report()
    html = render_html(report)
    assert "TOI-1234.01" in html


def test_render_html_is_valid_html_structure() -> None:
    """Output has basic HTML document structure."""
    report = _build_mock_report()
    html = render_html(report)
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html
    assert "<head>" in html
    assert "</head>" in html
    assert "<body>" in html
    assert "</body>" in html


def test_render_html_dark_theme_colors() -> None:
    """Output includes the expected dark theme color values."""
    report = _build_mock_report()
    html = render_html(report)
    assert "#1a1a2e" in html  # background
    assert "#f0b429" in html  # accent


def test_render_html_with_integration_report() -> None:
    """render_html works with a full build_report() output (integration)."""
    time, flux, flux_err = _make_box_transit_lc(
        depth_frac=0.01, noise_ppm=50.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate, tic_id=99999, toi="TOI-9999.01")
    html = render_html(report)

    assert isinstance(html, str)
    assert len(html) > 1000  # non-trivial output
    assert "TOI-9999.01" in html
    assert "99999" in html
    assert "plotly-2.35.2.min.js" in html
    assert "Plotly.newPlot" in html
    # All default checks should appear
    for cid in ["V01", "V02", "V04", "V05", "V13", "V15"]:
        assert cid in html


def test_render_html_with_minimal_report() -> None:
    """render_html handles a minimal ReportData with mostly None fields."""
    report = ReportData()
    html = render_html(report)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    # Should not crash; plot divs are present but JS will skip rendering
    assert 'id="full-lc-plot"' in html
    assert 'id="per-transit-plot"' in html


def test_render_html_check_data_attributes() -> None:
    """Check cards have data-check-id attributes for programmatic access."""
    report = _build_mock_report()
    html = render_html(report)
    assert 'data-check-id="V01"' in html
    assert 'data-check-id="V02"' in html
    assert 'data-check-id="V03"' in html


def test_render_html_version_in_footer() -> None:
    """Output includes version info in footer."""
    report = _build_mock_report()
    html = render_html(report)
    assert "1.0.0" in html
