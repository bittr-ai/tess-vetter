"""Compatibility shim for report build internals.

This module keeps the historical import surface stable while implementation
has been split across `_build_core`, `_build_panels`, and `_build_utils`.
"""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.validation.report_bridge import (
    compute_alias_diagnostics,
    compute_timing_series,
    run_lc_checks,
)

from . import _build_core as _core
from . import _build_panels as _panels
from ._build_core import _DEFAULT_ENABLED, _compute_lc_summary, _validate_build_inputs
from ._build_panels import (
    _build_alias_harmonic_summary_data,
    _build_full_lc_plot_data,
    _build_lc_robustness_data,
    _build_lc_robustness_fp_signals,
    _build_lc_robustness_metrics,
    _build_lc_robustness_red_noise,
    _build_local_detrend_diagnostic_plot_data,
    _build_odd_even_phase_plot_data,
    _build_oot_context_plot_data,
    _build_per_transit_stack_plot_data,
    _build_phase_folded_plot_data,
    _build_secondary_scan_plot_data,
    _build_timing_series_plot_data,
    _compute_secondary_scan_quality,
    _secondary_scan_render_hints,
    _to_timing_plot_data,
)
from ._build_utils import (
    _bin_phase_data,
    _depth_ppm_to_flux,
    _downsample_phase_preserving_transit,
    _downsample_preserving_transits,
    _get_valid_time_flux,
    _get_valid_time_flux_quality,
    _red_noise_beta,
    _suggest_flux_y_range,
    _thin_evenly,
    _to_internal_lightcurve,
)


def build_report(lc: Any, candidate: Any, **kwargs: Any):
    """Compatibility wrapper delegating to refactored core implementation."""
    # Preserve historical monkeypatch points from report._build by syncing
    # patched module-level call targets into the new implementation modules.
    _core.run_lc_checks = run_lc_checks
    _core._DEFAULT_ENABLED = _DEFAULT_ENABLED
    _core._validate_build_inputs = _validate_build_inputs
    _core._compute_lc_summary = _compute_lc_summary
    _core._to_internal_lightcurve = _to_internal_lightcurve
    _core._build_full_lc_plot_data = _build_full_lc_plot_data
    _core._build_phase_folded_plot_data = _build_phase_folded_plot_data
    _core._build_per_transit_stack_plot_data = _build_per_transit_stack_plot_data
    _core._build_odd_even_phase_plot_data = _build_odd_even_phase_plot_data
    _core._build_local_detrend_diagnostic_plot_data = _build_local_detrend_diagnostic_plot_data
    _core._build_oot_context_plot_data = _build_oot_context_plot_data
    _core._build_timing_series_plot_data = _build_timing_series_plot_data
    _core._build_alias_harmonic_summary_data = _build_alias_harmonic_summary_data
    _core._build_secondary_scan_plot_data = _build_secondary_scan_plot_data
    _core._build_lc_robustness_data = _build_lc_robustness_data

    _panels.compute_timing_series = compute_timing_series
    _panels.compute_alias_diagnostics = compute_alias_diagnostics
    _panels._to_internal_lightcurve = _to_internal_lightcurve
    _panels._downsample_preserving_transits = _downsample_preserving_transits
    _panels._downsample_phase_preserving_transit = _downsample_phase_preserving_transit
    _panels._suggest_flux_y_range = _suggest_flux_y_range
    _panels._depth_ppm_to_flux = _depth_ppm_to_flux
    _panels._get_valid_time_flux = _get_valid_time_flux
    _panels._thin_evenly = _thin_evenly
    _panels._red_noise_beta = _red_noise_beta
    _panels._get_valid_time_flux_quality = _get_valid_time_flux_quality
    _panels._bin_phase_data = _bin_phase_data
    _panels._to_timing_plot_data = _to_timing_plot_data
    _panels._compute_secondary_scan_quality = _compute_secondary_scan_quality
    _panels._secondary_scan_render_hints = _secondary_scan_render_hints
    _panels._build_lc_robustness_metrics = _build_lc_robustness_metrics
    _panels._build_lc_robustness_red_noise = _build_lc_robustness_red_noise
    _panels._build_lc_robustness_fp_signals = _build_lc_robustness_fp_signals
    return _core.build_report(lc, candidate, **kwargs)


__all__ = [
    "build_report",
    "run_lc_checks",
    "compute_timing_series",
    "compute_alias_diagnostics",
    "_DEFAULT_ENABLED",
    "_validate_build_inputs",
    "_compute_lc_summary",
    "_downsample_preserving_transits",
    "_build_full_lc_plot_data",
    "_downsample_phase_preserving_transit",
    "_build_phase_folded_plot_data",
    "_suggest_flux_y_range",
    "_depth_ppm_to_flux",
    "_to_internal_lightcurve",
    "_get_valid_time_flux",
    "_thin_evenly",
    "_build_per_transit_stack_plot_data",
    "_build_odd_even_phase_plot_data",
    "_build_local_detrend_diagnostic_plot_data",
    "_build_oot_context_plot_data",
    "_build_timing_series_plot_data",
    "_to_timing_plot_data",
    "_build_alias_harmonic_summary_data",
    "_build_lc_robustness_data",
    "_build_lc_robustness_metrics",
    "_red_noise_beta",
    "_build_lc_robustness_red_noise",
    "_build_lc_robustness_fp_signals",
    "_get_valid_time_flux_quality",
    "_build_secondary_scan_plot_data",
    "_compute_secondary_scan_quality",
    "_secondary_scan_render_hints",
    "_bin_phase_data",
]
