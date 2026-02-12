# Report Generation

This module builds and renders vetting reports from light-curve inputs.

It has two responsibilities:
- Assemble structured report data (`ReportData`) from LC + candidate inputs.
- Render `ReportData` as self-contained HTML (Plotly-based).

## Public Entry Points

- `build_report(...)` in `bittr_tess_vetter.report`
- `render_html(...)` in `bittr_tess_vetter.report`

Most callers should import from `bittr_tess_vetter.report` (not private files).

## Typical Usage

### LC-only report data + HTML

```python
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.report import build_report, render_html

lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
candidate = Candidate(
    ephemeris=Ephemeris(
        period_days=3.5,
        t0_btjd=1850.0,
        duration_hours=2.5,
    ),
    depth_ppm=500.0,
)

report = build_report(
    lc,
    candidate,
    tic_id=123456789,
    toi="TOI-1234.01",
)

html = render_html(report)
```

### End-to-end API (download + stitch + report + optional enrichment)

```python
from bittr_tess_vetter.api.generate_report import generate_report, EnrichmentConfig

result = generate_report(
    tic_id=340458804,
    period_days=194.243,
    t0_btjd=2039.7625,
    duration_hours=9.256,
    depth_ppm=811.0,
    toi="4510.01",
    include_html=True,
    include_enrichment=True,
    enrichment_config=EnrichmentConfig(network=False),
)

report = result.report
report_json = result.report_json
html = result.html
```

## Data Contract

`ReportData` is the canonical report packet. `to_json()` is JSON-safe and intended for:
- frontend rendering
- artifact persistence
- API transport

Key top-level blocks:
- Identity/inputs: `tic_id`, `toi`, `ephemeris`, `stellar`, `input_depth_ppm`
- Summary: `lc_summary`
- Checks: `checks`, `bundle_summary`, `checks_run`
- Plot payloads: `full_lc`, `phase_folded`, `per_transit_stack`, `local_detrend`, `odd_even_phase`, `secondary_scan`, `oot_context`, `timing_series`, `alias_summary`
- Robustness: `lc_robustness`
- Optional enrichment: `enrichment` with `pixel_diagnostics`, `catalog_context`, `followup_context`

## Current HTML Sections

- Light Curve Summary
- Full Light Curve
- Phase-Folded Transit
- Per-Transit Stack
- Local Detrend Baseline Diagnostic
- Odd vs Even Transits
- Secondary Eclipse / Phase Scan
- Out-of-Transit Noise Context
- LC Robustness Summary
- Enrichment Summary (lightweight status/provenance panel)
- Check Results

## Performance and Payload Controls

Use `build_report(...)` budgets to control payload size:
- `max_lc_points`
- `max_phase_points`
- `max_transit_windows`
- `max_points_per_window`
- `max_timing_points`
- `max_lc_robustness_epochs`

Use `generate_report(..., enrichment_config=...)` for non-LC controls:
- `max_network_seconds`
- `per_request_timeout_seconds`
- `max_concurrent_requests`
- `max_pixel_points`
- `max_catalog_rows`

## Extension Guidelines

When adding new report capabilities:
- Add structured fields in `_data.py` first.
- Compute/assemble in `_build.py` (or API layer for non-LC enrichment).
- Keep renderer (`_render_html.py`) presentation-only.
- Prefer additive fields and deterministic defaults.
- Keep JSON primitive-safe (`dict/list/str/int/float/bool/null`).

## File Map

- `__init__.py`: public exports
- `_data.py`: report dataclasses + serialization
- `_build.py`: report assembly logic
- `_render_html.py`: HTML renderer
