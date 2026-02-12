"""Lightweight HTML renderer for LC-only vetting reports.

Consumes a ReportData object and produces a self-contained HTML string
with interactive Plotly charts.  No external dependencies beyond the
Plotly.js CDN.
"""

from __future__ import annotations

import json
from typing import Any

from bittr_tess_vetter.report._data import ReportData

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# -- Color palette (dark terminal aesthetic) ----------------------------------
_BG = "#1a1a2e"
_BG_CARD = "#16213e"
_BG_SURFACE = "#0f3460"
_ACCENT = "#f0b429"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#8a8a9a"
_OK = "#2ecc71"
_FAIL = "#e74c3c"
_SKIP = "#7f8c8d"
_TRANSIT_COLOR = "#e74c3c"
_OOT_COLOR = "#5dade2"
_BIN_COLOR = "#f0b429"


def render_html(report: ReportData, *, title: str | None = None) -> str:
    """Render a ReportData object to a self-contained HTML string.

    Args:
        report: The assembled report data packet.
        title: Optional page title. Defaults to "TESS Vetting Report".

    Returns:
        A complete HTML document as a string with embedded Plotly charts.
    """
    data = report.to_json()
    page_title = title or _build_default_title(data)
    data_json = json.dumps(data, indent=None, allow_nan=False)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(page_title)}</title>
<script src="{_PLOTLY_CDN}"></script>
{_css_block()}
</head>
<body>

{_header_section(data)}
{_lc_summary_section(data)}

<div class="plot-panel">
  <h2>Full Light Curve</h2>
  <div id="full-lc-plot" class="plot-container"></div>
</div>

<div class="plot-panel">
  <h2>Phase-Folded Transit</h2>
  <div id="phase-plot" class="plot-container"></div>
</div>

<div class="plot-panel">
  <h2>Per-Transit Stack</h2>
  <div id="per-transit-plot" class="plot-container"></div>
</div>

<div class="plot-panel">
  <h2>Local Detrend Baseline Diagnostic</h2>
  <div id="local-detrend-plot" class="plot-container"></div>
</div>

<div class="plot-panel">
  <h2>Odd vs Even Transits</h2>
  <div id="odd-even-plot" class="plot-container"></div>
</div>

<div class="plot-panel">
  <h2>Secondary Eclipse / Phase Scan</h2>
  <div id="secondary-scan-plot" class="plot-container"></div>
</div>

<div class="plot-panel">
  <h2>Out-of-Transit Noise Context</h2>
  <div id="oot-context-plot" class="plot-container"></div>
</div>

{_lc_robustness_section(data)}

{_enrichment_section(data)}

{_checks_section(data)}

<script>
// Embed report data
var REPORT = {data_json};

{_percentile_helper_js()}

{_full_lc_js()}

{_phase_folded_js()}

{_per_transit_stack_js()}

{_local_detrend_js()}

{_odd_even_js()}

{_secondary_scan_js()}

{_oot_context_js()}
</script>

<footer class="footer">
  <span>bittr-tess-vetter &middot; report v{_esc(str(data.get("version", "?")))}</span>
</footer>
</body>
</html>"""


# -- Private helpers ----------------------------------------------------------


def _build_default_title(data: dict[str, Any]) -> str:
    parts = ["TESS Vetting Report"]
    if data.get("toi"):
        parts.append(str(data["toi"]))
    elif data.get("tic_id") is not None:
        parts.append(f"TIC {data['tic_id']}")
    return " - ".join(parts)


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(value: Any, precision: int = 2, suffix: str = "") -> str:
    """Format a numeric value for display, handling None gracefully."""
    if value is None:
        return "&mdash;"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


# -- CSS ----------------------------------------------------------------------


def _css_block() -> str:
    return f"""\
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 20px;
    font-family: 'Menlo', 'Consolas', 'Monaco', monospace;
    background: {_BG}; color: {_TEXT};
    line-height: 1.5;
  }}
  h1, h2, h3 {{ color: {_ACCENT}; margin-top: 0; font-weight: 600; }}
  h1 {{ font-size: 1.4em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1em; margin-bottom: 12px; }}

  .header {{
    background: {_BG_CARD};
    border: 1px solid {_BG_SURFACE};
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .header .subtitle {{
    color: {_TEXT_DIM}; font-size: 0.85em; margin: 0;
  }}
  .ephemeris-grid {{
    display: flex; flex-wrap: wrap; gap: 16px 32px;
    margin-top: 12px;
  }}
  .eph-item {{
    display: flex; flex-direction: column;
  }}
  .eph-label {{ color: {_TEXT_DIM}; font-size: 0.75em; text-transform: uppercase; }}
  .eph-value {{ color: {_TEXT}; font-size: 1.0em; font-weight: 600; }}

  .summary-panel {{
    background: {_BG_CARD};
    border: 1px solid {_BG_SURFACE};
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .vitals-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
  }}
  .vital {{
    background: {_BG_SURFACE};
    border-radius: 6px;
    padding: 10px 14px;
  }}
  .vital-label {{ color: {_TEXT_DIM}; font-size: 0.7em; text-transform: uppercase; }}
  .vital-value {{ color: {_TEXT}; font-size: 1.05em; font-weight: 600; }}
  .section-note {{ color: {_TEXT_DIM}; font-size: 0.8em; margin: 0 0 10px 0; }}

  .plot-panel {{
    background: {_BG_CARD};
    border: 1px solid {_BG_SURFACE};
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .plot-container {{
    width: 100%; min-height: 340px;
  }}

  .checks-panel {{
    background: {_BG_CARD};
    border: 1px solid {_BG_SURFACE};
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .checks-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 12px;
  }}
  .check-card {{
    background: {_BG_SURFACE};
    border-radius: 6px;
    padding: 14px 16px;
    border-left: 4px solid {_SKIP};
  }}
  .check-card.ok {{ border-left-color: {_OK}; }}
  .check-card.error {{ border-left-color: {_FAIL}; }}
  .check-card.skipped {{ border-left-color: {_SKIP}; }}
  .check-header {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px;
  }}
  .check-id {{
    font-weight: 700; font-size: 0.9em;
  }}
  .check-name {{
    color: {_TEXT_DIM}; font-size: 0.8em;
  }}
  .check-status {{
    margin-left: auto;
    font-size: 0.75em; font-weight: 700;
    text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
  }}
  .check-status.ok {{ background: {_OK}22; color: {_OK}; }}
  .check-status.error {{ background: {_FAIL}22; color: {_FAIL}; }}
  .check-status.skipped {{ background: {_SKIP}22; color: {_SKIP}; }}
  .check-metrics {{
    font-size: 0.78em; color: {_TEXT_DIM};
    margin-top: 4px;
  }}
  .check-metrics span {{
    display: inline-block; margin-right: 12px;
  }}
  .check-flags {{
    font-size: 0.73em; color: {_FAIL}; margin-top: 4px;
  }}
  .check-notes {{
    font-size: 0.73em; color: {_TEXT_DIM}; margin-top: 2px; font-style: italic;
  }}

  .bundle-bar {{
    display: flex; gap: 16px; margin-bottom: 14px;
    font-size: 0.85em;
  }}
  .bundle-stat {{
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 600;
  }}
  .bundle-ok {{ background: {_OK}22; color: {_OK}; }}
  .bundle-fail {{ background: {_FAIL}22; color: {_FAIL}; }}
  .bundle-skip {{ background: {_SKIP}22; color: {_SKIP}; }}

  .footer {{
    text-align: center;
    color: {_TEXT_DIM}; font-size: 0.7em;
    padding: 16px 0 8px;
  }}

  @media (max-width: 640px) {{
    body {{ padding: 10px; }}
    .vitals-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .checks-grid {{ grid-template-columns: 1fr; }}
  }}
</style>"""


# -- Header section -----------------------------------------------------------


def _header_section(data: dict[str, Any]) -> str:
    tic_id = data.get("tic_id")
    toi = data.get("toi")
    eph = data.get("ephemeris", {})
    input_depth = data.get("input_depth_ppm")

    # Title line
    if toi:
        title_text = _esc(str(toi))
        subtitle = f"TIC {tic_id}" if tic_id is not None else ""
    elif tic_id is not None:
        title_text = f"TIC {tic_id}"
        subtitle = ""
    else:
        title_text = "TESS Candidate"
        subtitle = ""

    subtitle_html = f'<p class="subtitle">{_esc(subtitle)}</p>' if subtitle else ""

    # Ephemeris items
    eph_items = ""
    if eph:
        eph_items = f"""\
    <div class="ephemeris-grid">
      <div class="eph-item">
        <span class="eph-label">Period</span>
        <span class="eph-value">{_fmt(eph.get("period_days"), 6, " d")}</span>
      </div>
      <div class="eph-item">
        <span class="eph-label">T0 (BTJD)</span>
        <span class="eph-value">{_fmt(eph.get("t0_btjd"), 6)}</span>
      </div>
      <div class="eph-item">
        <span class="eph-label">Duration</span>
        <span class="eph-value">{_fmt(eph.get("duration_hours"), 3, " hr")}</span>
      </div>
      <div class="eph-item">
        <span class="eph-label">Input Depth</span>
        <span class="eph-value">{_fmt(input_depth, 0, " ppm")}</span>
      </div>
    </div>"""

    return f"""\
<div class="header">
  <h1>{title_text}</h1>
  {subtitle_html}
  {eph_items}
</div>"""


# -- LC Summary section -------------------------------------------------------


def _lc_summary_section(data: dict[str, Any]) -> str:
    lc = data.get("lc_summary")
    if not lc:
        return ""

    def _vital(label: str, value: str) -> str:
        return f"""\
      <div class="vital">
        <div class="vital-label">{label}</div>
        <div class="vital-value">{value}</div>
      </div>"""

    depth_str = _fmt(lc.get("depth_ppm"), 0, " ppm")
    err = lc.get("depth_err_ppm")
    if err is not None:
        depth_str += f" &plusmn; {_fmt(err, 0)}"

    vitals = [
        _vital("SNR", _fmt(lc.get("snr"), 1)),
        _vital("Depth", depth_str),
        _vital("Transits", _fmt(lc.get("n_transits"))),
        _vital("In-transit pts", _fmt(lc.get("n_in_transit_total"))),
        _vital("Scatter (std)", _fmt(lc.get("flux_std_ppm"), 0, " ppm")),
        _vital("Scatter (MAD)", _fmt(lc.get("flux_mad_ppm"), 0, " ppm")),
        _vital("Cadence", _fmt(lc.get("cadence_seconds"), 0, " s")),
        _vital("Baseline", _fmt(lc.get("duration_days"), 1, " d")),
        _vital("Gap fraction", _fmt(lc.get("gap_fraction"), 3)),
        _vital("Points", f"{_fmt(lc.get('n_valid'))} / {_fmt(lc.get('n_points'))}"),
    ]

    return f"""\
<div class="summary-panel">
  <h2>Light Curve Summary</h2>
  <div class="vitals-grid">
{"".join(vitals)}
  </div>
</div>"""


def _lc_robustness_section(data: dict[str, Any]) -> str:
    lc_robustness = data.get("lc_robustness")
    if not lc_robustness:
        return ""

    rb = lc_robustness.get("robustness", {})
    rn = lc_robustness.get("red_noise", {})
    fp = lc_robustness.get("fp_signals", {})
    per_epoch = lc_robustness.get("per_epoch", [])

    def _vital(label: str, value: str) -> str:
        return f"""\
      <div class="vital">
        <div class="vital-label">{label}</div>
        <div class="vital-value">{value}</div>
      </div>"""

    loto_triplet = (
        f"{_fmt(rb.get('loto_snr_min'), 2)} / "
        f"{_fmt(rb.get('loto_snr_mean'), 2)} / "
        f"{_fmt(rb.get('loto_snr_max'), 2)}"
    )
    beta_triplet = (
        f"{_fmt(rn.get('beta_30m'), 2)} / "
        f"{_fmt(rn.get('beta_60m'), 2)} / "
        f"{_fmt(rn.get('beta_duration'), 2)}"
    )

    vitals = [
        _vital("LC Robustness Version", _fmt(lc_robustness.get("version"))),
        _vital("Epochs (stored/measured)", f"{len(per_epoch)} / {_fmt(rb.get('n_epochs_measured'))}"),
        _vital("Dominance Index", _fmt(rb.get("dominance_index"), 3)),
        _vital("LOTO SNR (min/mean/max)", loto_triplet),
        _vital("LOTO Depth Shift", _fmt(rb.get("loto_depth_shift_ppm_max"), 1, " ppm")),
        _vital("Red Noise β (30m/60m/dur)", beta_triplet),
        _vital("Odd-Even Δ Depth", _fmt(fp.get("odd_even_depth_diff_sigma"), 2, " sigma")),
        _vital("Secondary Depth", _fmt(fp.get("secondary_depth_sigma"), 2, " sigma")),
        _vital("Phase 0.5 Depth", _fmt(fp.get("phase_0p5_bin_depth_ppm"), 1, " ppm")),
    ]

    return f"""\
<div class="summary-panel">
  <h2>LC Robustness Summary</h2>
  <p class="section-note">Computed LC-only fragility and false-positive summary metrics.</p>
  <div class="vitals-grid">
{"".join(vitals)}
  </div>
</div>"""


def _enrichment_section(data: dict[str, Any]) -> str:
    enrichment = data.get("enrichment")
    if not enrichment:
        return ""

    cards = [
        _enrichment_block_card("Pixel Diagnostics", enrichment.get("pixel_diagnostics")),
        _enrichment_block_card("Catalog Context", enrichment.get("catalog_context")),
        _enrichment_block_card("Followup Context", enrichment.get("followup_context")),
    ]

    return f"""\
<div class="checks-panel">
  <h2>Enrichment Summary</h2>
  <p class="section-note">Optional non-LC context blocks with deterministic status and provenance.</p>
  <div class="checks-grid">
{"".join(cards)}
  </div>
</div>"""


def _enrichment_block_card(label: str, block: dict[str, Any] | None) -> str:
    if block is None:
        status = "skipped"
        flags: list[str] = ["BLOCK_DISABLED"]
        quality: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        payload: dict[str, Any] = {}
    else:
        status = str(block.get("status", "skipped"))
        flags = list(block.get("flags", []))
        quality = dict(block.get("quality", {}))
        provenance = dict(block.get("provenance", {}))
        payload = dict(block.get("payload", {}))

    status_cls = status if status in {"ok", "error", "skipped"} else "skipped"
    flags_html = (
        f'<div class="check-flags">{_esc(", ".join(str(f) for f in flags))}</div>' if flags else ""
    )

    quality_items = "".join(
        f"<span><strong>{_esc(str(k))}:</strong> {_esc(str(v))}</span>"
        for k, v in sorted(quality.items())
    )
    quality_html = f'<div class="check-metrics">{quality_items}</div>' if quality_items else ""

    prov_bits: list[str] = []
    block_name = provenance.get("block")
    if block_name is not None:
        prov_bits.append(f"<span><strong>block:</strong> {_esc(str(block_name))}</span>")
    budget = provenance.get("budget")
    if isinstance(budget, dict):
        for k in ("budget_applied", "n_points", "max_pixel_points"):
            if k in budget:
                prov_bits.append(f"<span><strong>{_esc(k)}:</strong> {_esc(str(budget[k]))}</span>")
    if "cache_hit" in provenance:
        prov_bits.append(f"<span><strong>cache_hit:</strong> {_esc(str(provenance['cache_hit']))}</span>")
    if "sector_selected" in provenance:
        prov_bits.append(
            f"<span><strong>sector_selected:</strong> {_esc(str(provenance['sector_selected']))}</span>"
        )
    if "selected_sector" in payload:
        prov_bits.append(
            f"<span><strong>selected_sector:</strong> {_esc(str(payload['selected_sector']))}</span>"
        )
    if "n_checks" in payload:
        prov_bits.append(f"<span><strong>n_checks:</strong> {_esc(str(payload['n_checks']))}</span>")
    if "catalog_rows" in payload:
        prov_bits.append(
            f"<span><strong>catalog_rows:</strong> {_esc(str(payload['catalog_rows']))}</span>"
        )
    prov_html = f'<div class="check-metrics">{"".join(prov_bits)}</div>' if prov_bits else ""

    return f"""\
    <div class="check-card {status_cls}">
      <div class="check-header">
        <span class="check-id">{_esc(label)}</span>
        <span class="check-status {status_cls}">{_esc(status)}</span>
      </div>
      {quality_html}
      {prov_html}
      {flags_html}
    </div>"""


# -- Checks section -----------------------------------------------------------


def _checks_section(data: dict[str, Any]) -> str:
    checks = data.get("checks", {})
    bundle = data.get("bundle_summary")

    if not checks:
        return ""

    # Bundle bar
    bundle_html = ""
    if bundle:
        bundle_html = f"""\
  <div class="bundle-bar">
    <span class="bundle-stat bundle-ok">{bundle.get("n_ok", 0)} passed</span>
    <span class="bundle-stat bundle-fail">{bundle.get("n_failed", 0)} failed</span>
    <span class="bundle-stat bundle-skip">{bundle.get("n_skipped", 0)} skipped</span>
  </div>"""

    # Order by check ID
    ordered = data.get("checks_run", sorted(checks.keys()))
    cards = []
    for cid in ordered:
        cr = checks.get(cid)
        if cr is None:
            continue
        cards.append(_check_card(cid, cr))

    return f"""\
<div class="checks-panel">
  <h2>Check Results</h2>
  {bundle_html}
  <div class="checks-grid">
{"".join(cards)}
  </div>
</div>"""


def _check_card(check_id: str, cr: dict[str, Any]) -> str:
    status = cr.get("status", "skipped")
    name = cr.get("name", "")
    metrics = cr.get("metrics", {})
    flags = cr.get("flags", [])
    notes = cr.get("notes", [])

    # Metrics display
    metrics_html = ""
    if metrics:
        spans = []
        for k, v in metrics.items():
            spans.append(f"<span><strong>{_esc(str(k))}:</strong> {_esc(str(v))}</span>")
        metrics_html = f'<div class="check-metrics">{"".join(spans)}</div>'

    # Flags display
    flags_html = ""
    if flags:
        flags_html = f'<div class="check-flags">{_esc(", ".join(str(f) for f in flags))}</div>'

    # Notes display
    notes_html = ""
    if notes:
        notes_html = f'<div class="check-notes">{_esc("; ".join(str(n) for n in notes))}</div>'

    return f"""\
    <div class="check-card {_esc(status)}" data-check-id="{_esc(check_id)}">
      <div class="check-header">
        <span class="check-id">{_esc(check_id)}</span>
        <span class="check-name">{_esc(name)}</span>
        <span class="check-status {_esc(status)}">{_esc(status)}</span>
      </div>
      {metrics_html}
      {flags_html}
      {notes_html}
    </div>"""


# -- Plotly JS builders -------------------------------------------------------


def _plotly_layout_defaults() -> str:
    """Return a JS object string with shared Plotly layout defaults."""
    return f"""\
{{
    paper_bgcolor: '{_BG_CARD}',
    plot_bgcolor: '{_BG}',
    font: {{ family: 'Menlo, Consolas, Monaco, monospace', color: '{_TEXT}', size: 11 }},
    margin: {{ t: 30, r: 20, b: 50, l: 60 }},
    xaxis: {{
      gridcolor: '{_BG_SURFACE}',
      zerolinecolor: '{_BG_SURFACE}',
    }},
    yaxis: {{
      gridcolor: '{_BG_SURFACE}',
      zerolinecolor: '{_BG_SURFACE}',
    }},
    legend: {{
      bgcolor: 'rgba(0,0,0,0)',
      font: {{ size: 10 }},
    }},
  }}"""


def _percentile_helper_js() -> str:
    """Return a JS function for computing percentile-based y-axis range."""
    return """\
function percentileRange(values, lo, hi, pad) {
  // Compute percentile-based y-axis range with padding.
  // lo/hi are percentiles (0-100), pad is fractional padding (e.g. 0.1).
  var sorted = values.slice().filter(function(v) { return isFinite(v); });
  if (sorted.length === 0) return null;
  sorted.sort(function(a, b) { return a - b; });
  var loIdx = Math.floor(lo / 100.0 * (sorted.length - 1));
  var hiIdx = Math.ceil(hi / 100.0 * (sorted.length - 1));
  var yLo = sorted[loIdx];
  var yHi = sorted[hiIdx];
  var span = yHi - yLo;
  if (span <= 0) span = Math.abs(yHi) * 0.01 || 1e-6;
  return [yLo - pad * span, yHi + pad * span];
}"""


def _full_lc_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.full_lc;
  if (!d) return;

  var ootTime = [], ootFlux = [], itTime = [], itFlux = [];
  for (var i = 0; i < d.time.length; i++) {{
    if (d.transit_mask[i]) {{
      itTime.push(d.time[i]);
      itFlux.push(d.flux[i]);
    }} else {{
      ootTime.push(d.time[i]);
      ootFlux.push(d.flux[i]);
    }}
  }}

  // Percentile-based y-axis range (2nd-98th with 10% padding)
  var yRange = percentileRange(d.flux, 2, 98, 0.1);

  var traceOOT = {{
    x: ootTime, y: ootFlux,
    mode: 'markers',
    type: 'scattergl',
    marker: {{ color: '{_OOT_COLOR}', size: 2, opacity: 0.5 }},
    name: 'Out-of-transit',
  }};

  var traceIT = {{
    x: itTime, y: itFlux,
    mode: 'markers',
    type: 'scattergl',
    marker: {{ color: '{_TRANSIT_COLOR}', size: 3, opacity: 0.8 }},
    name: 'In-transit',
  }};

  var yaxisCfg = Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
    title: {{ text: 'Normalized Flux', standoff: 8 }},
  }});
  if (yRange) yaxisCfg.range = yRange;

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Time (BTJD)', standoff: 8 }},
    }}),
    yaxis: yaxisCfg,
  }});

  Plotly.newPlot('full-lc-plot', [traceOOT, traceIT], layout, {{responsive: true}});
}})();"""


def _phase_folded_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.phase_folded;
  if (!d) return;

  var phaseRange = d.phase_range;
  var halfDur = d.transit_duration_phase / 2.0;

  // Raw scatter (within view range for performance)
  var rawPhase = [], rawFlux = [];
  for (var i = 0; i < d.phase.length; i++) {{
    rawPhase.push(d.phase[i]);
    rawFlux.push(d.flux[i]);
  }}

  // Percentile-based y-axis range (2nd-98th with 10% padding)
  var yRange = percentileRange(d.flux, 2, 98, 0.1);

  var traceRaw = {{
    x: rawPhase, y: rawFlux,
    mode: 'markers',
    type: 'scattergl',
    marker: {{ color: '{_OOT_COLOR}', size: 2, opacity: 0.3 }},
    name: 'Raw',
  }};

  // Binned data with error bars
  var binErr = [];
  var binErrVisible = [];
  for (var i = 0; i < d.bin_err.length; i++) {{
    if (d.bin_err[i] !== null) {{
      binErr.push(d.bin_err[i]);
      binErrVisible.push(true);
    }} else {{
      binErr.push(0);
      binErrVisible.push(false);
    }}
  }}

  // Build asymmetric error bars that respect per-point visibility
  // (Plotly doesn't support per-point visible, so set to 0 for hidden)
  var traceBin = {{
    x: d.bin_centers, y: d.bin_flux,
    mode: 'markers',
    type: 'scatter',
    marker: {{ color: '{_BIN_COLOR}', size: 8, symbol: 'circle' }},
    error_y: {{
      type: 'data',
      array: binErr,
      visible: true,
      color: '{_BIN_COLOR}',
      thickness: 2,
      width: 4,
    }},
    name: 'Binned (' + d.bin_minutes + ' min)',
  }};

  // Transit duration shaded region
  var shapes = [
    {{
      type: 'rect',
      x0: -halfDur, x1: halfDur,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      fillcolor: '{_TRANSIT_COLOR}',
      opacity: 0.08,
      line: {{ width: 0 }},
    }},
    {{
      type: 'line',
      x0: -halfDur, x1: -halfDur,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: {{ color: '{_TRANSIT_COLOR}', width: 1, dash: 'dot' }},
    }},
    {{
      type: 'line',
      x0: halfDur, x1: halfDur,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: {{ color: '{_TRANSIT_COLOR}', width: 1, dash: 'dot' }},
    }},
  ];

  // Transit depth reference line (if input_depth_ppm is available)
  var depthPpm = (REPORT.input_depth_ppm != null) ? REPORT.input_depth_ppm : null;
  if (depthPpm !== null) {{
    var depthFlux = 1.0 - depthPpm / 1e6;
    shapes.push({{
      type: 'line',
      x0: phaseRange[0], x1: phaseRange[1],
      y0: depthFlux, y1: depthFlux,
      xref: 'x', yref: 'y',
      line: {{ color: '{_ACCENT}', width: 1.5, dash: 'dash' }},
    }});
  }}

  var yaxisCfg = Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
    title: {{ text: 'Normalized Flux', standoff: 8 }},
  }});
  if (yRange) yaxisCfg.range = yRange;

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Orbital Phase', standoff: 8 }},
      range: [phaseRange[0], phaseRange[1]],
    }}),
    yaxis: yaxisCfg,
    shapes: shapes,
  }});

  Plotly.newPlot('phase-plot', [traceRaw, traceBin], layout, {{responsive: true}});
}})();"""


def _per_transit_stack_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.per_transit_stack;
  if (!d || !d.windows || d.windows.length === 0) return;

  var allFlux = [];
  for (var i = 0; i < d.windows.length; i++) {{
    var w = d.windows[i];
    for (var j = 0; j < w.flux.length; j++) allFlux.push(w.flux[j]);
  }}
  var yBase = percentileRange(allFlux, 2, 98, 0.0);
  var span = (yBase && yBase[1] > yBase[0]) ? (yBase[1] - yBase[0]) : 0.002;
  var offset = span * 1.4;

  var traces = [];
  for (var i = 0; i < d.windows.length; i++) {{
    var w = d.windows[i];
    var y = [];
    for (var j = 0; j < w.flux.length; j++) y.push(w.flux[j] + i * offset);
    traces.push({{
      x: w.dt_hours,
      y: y,
      mode: 'markers',
      type: 'scattergl',
      marker: {{ color: '{_OOT_COLOR}', size: 3, opacity: 0.55 }},
      name: 'E' + w.epoch,
      showlegend: false,
      hovertemplate: 'Epoch ' + w.epoch + '<br>dt=%{{x:.2f}} h<br>flux=%{{customdata:.6f}}<extra></extra>',
      customdata: w.flux,
    }});

    var xIT = [], yIT = [];
    for (var j = 0; j < w.dt_hours.length; j++) {{
      if (w.in_transit_mask[j]) {{
        xIT.push(w.dt_hours[j]);
        yIT.push(w.flux[j] + i * offset);
      }}
    }}
    if (xIT.length > 0) {{
      traces.push({{
        x: xIT,
        y: yIT,
        mode: 'markers',
        type: 'scattergl',
        marker: {{ color: '{_TRANSIT_COLOR}', size: 4, opacity: 0.85 }},
        showlegend: false,
        hoverinfo: 'skip',
      }});
    }}
  }}

  var shapes = [
    {{
      type: 'rect',
      x0: -(d.window_half_hours / 3.0) / 2.0,
      x1: (d.window_half_hours / 3.0) / 2.0,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      fillcolor: '{_TRANSIT_COLOR}',
      opacity: 0.06,
      line: {{ width: 0 }},
    }}
  ];

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Hours From Transit Midpoint', standoff: 8 }},
      range: [-d.window_half_hours, d.window_half_hours],
    }}),
    yaxis: Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
      title: {{ text: 'Flux (Stacked Offsets)', standoff: 8 }},
      showticklabels: false,
    }}),
    shapes: shapes,
  }});

  Plotly.newPlot('per-transit-plot', traces, layout, {{responsive: true}});
}})();"""


def _odd_even_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.odd_even_phase;
  if (!d) return;

  var traces = [];
  if (d.odd_phase && d.odd_phase.length > 0) {{
    traces.push({{
      x: d.odd_phase, y: d.odd_flux,
      mode: 'markers',
      type: 'scattergl',
      marker: {{ color: '{_OOT_COLOR}', size: 2, opacity: 0.28 }},
      name: 'Odd (raw)',
    }});
  }}
  if (d.even_phase && d.even_phase.length > 0) {{
    traces.push({{
      x: d.even_phase, y: d.even_flux,
      mode: 'markers',
      type: 'scattergl',
      marker: {{ color: '{_ACCENT}', size: 2, opacity: 0.28 }},
      name: 'Even (raw)',
    }});
  }}
  if (d.odd_bin_centers && d.odd_bin_centers.length > 0) {{
    traces.push({{
      x: d.odd_bin_centers, y: d.odd_bin_flux,
      mode: 'lines+markers',
      type: 'scatter',
      line: {{ color: '{_OOT_COLOR}', width: 2 }},
      marker: {{ size: 5 }},
      name: 'Odd (binned)',
    }});
  }}
  if (d.even_bin_centers && d.even_bin_centers.length > 0) {{
    traces.push({{
      x: d.even_bin_centers, y: d.even_bin_flux,
      mode: 'lines+markers',
      type: 'scatter',
      line: {{ color: '{_ACCENT}', width: 2 }},
      marker: {{ size: 5 }},
      name: 'Even (binned)',
    }});
  }}
  if (traces.length === 0) return;

  var allFlux = [];
  if (d.odd_flux) allFlux = allFlux.concat(d.odd_flux);
  if (d.even_flux) allFlux = allFlux.concat(d.even_flux);
  var yRange = percentileRange(allFlux, 2, 98, 0.12);

  var yaxisCfg = Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
    title: {{ text: 'Normalized Flux', standoff: 8 }},
  }});
  if (yRange) yaxisCfg.range = yRange;

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Orbital Phase', standoff: 8 }},
      range: d.phase_range,
    }}),
    yaxis: yaxisCfg,
  }});

  Plotly.newPlot('odd-even-plot', traces, layout, {{responsive: true}});
}})();"""


def _local_detrend_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.local_detrend;
  if (!d || !d.windows || d.windows.length === 0) return;

  var allResidual = [];
  for (var i = 0; i < d.windows.length; i++) {{
    var w = d.windows[i];
    for (var j = 0; j < w.flux.length; j++) {{
      var f = w.flux[j];
      var b = w.baseline_flux[j];
      if (isFinite(f) && isFinite(b)) allResidual.push(f - b);
    }}
  }}
  var yBase = percentileRange(allResidual, 2, 98, 0.0);
  var span = (yBase && yBase[1] > yBase[0]) ? (yBase[1] - yBase[0]) : 0.0008;
  var offset = span * 1.8;

  var traces = [];
  for (var i = 0; i < d.windows.length; i++) {{
    var w = d.windows[i];
    var yResidual = [];
    var yZeroLine = [];
    var residualPpm = [];
    for (var j = 0; j < w.flux.length; j++) {{
      var f = w.flux[j];
      var b = w.baseline_flux[j];
      if (isFinite(f) && isFinite(b)) {{
        var r = f - b;
        yResidual.push(r + i * offset);
        residualPpm.push(r * 1e6);
      }} else {{
        yResidual.push(null);
        residualPpm.push(null);
      }}
      yZeroLine.push(i * offset);
    }}
    traces.push({{
      x: w.dt_hours,
      y: yResidual,
      mode: 'markers',
      type: 'scattergl',
      marker: {{ color: '{_OOT_COLOR}', size: 3, opacity: 0.45 }},
      name: 'E' + w.epoch + ' residual',
      showlegend: false,
      hovertemplate: 'Epoch ' + w.epoch + '<br>dt=%{{x:.2f}} h<br>resid=%{{customdata:.0f}} ppm<extra></extra>',
      customdata: residualPpm,
    }});
    traces.push({{
      x: w.dt_hours,
      y: yZeroLine,
      mode: 'lines',
      type: 'scatter',
      line: {{ color: '{_ACCENT}', width: 1.3 }},
      name: 'E' + w.epoch + ' baseline zero',
      showlegend: false,
      hoverinfo: 'skip',
    }});

    var xIT = [], yIT = [];
    for (var j = 0; j < w.dt_hours.length; j++) {{
      if (w.in_transit_mask[j]) {{
        var f = w.flux[j];
        var b = w.baseline_flux[j];
        if (!isFinite(f) || !isFinite(b)) continue;
        var r = f - b;
        xIT.push(w.dt_hours[j]);
        yIT.push(r + i * offset);
      }}
    }}
    if (xIT.length > 0) {{
      traces.push({{
        x: xIT,
        y: yIT,
        mode: 'markers',
        type: 'scattergl',
        marker: {{ color: '{_TRANSIT_COLOR}', size: 4, opacity: 0.85 }},
        showlegend: false,
        hoverinfo: 'skip',
      }});
    }}
  }}

  var halfTransitHours = (d.window_half_hours / 3.0) / 2.0;
  var shapes = [
    {{
      type: 'rect',
      x0: -halfTransitHours,
      x1: halfTransitHours,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      fillcolor: '{_TRANSIT_COLOR}',
      opacity: 0.05,
      line: {{ width: 0 }},
    }}
  ];

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Hours From Transit Midpoint', standoff: 8 }},
      range: [-d.window_half_hours, d.window_half_hours],
    }}),
    yaxis: Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
      title: {{ text: 'Flux - Local Baseline (Stacked)', standoff: 8 }},
      showticklabels: false,
    }}),
    shapes: shapes,
  }});

  Plotly.newPlot('local-detrend-plot', traces, layout, {{responsive: true}});
}})();"""


def _secondary_scan_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.secondary_scan;
  if (!d) return;
  var hints = d.render_hints || {{}};

  var traceRaw = {{
    x: d.phase, y: d.flux,
    mode: 'markers',
    type: 'scattergl',
    marker: {{
      color: '{_OOT_COLOR}',
      size: 2,
      opacity: (hints.raw_marker_opacity != null) ? hints.raw_marker_opacity : 0.25
    }},
    name: 'Raw',
  }};

  var connectBins = (hints.connect_bins !== false);
  var maxGap = (hints.max_connect_phase_gap != null) ? hints.max_connect_phase_gap : 0.02;
  var showErrorBars = (hints.show_error_bars === true);
  var errStride = Math.max(1, (hints.error_bar_stride != null) ? hints.error_bar_stride : 1);
  var styleMode = (hints.style_mode != null) ? hints.style_mode : 'normal';

  var binnedMode;
  if (styleMode === 'normal') {{
    binnedMode = 'lines';
  }} else {{
    binnedMode = connectBins ? 'lines+markers' : 'markers';
  }}
  var lineWidth = (hints.binned_line_width != null) ? hints.binned_line_width : 1.5;
  var markerSize = (hints.binned_marker_size != null) ? hints.binned_marker_size : 5;
  var lineOpacity = (styleMode === 'normal') ? 0.8 : 1.0;

  var binX = [];
  var binY = [];
  if (connectBins) {{
    for (var i = 0; i < d.bin_centers.length; i++) {{
      if (i > 0 && (d.bin_centers[i] - d.bin_centers[i - 1]) > maxGap) {{
        // Break line across sparse/gap regions to avoid misleading zig-zags.
        binX.push(null);
        binY.push(null);
      }}
      binX.push(d.bin_centers[i]);
      binY.push(d.bin_flux[i]);
    }}
  }} else {{
    binX = d.bin_centers.slice();
    binY = d.bin_flux.slice();
  }}

  var binErr = [];
  for (var i = 0; i < d.bin_err.length; i++) {{
    var e = d.bin_err[i];
    if (!showErrorBars || e === null || (i % errStride) !== 0) {{
      binErr.push(0);
    }} else {{
      binErr.push(e);
    }}
  }}

  var traceBin = {{
    x: binX, y: binY,
    mode: binnedMode,
    type: 'scatter',
    connectgaps: false,
    opacity: lineOpacity,
    line: {{ color: '{_BIN_COLOR}', width: lineWidth }},
    marker: {{ color: '{_BIN_COLOR}', size: markerSize }},
    error_y: {{
      type: 'data',
      array: binErr,
      visible: showErrorBars
    }},
    name: 'Binned (' + d.bin_minutes + ' min)',
  }};

  var shapes = [
    {{
      type: 'line',
      x0: d.primary_phase, x1: d.primary_phase,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: {{ color: '{_TRANSIT_COLOR}', width: 1.5, dash: 'dot' }},
    }},
    {{
      type: 'line',
      x0: d.secondary_phase, x1: d.secondary_phase,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: {{ color: '{_ACCENT}', width: 1.5, dash: 'dot' }},
    }},
  ];

  var ann = [];
  if (d.strongest_dip_phase !== null && d.strongest_dip_flux !== null) {{
    ann.push({{
      x: d.strongest_dip_phase,
      y: d.strongest_dip_flux,
      xref: 'x',
      yref: 'y',
      text: 'strongest dip',
      showarrow: true,
      arrowcolor: '{_TEXT_DIM}',
      font: {{ color: '{_TEXT_DIM}', size: 10 }},
      ax: 0,
      ay: -30,
    }});
  }}

  var yRange = percentileRange(d.flux, 2, 98, 0.12);
  var yaxisCfg = Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
    title: {{ text: 'Normalized Flux', standoff: 8 }},
  }});
  if (yRange) yaxisCfg.range = yRange;

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Orbital Phase', standoff: 8 }},
      range: [-0.5, 0.5],
    }}),
    yaxis: yaxisCfg,
    shapes: shapes,
    annotations: ann,
  }});

  Plotly.newPlot('secondary-scan-plot', [traceRaw, traceBin], layout, {{responsive: true}});
}})();"""


def _oot_context_js() -> str:
    return f"""\
(function() {{
  var d = REPORT.oot_context;
  if (!d) return;
  var hasHist = d.hist_centers && d.hist_centers.length > 0;
  var hasSample = d.flux_residual_ppm_sample && d.flux_residual_ppm_sample.length > 0;
  if (!hasHist && !hasSample) return;

  var traces = [];
  if (hasSample) {{
    traces.push({{
      x: d.sample_indices,
      y: d.flux_residual_ppm_sample,
      mode: 'markers',
      type: 'scattergl',
      marker: {{ color: '{_OOT_COLOR}', size: 2, opacity: 0.35 }},
      name: 'OOT sample',
      xaxis: 'x2',
      yaxis: 'y2',
    }});
  }}
  if (hasHist) {{
    traces.push({{
      x: d.hist_centers,
      y: d.hist_counts,
      type: 'bar',
      marker: {{ color: '{_BIN_COLOR}', opacity: 0.7 }},
      name: 'Histogram',
      xaxis: 'x',
      yaxis: 'y',
    }});
  }}

  var annText = '';
  if (d.robust_sigma_ppm !== null) annText += 'robust sigma: ' + d.robust_sigma_ppm.toFixed(0) + ' ppm';
  if (d.mad_ppm !== null) annText += (annText ? ' | ' : '') + 'MAD: ' + d.mad_ppm.toFixed(0) + ' ppm';
  if (d.std_ppm !== null) annText += (annText ? ' | ' : '') + 'std: ' + d.std_ppm.toFixed(0) + ' ppm';
  if (d.n_oot_points !== null) annText += (annText ? ' | ' : '') + 'N=' + d.n_oot_points;

  var layout = Object.assign({{}}, {_plotly_layout_defaults()}, {{
    grid: {{rows: 1, columns: 2, pattern: 'independent'}},
    xaxis: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'OOT Residual (ppm)', standoff: 8 }},
    }}),
    yaxis: Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
      title: {{ text: 'Count', standoff: 8 }},
    }}),
    xaxis2: Object.assign({{}}, {_plotly_layout_defaults()}.xaxis, {{
      title: {{ text: 'Sample Index (OOT)', standoff: 8 }},
    }}),
    yaxis2: Object.assign({{}}, {_plotly_layout_defaults()}.yaxis, {{
      title: {{ text: 'OOT Residual (ppm)', standoff: 8 }},
    }}),
    annotations: annText ? [{{
      text: annText,
      xref: 'paper',
      yref: 'paper',
      x: 0.5,
      y: 1.1,
      showarrow: false,
      font: {{ color: '{_TEXT_DIM}', size: 11 }},
    }}] : [],
  }});

  Plotly.newPlot('oot-context-plot', traces, layout, {{responsive: true}});
}})();"""
