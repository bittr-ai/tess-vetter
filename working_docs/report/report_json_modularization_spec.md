# Report JSON Modularization Spec

## 1) Problem and Goals

Current `ReportData.to_json()` mixes two concerns in one payload:
- decision/triage state (`checks`, `lc_summary`, `enrichment`, etc.)
- plotting payloads (large arrays for Plotly)

Goals:
- Split JSON into `summary` and `plot_data` domains.
- Keep rendering behavior unchanged.
- Support phased migration with low break risk.
- Enable independent hashing/caching for each domain.

Non-goals:
- changing scientific calculations
- redesigning plots

## 2) Proposed Schema

Top-level shape:

```json
{
  "schema_version": "1.1.0",
  "summary": { ... },
  "plot_data": { ... },
  "payload_meta": {
    "summary_version": "1",
    "plot_data_version": "1",
    "summary_hash": "...",
    "plot_data_hash": "..."
  }
}
```

### 2.1 `summary` contract (no heavy arrays)

`summary` includes:
- identity: `tic_id`, `toi`
- run metadata: `checks_run`
- inputs: `ephemeris`, `input_depth_ppm`, `stellar`
- metrics: `lc_summary`, `bundle_summary`
- checks projection: `checks`
- enrichment envelope: `enrichment`
- robustness scalar summary only: `lc_robustness_summary` (optional)

#### Checks projection (strict)

`summary.checks[check_id]` must include only:
- `id`, `name`, `status`, `confidence`
- `flags`, `notes`
- `metrics`
- `provenance`

Must exclude:
- `raw.plot_data`
- any array-like heavy payloads

Per-check plot payloads move to:
- `plot_data.check_overlays[check_id]` (optional)

### 2.2 `plot_data` contract

`plot_data` includes plot-ready payload blocks:
- `full_lc`
- `phase_folded`
- `per_transit_stack`
- `local_detrend`
- `odd_even_phase`
- `secondary_scan`
- `oot_context`
- `timing_series`
- `alias_summary`
- `check_overlays` (optional map keyed by check ID)

Rules:
- JSON primitives only
- render hints/quality flags allowed
- no check decision envelope duplication unless needed for plotting

## 3) Legacy to Modular Mapping

| Legacy key | Modular key | Owner |
|---|---|---|
| `version` | `schema_version` | top-level |
| `tic_id` | `summary.tic_id` | summary |
| `toi` | `summary.toi` | summary |
| `ephemeris` | `summary.ephemeris` | summary |
| `input_depth_ppm` | `summary.input_depth_ppm` | summary |
| `stellar` | `summary.stellar` | summary |
| `lc_summary` | `summary.lc_summary` | summary |
| `checks_run` | `summary.checks_run` | summary |
| `checks` | `summary.checks` (projected) | summary |
| `bundle_summary` | `summary.bundle_summary` | summary |
| `enrichment` | `summary.enrichment` | summary |
| `full_lc` | `plot_data.full_lc` | plot_data |
| `phase_folded` | `plot_data.phase_folded` | plot_data |
| `per_transit_stack` | `plot_data.per_transit_stack` | plot_data |
| `local_detrend` | `plot_data.local_detrend` | plot_data |
| `odd_even_phase` | `plot_data.odd_even_phase` | plot_data |
| `secondary_scan` | `plot_data.secondary_scan` | plot_data |
| `oot_context` | `plot_data.oot_context` | plot_data |
| `timing_series` | `plot_data.timing_series` | plot_data |
| `alias_summary` | `plot_data.alias_summary` | plot_data |
| `lc_robustness` | split: `summary.lc_robustness_summary` + optional `plot_data.lc_robustness_plot` | split |

Dual-mode precedence rule:
- Renderer/API should prefer modular keys when both exist.

## 4) Canonical Hashing (Deterministic)

`summary_hash` and `plot_data_hash` are computed after normalization:
1. Convert tuples to lists recursively.
2. Scrub non-finite floats (`NaN`, `+Inf`, `-Inf`) to `null` recursively.
3. Serialize using JSON with:
   - `sort_keys=True`
   - `separators=(",", ":")`
   - UTF-8 bytes
   - `ensure_ascii=False`
4. Hash bytes with SHA-256.

Float policy:
- keep numeric values as JSON numbers (no string formatting pass).
- non-finite values must be scrubbed before serialization/hashing.

Add golden tests for hash stability on fixed fixtures.

## 5) Migration / Compatibility Plan

### Phase A: Dual-write
- `to_json(contract="dual")` emits modular keys plus legacy keys.
- legacy keys remain read-compatible.

### Phase B: Renderer cutover
- renderer reads modular first, legacy fallback.
- add explicit payload-based rendering seam (see Section 6).

### Phase C: API contract option
- `generate_report(..., json_contract="dual|modular|legacy")`
- default `dual` initially.

### Phase D: Deprecation
- remove legacy plot keys once all first-party consumers use modular.
- bump `schema_version` and changelog.

## 6) Renderer Contract (`report/_render_html.py`)

Current function signature is `render_html(report: ReportData, ...)`.
To test modular and legacy payloads directly, add:
- `render_html_from_payload(payload: dict[str, Any], *, title: str | None = None) -> str`

Then:
- `render_html(report)` calls `render_html_from_payload(report.to_json(...))`.
- tests can pass modular-only or legacy-only payload dicts directly.

Read policy in renderer:
- `summary = payload.get("summary", payload)`
- `plots = payload.get("plot_data", payload)`

## 7) API Implications (`api/generate_report.py`)

Add `json_contract` option and return shape behavior:
- `legacy`: old top-level structure only.
- `modular`: `schema_version`, `summary`, `plot_data`, `payload_meta` only.
- `dual`: both structures present.

Transport split note:
- This spec does **not** add separate endpoints yet.
- “summary-first/lazy plots” is enabled by contract shape + future endpoint work.
- follow-up API spec should define separate endpoint/fields if required.

## 8) Validation/Test Plan

Serialization tests:
- modular shape present and JSON-safe
- checks projection excludes heavy arrays
- dual mode parity checks between legacy and modular values

Hash tests:
- deterministic hash golden tests for `summary_hash` and `plot_data_hash`
- tuple/list and NaN/Inf normalization tests

Renderer tests:
- `render_html_from_payload()` with modular-only payload
- `render_html_from_payload()` with legacy-only payload
- `render_html()` unchanged for `ReportData`

API tests:
- `json_contract` modes (`legacy`, `modular`, `dual`) return expected keys

Invariance tests:
- scientific check metrics unchanged by serialization refactor

## 9) Risks and Open Questions

Risks:
- dual mode temporarily increases payload size
- drift between legacy and modular during migration
- clients accidentally reading mixed sources

Open questions:
- exact split strategy for `lc_robustness`
- whether to expose separate transport endpoints in same release or follow-up

## 10) Implementation Checklist (File-Level)

1. `src/bittr_tess_vetter/report/_data.py`
- add `_to_summary_json()` with check projection
- add `_to_plot_data_json()`
- add deterministic normalization + hashing helpers
- add mode-aware `to_json(contract=...)`

2. `src/bittr_tess_vetter/report/_render_html.py`
- add `render_html_from_payload()`
- migrate internals to modular-first reads with fallback

3. `src/bittr_tess_vetter/api/generate_report.py`
- add `json_contract` parameter
- wire `report.to_json(contract=...)`

4. Tests
- `tests/test_report/test_report.py`: modular serialization, projection, hashing
- `tests/test_report/test_render_html.py`: payload seam + fallback behavior
- `tests/test_api/test_generate_report.py`: contract mode coverage

5. Docs
- update report README and migration note under `working_docs/report`

## 11) Recommended Rollout

- Release N: dual default + renderer modular-first.
- Release N+1: switch default to modular.
- Release N+2: remove legacy plot keys.
