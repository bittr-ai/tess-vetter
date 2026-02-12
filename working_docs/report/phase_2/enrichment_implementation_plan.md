# Enrichment Implementation Plan (Remaining Work)

## Context
Scaffold is complete:
- `generate_report(..., include_enrichment, enrichment_config)` exists.
- `ReportData.enrichment` exists with domain-named blocks.
- Deterministic skipped scaffold blocks are returned when enabled.

This plan covers the remaining implementation from scaffold to production-ready enrichment.

## Delivery Strategy
Implement in small PRs with strict regression gates:
1. Catalog context execution
2. Pixel diagnostics execution
3. Follow-up context + artifact references
4. Hardening and batch robustness

Each PR must keep LC-only outputs unchanged when enrichment is disabled.

## PR-1: Catalog Context (First Real Block)

## Scope
- Replace catalog scaffold block with real execution path.
- Run catalog-tier checks using `VettingPipeline` subset (`CheckTier.CATALOG`).
- Normalize outputs into `EnrichmentBlockData`.

## Files (expected)
- `src/bittr_tess_vetter/api/generate_report.py`
- new helper module (recommended): `src/bittr_tess_vetter/api/report_enrichment.py`
- `src/bittr_tess_vetter/report/_data.py` (if small schema refinements needed)
- tests:
  - `tests/test_api/test_generate_report.py`
  - new: `tests/test_api/test_report_enrichment_catalog.py`

## Required behavior
- Respect `network`, timeout, and budget flags.
- Status mapping:
  - `ok` when minimal catalog payload exists
  - `skipped` on explicit gating (offline/no-coordinates/etc.)
  - `error` on attempted execution failure
- Populate:
  - `checks` map
  - `flags`
  - `quality.is_degraded`
  - `provenance` (inputs, timings, warnings, budget application)

## Acceptance criteria
- `include_enrichment=True` + network enabled gives non-scaffold `catalog_context`.
- offline mode yields deterministic `skipped` with reason flags.
- `json.dumps(report.to_json())` always succeeds.

## PR-2: Pixel Diagnostics (TPF-Aware)

## Scope
- Replace pixel scaffold block with real execution.
- Add TPF acquisition policy in API orchestration.
- Run pixel-tier checks via pipeline subset (`CheckTier.PIXEL`).

## Required behavior
- Honor:
  - `fetch_tpf`
  - `tpf_sector_strategy` (`best|all|requested`)
  - `sectors_for_tpf`
  - `max_pixel_points`
- Implement deterministic `best` sector scoring and tie-breaks.
- Include provenance with selected sector(s), score components, and budget/truncation metadata.

## Tests
- new: `tests/test_api/test_report_enrichment_pixel.py`
- explicit cases:
  - no TPF available -> `skipped` with flags
  - requested sectors path
  - deterministic `best` choice
  - budget truncation invariance

## Acceptance criteria
- Pixel block transitions from scaffold to real payload for TPF-available targets.
- Deterministic output under fixed inputs/config.

## PR-3: Follow-up Context + Artifact References

## Scope
- Replace follow-up scaffold with metadata/reference payload.
- Keep artifacts external by default; include inline only when explicitly enabled.

## Required behavior
- Populate follow-up metadata references and availability flags.
- If `include_phase2_artifacts_inline` equivalent remains disabled, emit pointers only.
- Preserve JSON-safety and bounded payload size.

## Tests
- `tests/test_api/test_report_enrichment_followup.py`
- verify reference-only default behavior
- verify explicit inline gate behavior (if enabled)

## Acceptance criteria
- follow-up block returns structured context with provenance.
- no large binary payloads by default.

## PR-4: Hardening + Runtime Controls

## Scope
- Enforce total/per-request network budgets and concurrency limits.
- Finalize fail-open/fail-closed semantics.
- Add regression and smoke coverage for batch-scale stability.

## Required behavior
- timeout exhaustion produces deterministic flags/statuses.
- `fail_open=True`: report returns with block-level errors captured.
- `fail_open=False`: first block failure aborts API call.
- no LC metric drift when enrichment toggled on/off.

## Tests
- Extend:
  - `tests/test_api/test_generate_report.py`
  - `tests/test_report/test_report.py`
- Add:
  - timeout budget tests
  - fail-open vs fail-closed tests
  - LC regression invariance checks
  - JSON serialization stress tests

## Cross-Cutting Standards

## Naming
- Keep domain names only:
  - `enrichment`
  - `pixel_diagnostics`
  - `catalog_context`
  - `followup_context`
- No roadmap naming in runtime schema.

## Serialization
- All enrichment outputs must be JSON-primitive-safe.
- NaN/Inf scrubbed to `None`.
- Binary data excluded unless explicitly requested.

## Provenance minimum per block
- data inputs used (IDs/sectors/query params)
- run timing and budget application
- warning list
- version identifiers for code/schema

## Suggested Execution Order
1. PR-1 Catalog Context
2. PR-2 Pixel Diagnostics
3. PR-3 Follow-up Context
4. PR-4 Hardening

## Definition of Done (Overall)
- All three enrichment blocks are real (no scaffold placeholders in normal runs).
- Behavior is deterministic and bounded by config budgets.
- LC-only report path remains backward-stable when enrichment disabled.
- Test suite covers gating, errors, budgets, serialization, and invariance.
