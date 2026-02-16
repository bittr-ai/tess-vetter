# Spec: Manual-TOI Composable Pipeline MVP

## Date
2026-02-16

## Context
For MVP, we want repeatable vetting pipelines without automatic target classification.
The operator (human or agent) will provide explicit TOIs and choose a pipeline profile.

This spec also defines a composition-injection model so agents in other repositories can define their own pipeline steps and run them through `btv` without forking this repo.

## Why Manual TOIs Simplifies MVP
Manual TOI selection removes three complex concerns:
- No star-type auto-identification/routing model required.
- No discovery/indexing/scoring stage.
- No ambiguity about why a target took a given path.

Result: deterministic orchestration and faster release.

## Goals
- Run one named pipeline over an explicit TOI list.
- Produce stable, machine-readable artifacts per TOI and per step.
- Support reusable built-in profiles in this repo.
- Support injected compositions from external repos/agents.
- Keep science logic in existing commands/APIs; pipeline layer is orchestration only.

## Non-Goals (MVP)
- Automatic star-type inference.
- New scientific checks or threshold inventions.
- Replacing individual diagnostic CLIs.
- Distributed execution orchestration.

## Proposed CLI
Primary command:
- `btv pipeline run --profile <name> --toi TOI-5807.01 --toi TOI-4846.01 --out-dir <dir> [options]`

Composition injection command path:
- `btv pipeline run --composition-file <path> --toi ... --out-dir <dir> [options]`

Optional stdin injection:
- `btv pipeline run --composition-file - --toi ... --out-dir <dir>`

Mutual exclusivity:
- Exactly one of `--profile` or `--composition-file` is required.

Common options:
- `--network-ok/--no-network`
- `--continue-on-error/--fail-fast`
- `--max-workers <int>` (default 4 for MVP; configurable)
- `--cache-dir <path>`
- `--report-seed-file <path>` (optional global seed)

## Output Contract
Directory layout:
- `<out-dir>/run_manifest.json`
- `<out-dir>/evidence_table.json`
- `<out-dir>/evidence_table.csv`
- `<out-dir>/<toi>/pipeline_result.json`
- `<out-dir>/<toi>/steps/<nn>_<step_id>.json`
- `<out-dir>/<toi>/logs/<step_id>.stderr.log`
- `<out-dir>/<toi>/checkpoints/<nn>_<step_id>.done.json`

`pipeline_result.json` (per TOI):
- `schema_version`
- `toi`
- `profile_id`
- `composition_digest`
- `status` (`ok|failed|partial`)
- `steps` (ordered summary)
- `verdict` (optional, if composition maps one)
- `artifacts`
- `provenance` (runner version, timestamps, options)

`run_manifest.json` (run-level):
- input TOIs, profile/composition metadata, counts, failures, elapsed time.

`evidence_table.json` and `evidence_table.csv` (run-level, one row per TOI):
- `toi`
- `model_compete_verdict`
- `systematics_verdict`
- `ephemeris_verdict`
- `timing_verdict`
- `localize_host_action_hint`
- `dilution_n_plausible_scenarios`
- `fpp`
- `concern_flags` (aggregated list/string)

Resume/retry markers (MVP):
- Step completion marker is `<out-dir>/<toi>/checkpoints/<nn>_<step_id>.done.json`.
- Marker payload includes at least: `toi`, `step_id`, `step_index`, `status`, `started_at`, `finished_at`, `attempt`, `step_output_path`, `input_fingerprint`.
- On retry/resume, executor skips steps with valid completion markers and matching `input_fingerprint`; downstream incomplete/invalid steps rerun.

## Composition Model

### 1) Built-in profile registry
Profiles stored in-repo:
- `src/bittr_tess_vetter/pipeline_profiles/*.yaml`

Registry API:
- `list_profiles()`
- `get_profile(profile_id)`
- `validate_profile(profile)`

### 2) Injected composition
External agents define composition as YAML/JSON matching the same schema.
Runner validates then executes exactly the declared steps.

Injection sources:
- file path (`--composition-file`)
- stdin (`--composition-file -`)

Trust model:
- MVP allows only known `btv` operations (allow-list), no arbitrary shell.

## Composition Schema (MVP)
Top-level:
- `schema_version: "pipeline.composition.v1"`
- `id: <string>`
- `description: <string>`
- `defaults: { network_ok, continue_on_error, flux_type, ... }`
- `steps: [Step...]`
- `final_mapping: { verdict_from, verdict_source_from }` (optional)

Step:
- `id: <string>`
- `op: <allow-listed operation id>`
- `ports`:
  - named outputs published by the step (primary composition mechanism)
- `inputs:`
  - literals
  - named-port references: `{"port": "<step_id>.<port_name>"}`
  - report-file auto-wiring: `{"report_from": "<step_id>"}` resolves to that step's canonical report artifact path
  - JSONPath references (kept for compatibility): `{"$ref": "steps.<step_id>.<jsonpath>"}`
- `outputs:`
  - named artifact label
- `on_error: continue|fail`

Allow-listed `op` examples (MVP):
- `vet`
- `measure-sectors`
- `fit`
- `report`
- `activity`
- `model_compete`
- `timing`
- `systematics_proxy`
- `ephemeris_reliability`
- `resolve_stellar`
- `resolve_neighbors`
- `localize_host`
- `dilution`
- `detrend_grid`
- `fpp`

## Minimal Built-in Profiles (MVP)
Ship 3-4 profiles only:
1. `triage_fast`
- vet (with lc summary) -> activity -> model_compete -> systematics_proxy -> report

2. `host_localization`
- resolve_neighbors -> localize_host -> dilution -> report

3. `fpp_validation_fast`
- vet -> optional stellar resolve -> fpp(fast preset) -> report

4. `full_vetting`
- measure-sectors -> vet -> fit -> activity -> model_compete -> systematics_proxy -> ephemeris_reliability -> timing -> resolve_stellar -> resolve_neighbors -> localize_host -> dilution -> fpp -> report

Each profile must be pure composition data, no hidden policy code.

## Architecture
New modules:
- `src/bittr_tess_vetter/pipeline_composition/schema.py`
- `src/bittr_tess_vetter/pipeline_composition/registry.py`
- `src/bittr_tess_vetter/pipeline_composition/executor.py`
- `src/bittr_tess_vetter/pipeline_composition/ref_resolver.py`
- `src/bittr_tess_vetter/cli/pipeline_run_cli.py`

Execution strategy:
- In-process calls to existing API/CLI adapters where possible.
- Reuse current input-resolution utilities (`--report-file`, sector handling) rather than duplicate.
- Default concurrent TOI execution with bounded workers and dependency-safe step ordering.
- Rate-limit-aware behavior is required: exponential backoff with jitter for retryable network/API errors, plus simple batching where upstream services require it.

Dependency boundary:
- Composition layer orchestrates.
- Existing command/API modules retain scientific logic and metrics semantics.

## Composition Injection for Other Repos
Yes, this is feasible and low-friction:
- External repo/agent writes composition YAML.
- Calls local `btv pipeline run --composition-file ... --toi ...`.
- Gets stable artifacts without modifying this repo.

Optional future enhancer:
- `btv pipeline validate --composition-file ...` for CI preflight.

## Risks
- Drift between op allow-list and evolving CLI/API signatures.
- Overly permissive references causing brittle pipelines.
- Hidden policy creeping into profile data.

Mitigations:
- Strict schema validation with versioning.
- Contract tests per op adapter.
- Keep final verdict mapping explicit and optional.

## Testing Strategy
Unit:
- schema validation
- reference resolution
- step dependency ordering
- error propagation behavior

Contract:
- each `op` adapter returns canonical step payload shape.

Integration:
- one built-in profile over 2-3 known TOIs.
- injected composition fixture from `tests/fixtures/compositions/`.

Regression:
- verify all generated artifacts are reproducible with fixed seeds.

## Rollout Plan
P0 (MVP core):
- composition schema + validator
- executor support for MVP allow-listed ops required by shipped built-in profiles
- `btv pipeline run` with manual TOIs
- artifact contracts + tests
- resume/retry semantics with step-level completion markers/checkpoints

P1:
- additional op adapters beyond MVP
- `pipeline validate` preflight command
- profile library expansion

P2:
- optional auto-routing stage (star-type inference) that selects profile before run.

## Success Criteria
- Given explicit TOIs and profile/composition, pipeline is reproducible and deterministic.
- Agents from other repos can inject compositions without touching source.
- No new science logic added to orchestrator layer.
