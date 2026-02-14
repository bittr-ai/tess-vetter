# Confidence Semantics and CLI Exit Codes

This document is the canonical contract for interpreting `CheckResult.confidence`
and for handling process-level `btv vet` exit codes.

## Confidence semantics (canonical)

`confidence` is an optional per-check scalar attached to each `CheckResult`.

- Source type: `float | None`
- Intended numeric domain: `[0.0, 1.0]`
- Scope: check-local evidence strength, not a global posterior probability
- Authority: interpret `status` first (`ok`, `skipped`, `error`), then `confidence`

### Status-first interpretation

- `status="ok"`: check executed successfully; `confidence` may be present.
- `status="skipped"`: check did not run due to prerequisites/gating; `confidence` is `None`.
- `status="error"`: check failed internally; `confidence` is `None`.

`confidence` is never a replacement for `status`. In particular:

- a high confidence value does not override an `error` status
- a missing confidence value does not imply failure when status is `skipped`

### What confidence is not

- Not calibrated as a probability that a candidate is a planet.
- Not comparable across all checks as if they were on one shared scale.
- Not a full-candidate verdict by itself.

Use `flags`, `metrics`, and check-specific context to interpret each value.

## `btv vet` CLI exit codes (canonical)

These are process exit codes for command execution, distinct from per-check
`status` values in the JSON payload.

| Exit code | Meaning | Typical trigger |
| --- | --- | --- |
| `0` | Success | Vetting payload emitted successfully (or `--resume` no-op skip). |
| `1` | Input/usage error | Invalid CLI argument shape (for example malformed `--extra-param`, invalid flag combination). |
| `2` | Runtime error | Unhandled runtime failure during execution. |
| `3` | Progress metadata I/O error | Read/write error for `--progress-path`. |
| `4` | Light curve / TPF unavailable | No light curve found, or `--require-tpf` could not load TPF. |
| `5` | Remote timeout | Timeout raised by remote/network operations. |

## `btv vet` JSON summary counters

When `btv vet` emits JSON, the top-level `summary` block includes:

- `n_ok`: checks with `status="ok"`
- `n_failed`: checks with `status="error"`
- `n_skipped`: checks with `status="skipped"`
- `n_flagged`: checks requiring attention despite `ok`/`skipped` status
- `n_network_errors`: skipped checks flagged as transient network failures
  (`SKIPPED:NETWORK_TIMEOUT` or `SKIPPED:NETWORK_ERROR`)

`n_network_errors` is additive context and does not change status semantics.

## Cross-reference contract

- `btv vet --help` points users to `docs/quickstart.rst` and `docs/api.rst`.
- `docs/quickstart.rst` points back here for canonical confidence semantics and
  exit-code handling.
