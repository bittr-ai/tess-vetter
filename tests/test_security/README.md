# Security Matrix Tests

`tests/test_security/test_matrix.py` is a fast CI guard for negative-path MCP and code-mode security behavior.

Executable deterministic checks:
- network denial for `search()` when network is not explicitly allowed
- sandbox-required denial for `execute()` when sandboxing is disabled
- payload hash mismatch rejection for `execute()` integrity checks
- sandbox-escape denial via AST policy (`import` is blocked with `POLICY_DENIED`)
- trace determinism basic check for `build_runtime_trace_metadata(...)`

Known gaps tracked as strict TODO `xfail` tests:
- resource bomb enforcement e2e is blocked until runtime emits deterministic per-step memory/CPU telemetry
- budget fairness across concurrent plans e2e is blocked until runtime exposes deterministic scheduler hooks and fairness counters

Constraints:
- tests stay unit-level and avoid real network/filesystem dependencies
- scenarios are additive and explicit (one guard per case)
- suite is a blocking CI signal and should remain quick to execute
