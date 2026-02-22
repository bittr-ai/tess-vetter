# Security Matrix Tests

`tests/test_security/test_matrix.py` is a fast CI guard for negative-path MCP and code-mode security behavior.

Executable deterministic checks:
- network denial for `search()` when network is not explicitly allowed
- execute request schema rejection for malformed `plan_code/context/catalog_version_hash` inputs
- sandbox-escape denial via AST policy (`import` is blocked with `POLICY_DENIED`)
- output size enforcement (`OUTPUT_LIMIT_EXCEEDED`) under constrained call budget
- catalog hash drift rejection (`CATALOG_DRIFT`) before plan execution begins
- trace determinism basic check for `build_runtime_trace_metadata(...)`

Known gaps tracked as strict TODO `xfail` tests:
- budget fairness across concurrent plans e2e is blocked until runtime exposes deterministic global scheduler hooks and cross-plan fairness counters/telemetry

Constraints:
- tests stay unit-level and avoid real network/filesystem dependencies
- scenarios are additive and explicit (one guard per case)
- suite is a blocking CI signal and should remain quick to execute
