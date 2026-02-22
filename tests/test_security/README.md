# Security Matrix Tests

`tests/test_security/test_matrix.py` is a fast CI guard for negative-path MCP security behavior.

It intentionally checks only deterministic adapter policy gates:
- network denial for `search()` when network is not explicitly allowed
- sandbox guard for `execute()` when sandboxing is disabled
- payload hash mismatch rejection for `execute()` integrity checks

Constraints:
- tests must stay unit-level and avoid real network/filesystem dependencies
- scenarios should remain additive and explicit (one guard per case)
- this suite is a blocking CI signal and should remain quick to execute
