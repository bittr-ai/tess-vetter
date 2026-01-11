# Module Review: `network/timeout.py` + `errors.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this matters

These modules are not astrophysics calculations, but they affect reliability and interpretability:
- timeouts bound external dependency hangs (Gaia/TAP, TRICERATOPS, MAST),
- a stable error taxonomy helps hosts distinguish “missing data” from “bad physics”.

## File: `network/timeout.py`

- Provides `network_timeout(seconds, operation=...)` context manager:
  - Uses Unix `SIGALRM` timer when supported.
  - On unsupported platforms, logs a debug message and runs without enforcing a timeout.
- Defines timeouts used by network-heavy clients (`MAST_QUERY_TIMEOUT`, `TRICERATOPS_*`).
- Raises `NetworkTimeoutError(operation, seconds)` on timeout.

## File: `errors.py`

- Provides a minimal, stable error envelope for hosts:
  - `ErrorType` enum (CACHE_MISS, INVALID_REF, INVALID_DATA, INTERNAL_ERROR)
  - `ErrorEnvelope` pydantic model with `type`, `message`, and `context`
  - `make_error(...)` helper.

## Fixes / follow-ups

No physics correctness issues identified in these modules.

