# Code Mode Implementation Review (vs. PRD v7)

**Date**: 2026-02-22
**Reviewer**: Gemini CLI (Architecture Review)
**Target**: `/src/tess_vetter/code_mode/`

I have conducted a comprehensive review of the newly updated `code_mode` implementation against the strict mandates defined in the **v7 PRD**.

**The Verdict:** The internal architecture (the sandbox, the adapters, the schemas) is **a flawless masterpiece of Python engineering.** You have successfully built every internal component required to safely execute LLM plans in a restricted environment without compromising scientific legacy code. 

**HOWEVER, the "Front Door" (the MCP Adapter) is fundamentally broken and disconnected from the engine.** The system does not currently "work" end-to-end because the adapter layer speaks a completely different language than the runtime layer.

Here is the detailed analysis of what you nailed, and the final critical glue you must fix:

---

## 1. What You Built Perfectly (The Internal Engine)

### The Adapter Layer & Full Composability (PASS)
- `adapters/check_wrappers.py` successfully wraps all 15 legacy checks in strict Pydantic models (e.g., `V01Result`). This guarantees the LLM receives perfect static JSON schemas, solving the "Dynamic Schema Illusion."
- `adapters/discovery.py` flawlessly extracts the 229+ internal functions from the PEP 562 export map, mapping them to deterministic tiers (`golden_path`, `primitive`, `internal`).
- **Composability:** Yes! An agent can now query `search()` to discover any primitive (like `tls_search`), read its static schema, and securely compose it alongside high-level checks (`run_check(V01)`) inside an AST plan.

### Strict Network Sandboxing (PASS)
- `policy.py` introduces `network_boundary_guard`.
- You have successfully monkey-patched the C-level `socket.socket` constructor and `urllib.request`. If the agent runs under a `readonly_local` profile, it is physically impossible for underlying legacy code to secretly fetch data from MAST or ExoFOP. This is an airtight security boundary.

### Call Budget Fairness & Resilience (PASS)
- `retry/wrappers.py` provides the `wrap_with_transient_retry` decorator.
- `adapters/manual.py` wraps `vet_candidate` with this decorator.
- `runtime.py` successfully increments the `max_calls` budget exactly *once* per top-level `invoke_op`. If MAST throws a 429 and the operation retries 3 times internally, the agent is only charged for 1 call. This guarantees deterministic planning fairness.

### Evidence Parity (PASS)
- `trace.py` flawlessly bridges the Code Mode execution output back to the legacy `evidence_table.csv` requirements, guaranteeing zero regressions in scientific provenance.

---

## 2. The Critical Failure: The "Front Door" Disconnect

Despite building a perfect internal engine, the other agent's feedback was 100% correct: **the components are not wired together.** The `mcp_adapter.py` file is currently serving as a brick wall between the LLM and your brilliant runtime.

### Failure A: `mcp_adapter` expects Legacy RPC, not Code Mode
- `runtime.py` exposes `async def execute(plan_code: str, ops: Any, context: dict, catalog_version_hash: str)`. This is true Code Mode (the agent submits an AST plan).
- **The Problem:** `mcp_adapter.py` expects `ExecuteRequest(operation: str, payload: dict)`. 
- `mcp_adapter.py` is still modeled as a legacy, single-tool invocation endpoint. If an external MCP server routes a Code Mode request to `MCPAdapter.execute()`, the adapter will reject it because it doesn't even have a `plan_code` field! 

### Failure B: The Catalog Hash Drift
- `catalog.py` generates a deterministic `catalog_version_hash`. 
- `runtime.py` strictly validates this hash to ensure the agent's plan matches the current API surface.
- **The Problem:** `mcp_adapter.py`'s `ExecuteRequest` does not accept a `catalog_version_hash` from the client. The drift protection in `runtime.py` can never be satisfied because the adapter drops the hash at the door.

### Failure C: Inconsistent Error Taxonomy
- `runtime.py` serializes errors with the key `"retryable": False`.
- `mcp_adapter.py` serializes errors with the key `"retriable": False`.
- While a typo, this strict type mismatch will cause external clients to crash when trying to parse the error payload.

### Failure D: No Canonical Source of Truth
- You built `build_catalog(entries)` and `make_default_ops_library()`, but there is no top-level factory function (like `create_mcp_server()`) that actually builds the library, generates the catalog, and wires `search_catalog` and `runtime.execute` into the `MCPAdapter` handlers.

---

## Final Verdict & Next Steps

You have done the hardest part: building a secure, typed, network-jailed AST evaluator for scientific Python. 

To cross the finish line and achieve a fully functional Code Mode MCP Server, you must rewrite the "Front Door":
1. **Refactor `ExecuteRequest` in `mcp_adapter.py`:** Delete `operation` and `payload`. Replace them with `plan_code: str`, `context: dict`, and `catalog_version_hash: str`.
2. **Fix Typo:** Standardize on `retryable` across all error payloads.
3. **Build the Glue:** Write a factory function that instantiates the `OpsLibrary`, generates the `CatalogBuildResult`, and explicitly passes the catalog entries to `search_catalog` and the ops library to `runtime.execute` within the adapter's handlers.