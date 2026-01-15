# Security & Input Validation Review

**Package:** bittr-tess-vetter
**Review Date:** 2025-01-14
**Scope:** Code execution safety, network handling, file path validation, cache security

---

## Executive Summary

The codebase demonstrates reasonable security practices for a scientific data analysis library. The primary security concern is **pickle usage in the persistent cache**, which deserves explicit documentation. No critical vulnerabilities were found, but several areas merit attention for hardening before open-source release.

| Category | Risk Level | Status |
|----------|------------|--------|
| Code Execution (eval/exec) | Low | No dangerous usage found |
| Pickle Deserialization | Medium | Present, documented with noqa |
| Network Request Handling | Low | Properly configured |
| File Path Validation | Low | Basic validation present |
| SQL/Query Injection | Low | Some f-string interpolation |

---

## 1. Code Execution: eval() and exec()

### Findings

**No dangerous eval/exec usage detected.**

The `eval()` pattern found in the codebase is exclusively `mx.eval()` from the MLX framework (Apple's GPU array library), not Python's built-in `eval()`:

```python
# src/bittr_tess_vetter/cli/mlx_quick_vet_cli.py
mx.eval(obs_score, obs_depth_hat, obs_depth_sigma)
```

This is the MLX array synchronization function that forces GPU computation to complete - entirely safe.

**No exec() usage found in the codebase.**

### Recommendation

- Status: **PASS**

---

## 2. Pickle Usage with Untrusted Data

### Findings

**Pickle is used in two locations for caching:**

#### 2.1 PersistentCache (platform/io/cache.py)

```python
# Line 267
value = pickle.load(f)  # noqa: S301

# Line 291
pickle.dump(value, f)  # noqa: S301
```

The cache stores `LightCurveData` objects and computed products to disk. The `# noqa: S301` comments indicate developers are aware of the security implications.

**Risk Assessment:**
- Cache files are written by the application itself
- Cache directory is user-controlled via environment variables (`BITTR_TESS_VETTER_CACHE_DIR`)
- No external/untrusted data is directly pickled

**Potential Attack Vector:**
An attacker with write access to the cache directory could inject malicious pickle files. When the application loads these files, arbitrary code could execute.

#### 2.2 TRICERATOPS Target Cache (validation/triceratops_fpp.py)

```python
# Line 326
return pickle.loads(path.read_bytes())

# Line 344
payload = pickle.dumps(target)
```

Caches TRICERATOPS target objects to avoid repeated expensive queries.

### Recommendations

1. **Document the trust model** - Add explicit documentation stating cache directories should be protected from untrusted writes
2. **Consider safer alternatives** - For new features, prefer JSON/msgpack for serializable data structures
3. **Add cache file integrity checks** - Optional: HMAC signatures on cached pickle files
4. Risk Level: **MEDIUM** (requires local file system access)

---

## 3. Network Request Handling

### 3.1 HTTP Clients

The codebase uses `requests` library for HTTP with proper configurations:

**Gaia TAP Client (platform/catalogs/gaia_client.py):**
```python
response = requests.get(
    f"{self.tap_url}/sync",
    params=params,
    timeout=self.timeout,  # Default 60s
)
response.raise_for_status()
```

**SIMBAD Client (platform/catalogs/simbad_client.py):**
- Same pattern with timeouts and status checking

**ExoFOP Client (platform/catalogs/exofop_toi_table.py):**
```python
timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)  # 10s, 45s
```

**MAST Client (platform/io/mast_client.py):**
- Wraps lightkurve library which handles its own network calls

### 3.2 urllib Usage

Found in three locations:

1. **CatalogSnapshotStore (platform/catalogs/store.py):**
```python
with urllib.request.urlopen(url, timeout=60) as response:
```

2. **TRICERATOPS FPP (validation/triceratops_fpp.py):**
```python
with urlopen(req, timeout=30) as resp:  # noqa: S310
```

The `S310` suppression acknowledges that `urlopen` can be used for arbitrary URLs. The caller controls the URL source (TRILEGAL server).

3. **Vendored TRICERATOPS code:**
- External library code, not authored by this project

### 3.3 Security Properties

| Property | Status |
|----------|--------|
| Timeouts configured | Yes |
| TLS/HTTPS preferred | Yes |
| Certificate validation | Default (enabled) |
| No SSL verification bypass | Correct |
| Retry with backoff | Yes |

### Recommendations

- **PASS** - Network handling is appropriately configured
- Minor: Consider adding User-Agent headers consistently for debugging

---

## 4. File Path Validation

### 4.1 Cache Directory Paths

The cache system validates paths in several ways:

**Environment Variable Expansion:**
```python
# platform/io/cache.py
explicit = os.getenv("BITTR_TESS_VETTER_CACHE_DIR")
if explicit:
    return Path(explicit).expanduser()
```

Uses `Path.expanduser()` which handles `~` safely.

**Catalog Name/Version Validation (platform/catalogs/store.py):**
```python
def _validate_name(self, name: str) -> None:
    if not name:
        raise ValueError("Catalog name cannot be empty")
    if not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(...)
    if name.startswith(".") or name.startswith("-"):
        raise ValueError(...)
```

This prevents path traversal attempts like `../../../etc/passwd`.

**TPF FITS References:**
- TIC ID must be positive integer
- Sector must be positive integer
- Author is validated against whitelist: `{"spoc", "qlp", "tess-spoc", "tasoc"}`

### 4.2 Missing Validations

**CatalogSnapshotStore URL handling:**
```python
def _download(self, url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
```

- Does not validate URL scheme (http/https/file)
- file:// URLs could read local files (potential for SSRF if URL comes from untrusted input)

In practice, URLs come from configuration, not user input, so risk is low.

### Recommendations

- Status: **PASS** with minor observations
- Consider adding URL scheme validation for defense in depth

---

## 5. SQL/ADQL Query Injection

### Findings

TAP service queries use f-string interpolation:

**Gaia Client:**
```python
query = f"""
SELECT ... FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
"""
```

**SIMBAD Client:**
```python
escaped_id = identifier.replace("'", "''")  # Basic escaping
query = f"""
SELECT TOP 1 main_id, ra, dec, otype
FROM basic
WHERE main_id = '{escaped_id}'
   OR oid IN (SELECT oidref FROM ident WHERE id = '{escaped_id}')
"""
```

**Exoplanet Archive:**
```python
safe_hostname = hostname.replace("'", "''")
query = f"""WHERE (hostname LIKE '{safe_hostname}%'..."""
```

### Risk Assessment

- Source IDs are validated as integers before interpolation
- String inputs use basic quote escaping
- TAP/ADQL services are read-only astronomical databases
- Even if injection succeeded, attacker could only read public astronomy data

### Recommendations

- **LOW RISK** - Validated integer inputs and read-only targets
- Consider using parameterized queries if TAP client libraries support them
- Current quote escaping is sufficient for the use case

---

## 6. Subprocess Usage

### Findings

CLI modules are designed for subprocess-safe execution:

```python
# cli/mlx_bls_search_cli.py
"""MLX BLS-like search CLI (subprocess-safe)."""
```

These are invoked via `python -m bittr_tess_vetter.cli.xxx` pattern, not through shell interpolation.

**No usage of:**
- `subprocess.Popen(..., shell=True)`
- `os.system()`
- Shell command injection vectors

### Recommendations

- Status: **PASS**

---

## 7. Environment Variable Handling

### Findings

Environment variables used:
- `BITTR_TESS_VETTER_CACHE_DIR` - Cache directory override
- `BITTR_TESS_VETTER_CACHE_ROOT` - Cache root directory
- `ASTRO_ARC_CACHE_ROOT` - Legacy cache root (ExoFOP)

All are used for path configuration, not for command execution or secrets.

### Recommendations

- Status: **PASS**
- Document expected environment variables in README

---

## 8. Dependency Security Considerations

### External Libraries

| Dependency | Security Notes |
|------------|----------------|
| requests | Well-maintained, handles TLS properly |
| astropy | Core astronomy library, widely audited |
| lightkurve | NASA-maintained TESS data library |
| numpy | Foundational, no known issues |
| pydantic | Input validation, helps security |
| mlx | Apple's GPU framework (optional) |

### Vendored Code

**triceratops_plus_vendor/**
- Third-party astronomy code vendored for FPP calculation
- Contains its own `urlopen` usage
- LICENSE file present

### Recommendations

- Consider using `safety` or `pip-audit` in CI
- Pin dependency versions in production deployments

---

## 9. Sensitive Data Handling

### Findings

The codebase processes:
- TIC IDs (public catalog identifiers)
- Coordinates (RA/Dec, public data)
- Light curve data (from public MAST archive)
- Derived transit parameters

**No handling of:**
- User credentials
- API keys (services are public/anonymous)
- Personally identifiable information

### Recommendations

- Status: **PASS**
- The `.env` file check in git is already configured

---

## 10. Summary of Recommendations

### Before Open-Source Release

1. **Document pickle cache trust model** in README or security policy
2. **Add security policy** (SECURITY.md) with:
   - How to report vulnerabilities
   - Scope of security concerns
   - Trust model for cache directories

### Future Enhancements

3. Consider JSON/msgpack as pickle alternative for new caching
4. Add URL scheme validation in `CatalogSnapshotStore._download()`
5. Set up automated dependency scanning in CI

### No Action Required

- Network handling is properly configured
- File path validation is adequate
- No dangerous eval/exec patterns
- No command injection vectors

---

## Appendix: Security-Relevant File Locations

| File | Security Concern |
|------|-----------------|
| `src/bittr_tess_vetter/platform/io/cache.py` | Pickle serialization |
| `src/bittr_tess_vetter/validation/triceratops_fpp.py` | Pickle caching, urlopen |
| `src/bittr_tess_vetter/platform/catalogs/store.py` | URL download, path handling |
| `src/bittr_tess_vetter/platform/catalogs/gaia_client.py` | TAP queries |
| `src/bittr_tess_vetter/platform/catalogs/simbad_client.py` | TAP queries |
| `src/bittr_tess_vetter/platform/catalogs/exoplanet_archive.py` | TAP queries |
| `src/bittr_tess_vetter/pixel/tpf_fits.py` | FITS file handling |
