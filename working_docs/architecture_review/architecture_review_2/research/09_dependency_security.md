# Dependency Security Audit

**Date:** 2026-01-14
**Package:** bittr-tess-vetter
**Python:** 3.11-3.12

---

## 1. Dependency Inventory

### Core Dependencies

| Package | Version Constraint | Current Status | License |
|---------|-------------------|----------------|---------|
| numpy | >=1.24.0,<2.4.0 | Active | BSD-3-Clause |
| scipy | >=1.10.0 | Active | BSD-3-Clause |
| pydantic | >=2.0.0 | Active | MIT |
| astropy | >=5.0.0 | Active | BSD-3-Clause |
| setuptools | >=70.0.0 | Active | MIT |
| requests | >=2.31.0 | Active | Apache-2.0 |
| transitleastsquares | >=1.32 | Low activity | MIT |
| numba | >=0.63.0 | Active | BSD-2-Clause |
| emcee | >=3.1.6 | Inactive | MIT |
| arviz | >=0.23.0 | Active | Apache-2.0 |
| ldtk | >=1.8.5 | Limited activity | GPL-3.0 |

### Optional Dependencies (triceratops extra)

| Package | Version Constraint | License | Risk |
|---------|-------------------|---------|------|
| lightkurve | >=2.0.0 | MIT | Low |
| pytransit | >=2.2 | **GPL-2.0** | **HIGH** |
| mechanicalsoup | >=0.12.0 | MIT | Low |
| seaborn | >=0.11.1 | BSD-3-Clause | Low |
| pyrr | >=0.10.3 | MIT | Low |
| celerite | >=0.4.0 | MIT | Low |
| corner | >=2.2.1 | BSD-2-Clause | Low |
| pandas | >=0.23.4 | BSD-3-Clause | Low |
| matplotlib | >=3.5.1 | PSF-based | Low |
| astroquery | >=0.4.6 | BSD-3-Clause | Low |

---

## 2. Known Vulnerabilities (CVEs)

### CRITICAL: setuptools (CVE-2024-6345, CVE-2025-47273)

**Status:** MITIGATED by version constraint

- **CVE-2024-6345** (CVSS 7.5 HIGH): Command injection via package URL in versions < 70.0
- **CVE-2025-47273** (HIGH): Path traversal in PackageIndex, fixed in 78.1.1

**Current constraint:** `>=70.0.0` - protects against CVE-2024-6345 but NOT CVE-2025-47273

**Recommendation:** Update to `>=78.1.1`

```toml
# pyproject.toml
"setuptools>=78.1.1",
```

### MEDIUM: requests (CVE-2024-47081)

**Status:** MITIGATED by version constraint

- **CVE-2024-47081** (CVSS 5.3): .netrc credential leak via malformed URLs, fixed in 2.32.4

**Current constraint:** `>=2.31.0` - does NOT protect against this CVE

**Recommendation:** Update to `>=2.32.4`

```toml
"requests>=2.32.4",
```

### LOW: pydantic (CVE-2024-3772)

**Status:** MITIGATED by version constraint

- **CVE-2024-3772** (CVSS 5.8): ReDoS via crafted email string, fixed in 2.4.0

**Current constraint:** `>=2.0.0` - does NOT guarantee protection

**Recommendation:** Update to `>=2.4.0`

```toml
"pydantic>=2.4.0",
```

### DISPUTED/LOW: numpy, scipy

Historical CVEs (2021-2023) are disputed by maintainers as not practical security risks:
- CVE-2021-33430, CVE-2021-34141 (numpy): Buffer overflow and incomplete string comparison
- CVE-2023-25399, CVE-2023-29824 (scipy): Memory leak and use-after-free

These are theoretical issues requiring privileged access or unusual API usage.

---

## 3. Version Constraint Analysis

### Too Loose (Security Risk)

| Package | Current | Recommendation | Reason |
|---------|---------|----------------|--------|
| setuptools | >=70.0.0 | >=78.1.1 | CVE-2025-47273 |
| requests | >=2.31.0 | >=2.32.4 | CVE-2024-47081 |
| pydantic | >=2.0.0 | >=2.4.0 | CVE-2024-3772 |

### Too Loose (Maintenance Risk)

| Package | Current | Recommendation | Reason |
|---------|---------|----------------|--------|
| pandas | >=0.23.4 | >=2.0.0 | Version 0.x is 5+ years old |
| seaborn | >=0.11.1 | >=0.13.0 | 0.11 is 4+ years old |
| mechanicalsoup | >=0.12.0 | >=1.0.0 | 0.x is legacy |

### Good Constraints

- `numpy>=1.24.0,<2.4.0`: Upper bound prevents NumPy 2.x breaking changes
- `numba>=0.63.0`: Correctly pinned for Python 3.12 compatibility
- `astropy>=5.0.0`: Reasonable minimum

### Constraint Conflicts (triceratops extras)

The optional `[triceratops]` extras have duplicate pins that may conflict:
- `emcee>=3.0.2` vs `emcee>=3.1.6` (core) - core wins
- `numba>=0.52.0` vs `numba>=0.63.0` (core) - core wins
- `arviz>=0.12.1` vs `arviz>=0.23.0` (core) - core wins

**Recommendation:** Remove duplicate pins from `[triceratops]` extras.

---

## 4. Maintenance Status Assessment

### Active & Well-Maintained
- numpy, scipy, pydantic, astropy, numba, arviz, matplotlib, pandas

### Inactive/Low Activity (Monitor)
| Package | Last Release | Weekly Downloads | Risk |
|---------|--------------|------------------|------|
| emcee | Apr 2024 | ~47k | Medium - stable API |
| transitleastsquares | Unknown | Low | Medium - niche astronomy |
| ldtk | Unknown | Limited | Medium - niche |
| celerite | Unknown | Low | Medium - superseded by celerite2 |

### Deprecated Concerns
- **celerite**: Consider migration to `celerite2` which is actively maintained
- **mechanicalsoup**: Still maintained but low activity; consider if web scraping is core functionality

---

## 5. License Compliance

### Project License: MIT

### License Compatibility Matrix

| License | Compatible with MIT? | Action Required |
|---------|---------------------|-----------------|
| MIT | Yes | None |
| BSD-2-Clause | Yes | None |
| BSD-3-Clause | Yes | None |
| Apache-2.0 | Yes | None |
| **GPL-2.0** | **Copyleft** | **See below** |
| **GPL-3.0** | **Copyleft** | **See below** |

### GPL Dependencies (Requires Attention)

#### pytransit (GPL-2.0) - OPTIONAL DEPENDENCY

**Impact:** If distributed together, the combined work must be GPL-licensed.

**Current mitigation:** pytransit is in `[triceratops]` optional extras, not core.

**Recommendation:**
1. Document clearly that `[triceratops]` extras include GPL dependencies
2. Consider if pytransit can be replaced with a permissive alternative
3. Add license notice to installation docs

#### ldtk (GPL-3.0) - CORE DEPENDENCY

**Impact:** ldtk is a CORE dependency with GPL-3.0 license.

**CRITICAL:** This may require the entire package to be GPL-3.0 licensed when distributed.

**Recommendations:**
1. **Immediate:** Verify ldtk license (some sources show MIT, need confirmation)
2. **If GPL-3.0:** Either change project license or make ldtk optional
3. **Document:** Add THIRD_PARTY_NOTICES.md entry for ldtk

---

## 6. Vendored Code: TRICERATOPS+

### Status: PROPERLY ISOLATED

**Location:** `src/bittr_tess_vetter/ext/triceratops_plus_vendor/`

**Documentation:**
- `VENDORED_FROM.md` - Provenance tracking (commit SHA, date, upstream URL)
- `THIRD_PARTY_NOTICES.md` - Root-level attribution

**Isolation:**
- Lazy loading via `__getattr__` (good - no import-time side effects)
- Excluded from ruff linting (appropriate for vendored code)
- Import paths rewritten (proper namespacing)

**License:** MIT (verified in VENDORED_FROM.md and setup.py)

### Concerns

1. **No LICENSE file in vendor directory** - Copy the MIT license text
2. **Stale vendored code** - Last vendored 2026-01-08; establish update schedule
3. **Dependency check** - vendored triceratops uses mechanicalsoup for web scraping (potential security vector)

### Recommendations

```bash
# Add LICENSE file to vendor directory
cp UPSTREAM_LICENSE src/bittr_tess_vetter/ext/triceratops_plus_vendor/LICENSE
```

---

## 7. Security Recommendations

### Priority 1 (Immediate)

1. **Update setuptools constraint:**
   ```toml
   "setuptools>=78.1.1",
   ```

2. **Update requests constraint:**
   ```toml
   "requests>=2.32.4",
   ```

3. **Update pydantic constraint:**
   ```toml
   "pydantic>=2.4.0",
   ```

### Priority 2 (Short-term)

4. **Verify ldtk license** and resolve GPL compliance if needed

5. **Add LICENSE file** to vendored TRICERATOPS+ directory

6. **Clean up duplicate pins** in `[triceratops]` optional dependencies

### Priority 3 (Maintenance)

7. **Modernize legacy version floors:**
   ```toml
   "pandas>=2.0.0",
   "seaborn>=0.13.0",
   "mechanicalsoup>=1.0.0",
   ```

8. **Consider celerite2** migration if celerite features are actively used

9. **Establish vendored code update schedule** (quarterly review)

### Priority 4 (Documentation)

10. **Add GPL notice** to installation docs for `[triceratops]` extras

11. **Create SECURITY.md** with vulnerability reporting process

---

## 8. Recommended pyproject.toml Changes

```toml
dependencies = [
  "numpy>=1.24.0,<2.4.0",
  "scipy>=1.10.0",
  "pydantic>=2.4.0",           # Updated: CVE-2024-3772
  "astropy>=5.0.0",
  "setuptools>=78.1.1",        # Updated: CVE-2025-47273
  "requests>=2.32.4",          # Updated: CVE-2024-47081
  "transitleastsquares>=1.32",
  "numba>=0.63.0",
  "emcee>=3.1.6",
  "arviz>=0.23.0",
  "ldtk>=1.8.5",
]

[project.optional-dependencies]
triceratops = [
  "lightkurve>=2.0.0",
  "pytransit>=2.2",            # GPL-2.0 - document in install docs
  "mechanicalsoup>=1.0.0",     # Updated from 0.12.0
  # Removed duplicates: emcee, numba, arviz (use core versions)
  "seaborn>=0.13.0",           # Updated from 0.11.1
  "pyrr>=0.10.3",
  "celerite>=0.4.0",
  "corner>=2.2.1",
  "pandas>=2.0.0",             # Updated from 0.23.4
  "matplotlib>=3.5.1",
  "astroquery>=0.4.6",
]
```

---

## Summary

| Category | Status | Action Items |
|----------|--------|--------------|
| **CVEs** | 3 actionable | Update setuptools, requests, pydantic |
| **Version Constraints** | Mixed | Tighten minimums for security |
| **Maintenance** | Mostly healthy | Monitor emcee, ldtk, celerite |
| **Licensing** | **Needs review** | Verify ldtk license; document GPL deps |
| **Vendored Code** | Well-isolated | Add LICENSE file |
| **Missing Files** | Yes | Add root LICENSE, SECURITY.md |
