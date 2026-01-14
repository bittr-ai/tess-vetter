# Naming Conflict Analysis: `catalog.py` vs `catalogs.py`

**Date:** 2026-01-14
**Scope:** `/src/bittr_tess_vetter/api/`

---

## Executive Summary

The API layer contains two modules with confusingly similar names that serve **entirely different purposes**:

| Module | Purpose | Primary Exports |
|--------|---------|-----------------|
| `catalog.py` | Vetting checks (V06-V07) | `vet_catalog()`, `nearby_eb_search()`, `exofop_disposition()` |
| `catalogs.py` | Catalog client wrappers | `GaiaClient`, `SimbadClient`, `ExoplanetArchiveClient`, `CatalogSnapshotStore` |

This is the **only singular/plural naming collision** in the api/ directory.

---

## 1. Module Contents Analysis

### `catalog.py` (341 lines)

**Purpose:** Provides API wrappers for catalog-based vetting checks (V06-V07).

**Exports:**
```python
# Line 127-178: V06 check
def nearby_eb_search(candidate, *, ra_deg, dec_deg, network=False, ...) -> CheckResult

# Line 184-225: V07 check
def exofop_disposition(candidate, *, tic_id, network=False, ...) -> CheckResult

# Line 232-340: Orchestrator
def vet_catalog(candidate, *, tic_id=None, ra_deg=None, ...) -> list[CheckResult]
```

**Internal helpers (not exported):**
- `_convert_result()` - converts internal VetterCheckResult to facade CheckResult
- `_make_skipped_result()` - creates skipped result when network disabled
- `_make_missing_metadata_result()` - creates result for missing metadata
- `_candidate_to_internal()` - converts facade Candidate to internal TransitCandidate

**Imports from:** `bittr_tess_vetter.validation.checks_catalog`

### `catalogs.py` (143 lines)

**Purpose:** Re-exports catalog clients from `bittr_tess_vetter.platform.catalogs`.

**Exports (via `__all__`, 68 items):**
```python
# Client classes
GaiaClient, SimbadClient, ExoplanetArchiveClient

# Storage
CatalogSnapshotStore, CatalogData, CatalogEntry

# Data types
GaiaSourceRecord, GaiaNeighbor, GaiaQueryResult
SimbadQueryResult, SimbadIdentifiers, SimbadSpectralInfo
KnownPlanet, KnownPlanetsResult

# Error types
GaiaTAPError, SimbadTAPError, CatalogNotFoundError, ...

# Functions
query_gaia_by_id_sync(), query_simbad_by_position_sync(), crossmatch(), ...

# ExoFOP
ExoFOPToiTable, fetch_exofop_toi_table, ExoFOPTargetSummary, ...
```

---

## 2. Semantic Distinction

| Aspect | `catalog.py` | `catalogs.py` |
|--------|--------------|---------------|
| **Abstraction level** | High (vetting API) | Low (platform clients) |
| **Purpose** | Run vetting checks | Query external databases |
| **Returns** | `CheckResult` objects | Raw catalog data |
| **Network access** | Via `network=` flag | Always requires network |
| **User-facing** | Yes (documented in `__init__.py`) | Partially (not in main `__all__`) |

The naming collision is conceptually coherent but practically confusing:
- `catalog.py` = "run catalog-related **vetting checks**"
- `catalogs.py` = "access external **catalog data sources**"

---

## 3. Import Patterns Across Codebase

### `catalog.py` imports (3 locations):
```python
# src/bittr_tess_vetter/api/__init__.py:413 (TYPE_CHECKING block)
from bittr_tess_vetter.api.catalog import (
    exofop_disposition,
    nearby_eb_search,
    vet_catalog,
)

# src/bittr_tess_vetter/api/vet.py:193 (runtime import)
from bittr_tess_vetter.api.catalog import vet_catalog

# tests/test_api/test_catalog_api.py:3
from bittr_tess_vetter.api.catalog import vet_catalog
```

### `catalogs.py` imports (2 locations):
```python
# tests/test_validation/test_stellar_dilution.py:5
from bittr_tess_vetter.api.catalogs import compute_dilution_factor as ...

# Internal docstring only; no direct imports found in main codebase
```

**Key observation:** `catalogs.py` exports are **not** re-exported via `api/__init__.py`'s `__all__`. Users must know to import directly from `api.catalogs`.

---

## 4. Confusion Risk Assessment

### Scenario 1: New user wants to query Gaia
```python
# User's mental model: "I need catalog data"
from bittr_tess_vetter.api.catalog import GaiaClient  # ImportError!

# Correct:
from bittr_tess_vetter.api.catalogs import GaiaClient
```
**Risk: HIGH** - Singular/plural distinction is not intuitive.

### Scenario 2: User wants to run catalog vetting
```python
# User's mental model: "I need to check against catalogs"
from bittr_tess_vetter.api.catalogs import vet_catalog  # ImportError!

# Correct:
from bittr_tess_vetter.api.catalog import vet_catalog
# Or:
from bittr_tess_vetter.api import vet_catalog  # Recommended
```
**Risk: MEDIUM** - Main API exports `vet_catalog`, reducing confusion.

### Scenario 3: IDE autocomplete
When typing `from bittr_tess_vetter.api.catalog`:
- `catalog` and `catalogs` appear adjacent in completion lists
- No visual cue distinguishes their purposes

**Risk: MEDIUM** - Easy to select wrong module.

### Scenario 4: Code review / maintenance
```python
from bittr_tess_vetter.api.catalog import nearby_eb_search
from bittr_tess_vetter.api.catalogs import GaiaClient
```
Reviewer may question: "Why different modules? Is this intentional?"

**Risk: LOW** - Experienced developers will recognize the pattern.

---

## 5. Resolution Options

### Option A: Rename `catalog.py` to `catalog_checks.py`
**Pros:**
- Clear semantic distinction: `*_checks.py` = vetting modules
- Consistent with internal `checks_catalog.py` in validation/
- No breaking changes to `api.__all__` exports

**Cons:**
- Breaks direct imports: `from bittr_tess_vetter.api.catalog import ...`
- Inconsistent with other check modules (`lc_only.py`, `pixel.py`, `exovetter.py`)

**Migration:**
```python
# Deprecation wrapper in catalog.py:
import warnings
from bittr_tess_vetter.api.catalog_checks import *
warnings.warn("Import from api.catalog_checks", DeprecationWarning)
```

### Option B: Rename `catalogs.py` to `catalog_clients.py`
**Pros:**
- Clear distinction: "clients" vs "checks"
- Less disruptive (catalogs.py has fewer imports)

**Cons:**
- `catalogs` is semantically accurate for a re-export module
- Breaks existing test imports

### Option C: Merge into `catalog.py` with namespacing
```python
# In catalog.py:
from bittr_tess_vetter.api import catalogs as _catalogs

# Re-export under namespace
class clients:
    GaiaClient = _catalogs.GaiaClient
    SimbadClient = _catalogs.SimbadClient
    ...
```
**Pros:**
- Single entry point
- Clear namespacing: `catalog.clients.GaiaClient`

**Cons:**
- Unusual pattern for Python
- Adds indirection

### Option D: Keep as-is + Document
Add clear docstrings and a note in `api/__init__.py`:

```python
# In api/__init__.py docstring:
"""
Note on catalog modules:
- `catalog`: Vetting checks (V06-V07) - use `vet_catalog()`, `nearby_eb_search()`
- `catalogs`: Low-level catalog clients - use `GaiaClient`, `SimbadClient`
"""
```

**Pros:**
- No breaking changes
- Zero migration effort

**Cons:**
- Confusion persists for new users
- Documentation often overlooked

---

## 6. Recommendation

**Short-term (immediate):** Option D - Document the distinction in module docstrings and add a note to the main `api/__init__.py` docstring.

**Medium-term (next major version):** Option A - Rename `catalog.py` to `catalog_checks.py` with a deprecation period.

**Rationale:**
1. The vetting checks in `catalog.py` are the user-facing API; they should have the clearer name
2. `catalogs.py` is a platform-level re-export; power users who need it will find it
3. Renaming to `catalog_checks.py` aligns with the internal module name `validation/checks_catalog.py`
4. The `*_checks` suffix matches the semantic pattern (like how `lc_only.py` could be `lc_checks.py`)

---

## 7. Other Naming Observations

No other singular/plural collisions exist in `api/`. However, some inconsistencies worth noting:

| Pattern | Examples |
|---------|----------|
| `*_primitives.py` | `transit_primitives.py`, `timing_primitives.py`, `recovery_primitives.py`, `sandbox_primitives.py`, `vetting_primitives.py`, `transit_fit_primitives.py` |
| Plural nouns | `references.py`, `types.py`, `tolerances.py`, `caps.py` |
| Singular nouns | `activity.py`, `recovery.py`, `report.py`, `target.py` |

The `*_primitives.py` pattern is internally consistent. The singular/plural noun usage for domain modules is acceptable and doesn't create ambiguity.

---

## Appendix: File Locations

```
src/bittr_tess_vetter/api/
├── catalog.py      # 341 lines - V06-V07 vetting checks
├── catalogs.py     # 143 lines - platform client re-exports
└── ...
```
