# Plotting Spec Revision Summary (v1.1)

**Date:** 2026-01-20
**Reviewer:** Staff Architect
**Spec Updated:** `plotting_spec.md` v1.0.0 -> v1.1.0

---

## Key Architectural Decisions

### 1. Style Management: Context Managers Over Global Mutation

**Decision:** Added `style_context()` as the preferred approach; deprecated global `apply_style()`.

**Rationale:** Global `plt.rcParams` mutations are sticky in notebooks and cause test pollution. Using `plt.rc_context()` ensures styles are scoped to individual plot calls without side effects.

**Impact:** All plot function implementations should use `with style_context(style):` internally.

---

### 2. Coordinate Conventions for Pixel Data

**Decision:** Locked down explicit conventions:
- Image storage: `[row][col]` (numpy/FITS standard)
- Coordinate mapping: `(x, y)` = `(column, row)`
- imshow: Always `origin="lower"`
- Centroid keys: `centroid_x` (column), `centroid_y` (row)

**Rationale:** Without explicit conventions, overlay positions will be wrong in subtle ways that structure tests cannot catch. This matches TESS TPF conventions.

**Impact:** Added Section 4.5 to spec; all pixel plot implementations must follow these conventions.

---

### 3. Multi-Sector Pixel Plots: Explicit Grid Variants

**Decision:** Provide `plot_*_grid()` functions for multi-sector layouts rather than overloading single-result functions.

**Rationale:**
- Keeps the common single-result API simple
- Makes multi-sector intent explicit in code
- Avoids complex type handling (`CheckResult | list[CheckResult]`)

**Impact:** Added Section 5.9; Phase 3 implementation adds `plot_centroid_shift_grid()`, `plot_difference_image_grid()`, etc.

---

### 4. Testing Strategy: Structure-First Approach

**Decision:** Structure tests are primary; image baselines reserved for high-risk plots only (V08, V09, DVR summary).

**Rationale:** Image regression tests are brittle across matplotlib versions, fonts, and platforms. Structure tests (labels exist, returns correct type, no warnings, errors on invalid input) provide sufficient coverage for most plots with much lower maintenance burden.

**Impact:** Updated Section 10.1-10.3; reduced baseline image count significantly.

---

### 5. Data Contract Hardening

**Decisions:**
- Added `version: 1` to all plot_data schemas
- Required explicit numpy -> Python type conversion for JSON safety
- Standardized key naming with unit suffixes (`_ppm`, `_btjd`, `_hours`)
- Defined array size caps (50 epochs, 200 bins, 21x21 stamps)

**Rationale:** Prevents future KeyError surprises, ensures serialization works, and makes units explicit for both human readers and downstream consumers.

**Impact:** Added Sections 4.2-4.6; all check implementations must follow conversion patterns.

---

### 6. Module Path Corrections

**Decision:** Updated all module references to match actual codebase structure.

**Changes:**
- `checks_lc_only.py` -> `lc_checks.py`
- `checks_false_alarm.py` -> `lc_false_alarm_checks.py`
- `checks_exovetter.py` -> `exovetter_checks.py`

**Impact:** Implementation work now maps 1:1 to actual files.

---

### 7. API Export Strategy

**Decision:** Rejected the suggestion to keep plotting functions out of `api` module.

**Rationale:** Conditional re-export from `api` is consistent with how other optional features (MLX, exovetter) are handled. The error messaging is already clear. Power users who prefer explicit imports can use `from bittr_tess_vetter.plotting import ...` directly.

---

## Minor Technical Fixes

1. **get_sector_color()**: Now returns hex string via `mcolors.to_hex()` instead of RGBA tuple
2. **add_colorbar()**: Documented `cax` support via `**kwargs` for multi-panel layouts
3. **V01 epoch cap**: Changed from 20 to 50 to cover typical multi-sector targets

---

## Next Steps

1. Implementers should prioritize the coordinate convention section (4.5) - getting this wrong causes hard-to-debug overlay issues
2. All plot functions need the `with style_context(style):` wrapper pattern
3. Phase 3 (pixel plots) now includes `*_grid` variants in deliverables
4. Test scaffolding should set up `matplotlib.use("Agg")` in conftest.py globally

---

## Files Modified

- `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/image_support/plotting_spec.md` (v1.0.0 -> v1.1.0)
