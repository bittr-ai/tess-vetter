# Plotting Spec Feedback (Consolidated)

**Date:** 2026-01-20  
**Scope reviewed:**  
- `working_docs/image_support/plotting_spec.md`  
- `working_docs/image_support/consolidated_plotting_implementation_plan.md`  
- `working_docs/image_support/research_*.md` (Q1–Q5, Kepler DVR, per-check plot recs, API integration)  
- Current repo structure under `src/bittr_tess_vetter/` (notably `validation/` and `api/`)

---

## TL;DR (highest-impact tweaks before implementation)

1. **Prevent global matplotlib side effects**: prefer `plt.rc_context(STYLES[style])` (per-call) over mutating `plt.rcParams` globally.
2. **Standardize `plot_data` ownership + naming**: this repo already uses `CheckResult.raw` for unstructured data; keep all plot payloads in `raw["plot_data"]` and use consistent key suffixes like `_ppm`, `_btjd`, `_hours`.
3. **Specify image coordinate conventions**: lock down `(row, col)` vs `(x, y)`, plus `imshow(origin=...)` so centroid/diff-image overlays don’t flip.
4. **Design explicitly for multi-sector pixel plots**: V08/V09/V10/V20 should accept *either* a single `CheckResult` or a per-sector list (or have separate `*_multi_sector` helpers) with consistent subplot-grid rules.
5. **Make tests robust across matplotlib versions**: favor “structure tests” (labels/artists/no warnings) for most plots; use `pytest-mpl` image baselines sparingly.

---

## Repo-Truth Mismatches to Fix in Docs/Spec

### 1) Validation module paths in the spec don’t match this repo

The spec references files like `validation/checks_lc_only.py` and `validation/checks_false_alarm.py`. In this repo, the relevant modules are closer to:
- LC checks: `src/bittr_tess_vetter/validation/lc_checks.py`
- False alarm checks: `src/bittr_tess_vetter/validation/lc_false_alarm_checks.py`
- Pixel checks: `src/bittr_tess_vetter/validation/checks_pixel.py`
- Catalog checks: `src/bittr_tess_vetter/validation/checks_catalog.py`
- Ghost features: `src/bittr_tess_vetter/validation/ghost_features.py`
- Sector consistency: `src/bittr_tess_vetter/validation/sector_consistency.py`

**Recommendation:** Update the spec’s file references so implementation work maps 1:1 to the actual codebase.

### 2) Some research notes use `details["plot_data"]`, but canonical storage is `raw["plot_data"]`

`CheckResult` is defined in `src/bittr_tess_vetter/validation/result_schema.py` and already supports `raw: dict[str, Any] | None`.

There is a backward-compatible `details` property, but it is a *derived view* that merges `metrics/flags/notes/raw`. The plotting spec’s “always include `raw["plot_data"]`” choice aligns better with the actual schema.

**Recommendation:** Treat `raw["plot_data"]` as canonical; if docs mention `details["plot_data"]`, rewrite to `raw["plot_data"]`.

### 3) `get_sector_color()` example returns RGBA, not a hex string

The spec types it as `-> str` but returns `plt.cm.tab10(idx % 10)`, which is an RGBA tuple-like.

**Recommendation:** Either (a) change return type to `tuple[float, float, float, float]` / “matplotlib color”, or (b) convert to hex via `matplotlib.colors.to_hex(...)`.

---

## API / UX Feedback

### Keep plotting imports stable (avoid conditional API “shape shifts” where possible)

Re-exporting plotting functions from `bittr_tess_vetter.api` is convenient, but it makes `from bittr_tess_vetter.api import plot_odd_even` succeed/fail depending on whether matplotlib is installed.

This is fine if intentional, but decide explicitly:
- **Option A (recommended):** Stable core API; keep plotting under `bittr_tess_vetter.plotting` only. Users opt into it.
- **Option B:** Re-export from `api` behind the guard (current spec direction) and accept the conditional surface.

If you keep Option B, ensure error messaging is consistent with the project’s other optional features (e.g., MLX).

### Be strict about JSON-serializability of `raw["plot_data"]`

`raw` can hold anything at runtime, but plot payloads will often be persisted/exported. Nested python lists of python scalars are safest.

**Recommendation:** In check implementations, explicitly convert:
- `np.float32/np.float64 -> float`
- `np.int* -> int`
- `np.ndarray -> .astype(np.float32).tolist()` (for 2D stamps)

### Add a small schema/version marker

Even if you don’t anticipate changes, a `plot_data_version: int` (or `plot_data={"version": 1, ...}`) prevents future “mystery KeyError” situations when formats evolve.

---

## Plot Implementation Details Worth Locking Down Early

### 1) Styles: prefer context-managed styles to avoid global mutations

The spec’s `_core.apply_style()` updates `plt.rcParams`. That’s convenient but sticky in notebooks/tests.

**Recommendation:** implement a helper like:
- `with plt.rc_context(STYLES[style]): ...` inside each `plot_*`
or provide `style_context(style)` as a context manager.

### 2) Image plots: define axis/coordinate conventions explicitly

For V08/V09/V20, pick and document:
- `reference_image[row][col]` storage
- whether centroid coordinates are `(x=col, y=row)` or `(x, y)` in pixel space
- `imshow(origin="lower")` vs `"upper"`
- how you mark `target_pixel` (is it `[col,row]` or `[row,col]`?)

Without this, overlays will be wrong in subtle ways and tests won’t catch it unless you have image-baselines.

### 3) Multi-sector pixel results: support lists + subplot grids

Your research (Q5) strongly supports per-sector display for pixel checks. The plotting API should make the “right” thing easy:
- accept `result: CheckResult | list[CheckResult]` for V08/V09/V10/V20, or
- provide `plot_*_multi_sector(results, ...)` wrappers that call single-sector implementations.

If you go the “accept list” route, standardize:
- sorting by sector
- panel titles like `Sector 55`
- shared colorbar strategy (per-panel vs shared)

### 4) Colorbars: design for multi-panel layouts

The spec’s `add_colorbar()` helper is good (minor ticks off). For multi-panel figures, though, `fig.colorbar(..., ax=ax)` will resize axes; you’ll eventually want `cax` or `cbar_kwargs={"cax": ...}` support.

**Recommendation:** keep `show_colorbar=True` default for images, but expose `cbar_kwargs` (including `cax`) and return `(ax, cbar|None)` consistently for all image plots.

---

## Data Contract Feedback (per-check plot_data)

### Standardize key names now

There are already minor naming differences across docs (e.g., `odd_depths` vs `odd_depths_ppm` in different places).

**Recommendation:** pick one convention and enforce it everywhere:
- floats include units in the name: `_ppm`, `_hours`, `_btjd`, `_arcsec`, `_pixels`
- indices are explicit: `_idx`, `_indices`
- coordinates are explicit and ordered: either always `(row, col)` or always `(x, y)`; avoid mixing.

### Cap sizes explicitly (especially for “per-epoch” and “phase arrays”)

Q1 notes total size is small, but it’s still worth hard caps to prevent surprises:
- epoch arrays: cap at 20–50
- modshift arrays: document expected bin count
- images: document maximum stamp size (e.g., 15×15 or 21×21)

This also helps keep unit tests and serialized bundles stable.

---

## Testing Feedback

### Prefer robust unit tests for most plots

Strongly recommended test approach for each plot:
- `pytest.importorskip("matplotlib")`
- use `matplotlib.use("Agg")` for tests
- verify:
  - function returns expected object (`Axes` / `(Axes, cbar)` / `Figure`)
  - expected labels/titles exist
  - no warnings raised (optionally)
  - handles missing plot_data with a clear `ValueError`

### Use `pytest-mpl` image baselines selectively

Image regression is valuable for a few “high-risk-to-get-wrong” plots:
- V08 centroid overlay
- V09 difference image overlay
- DVR summary layout (panel placement)

For the rest, “structure tests” are usually enough and far less brittle across matplotlib/font/platform variations.

---

## Suggested small spec edits (low effort, high payoff)

- Add a “**Coordinate conventions**” subsection under Data Contracts for all pixel/image plot_data.
- Add a “**Multi-sector handling**” subsection per plot category (LC-only vs pixel-level).
- Replace any `details["plot_data"]` references with `raw["plot_data"]`.
- Fix `get_sector_color()` return type documentation.
- Consider bumping the plotting extra to a modern matplotlib baseline (this project supports Python 3.11–3.12).

