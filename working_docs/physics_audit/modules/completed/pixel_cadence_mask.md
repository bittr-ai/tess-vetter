# Module Review: `pixel/cadence_mask.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

`default_cadence_mask()` is the shared cadence-hygiene gate for pixel-level algorithms (localization, difference images, centroid shifts, aperture checks). Any mistake here propagates broadly across V08–V10 style evidence.

## Function

- Name: `default_cadence_mask`
- Location: `src/bittr_tess_vetter/pixel/cadence_mask.py`
- Public API? no (internal utility; consumed by pixel modules)
- Called early by agents? yes (pixel stack is typically invoked early in vetting for host localization)

## Inputs / outputs

- Units + conventions:
  - `time`: cadence timestamps (days; typically BTJD from TPF FITS `TIME`)
  - `flux`: pixel cube/array with cadence as axis 0 (expected shapes `(n_cadences, n_rows, n_cols)` or `(n_cadences, n_pixels)`).
  - `quality`: cadence quality flags (SPOC `QUALITY` column; integer; “good” is typically `0`).
- Valid ranges:
  - No explicit validation. Assumes `time.shape[0] == flux.shape[0] == quality.shape[0]`.
- Output semantics:
  - Returns boolean mask selecting cadences that pass:
    - `quality == 0`
    - `time` is finite
    - optionally (default) “has at least one finite pixel value” across the stamp

## Physics correctness

- Formula/source: not physics; data hygiene gate.
- Assumptions:
  - `quality == 0` is the desired “good cadence” predicate (conservative; excludes any cadence with any flagged bit).
  - Pixel flux NaNs are allowed as long as at least one finite pixel exists (when `require_finite_pixels=True`).
  - Cadence axis is always axis 0 in `flux`.
- Known failure regimes:
  - If upstream passes a `quality` array where “good” is not encoded as literal `0` (or uses non-SPOC semantics), this will over-mask.
  - If upstream passes `flux` with cadence not on axis 0, the `reshape(flux.shape[0], -1)` heuristic will be wrong.
  - Does not enforce monotonic time or minimum cadence count; downstream functions must handle small/empty masks.

## Statistics / uncertainty

Not applicable (masking only).

## Numerical stability

- NaN handling:
  - Explicitly excludes non-finite `time`.
  - Optionally requires at least one finite pixel per cadence; does not require *all* pixels finite.
- Empty-mask handling:
  - Can return an all-False mask; downstream callers should guard against `n=0` (several do).
- Sensitivity to cadence/gaps:
  - This mask does not detect cadence gaps; it only excludes flagged/non-finite samples.

## Tests

- Existing tests covering this:
  - No direct unit tests found for `default_cadence_mask` (it is exercised indirectly by pixel tests).
- New tests to add:
  - A small unit test that verifies:
    - `quality != 0` cadences are excluded
    - non-finite `time` is excluded
    - `require_finite_pixels=True` excludes all-NaN frames
    - `require_finite_pixels=False` keeps frames even if all-NaN

## Fixes / changes (if any)

- Proposed fix: none required for current semantics.
- Notes / optional refinement:
  - Consider allowing a configurable “quality bitmask” (e.g., `(quality & mask) == 0`) if callers need less-conservative behavior.
  - Consider (optionally) requiring finite `flux` for the target pixel (or aperture) rather than “any finite pixel”, depending on the algorithm.

