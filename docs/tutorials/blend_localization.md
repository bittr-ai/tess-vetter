# Blend / Off-Target Diagnosis with Pixel Localization + Dilution Plausibility

This tutorial demonstrates a common TESS failure mode: a transit signal can look planet-like in the stitched light curve, but still be caused by a contaminating eclipsing binary inside the TESS aperture. Here we combine:

- WCS-aware difference-image localization (`tess_vetter.api.localize_transit_source`)
- Gaia neighbor hypotheses (`tess_vetter.api.catalogs.query_gaia_by_position_sync`)
- Dilution plausibility (“how deep would the eclipse have to be on that neighbor?”) (`tess_vetter.api.stellar_dilution.compute_dilution_scenarios`)
- An aperture-family sanity check (`tess_vetter.api.compute_aperture_family_depth_curve`)

**Example target**: `TIC 188646744` (TOI-5807.01). This is an extremely bright star where localization can be baseline-sensitive, so the “localization + physics” combination is the point of the example.

## Prerequisites

You need a way to acquire TPF FITS files. This tutorial uses `lightkurve` to download TESS Target Pixel Files from MAST:

```bash
pip install tess-vetter
pip install lightkurve astroquery
```

Network access is also needed for the Gaia query step. If you cannot use network, you can still run localization with just the target hypothesis (but you won’t get a neighbor table or dilution plausibility).

## Candidate ephemeris (example values)

Use the archive ephemeris (or your own refined fit):

- Period: `P = 14.2423724 d`
- Mid-transit: `T0 = 3540.263170 BTJD`
- Duration: `4.56 h`
- Observed depth: `~232 ppm`

## Run the full demo (script)

The runnable reference implementation is:

- `examples/blend_localization_tic188646744.py`

Example run (multi-sector localization, Gaia hypotheses, dilution plausibility, aperture-family check):

```bash
uv run python examples/blend_localization_tic188646744.py \
  --tic-id 188646744 \
  --sectors 55,75,82,83 \
  --period-days 14.2423724 \
  --t0-btjd 3540.263170 \
  --duration-hours 4.56 \
  --depth-ppm 232 \
  --author SPOC \
  --gaia-radius-arcsec 60
```

## How to interpret the outputs

### 1) Localization verdicts per sector

`localize_transit_source(...)` produces a difference image and estimates the sky position of the transit source. With multiple reference sources (target + Gaia neighbors), it provides a best-match hypothesis and a localization verdict (`ON_TARGET`, `OFF_TARGET`, or `AMBIGUOUS`).

For very bright/saturated targets, the “best” hypothesis can flip across sectors or window choices. Treat localization as *evidence*, not an oracle.

### 2) Dilution plausibility (“physics filter”)

For each potential host star, dilution tells you how much the observed depth must be corrected to get the true depth on that host:

- `true_depth_ppm = observed_depth_ppm / host_flux_fraction`

If a Gaia neighbor contributes only a tiny fraction of the aperture flux, the implied eclipse depth on that neighbor can exceed 100% (physically impossible), or be so large that an eclipsing stellar companion is required.

This is the fastest way to dismiss “pixel-fit winners” that are simply too faint to host the signal.

### 3) Aperture-family sanity check

`compute_aperture_family_depth_curve(...)` re-measures the depth vs. aperture radius. A contaminant that is far from the target often produces a depth curve that changes strongly with aperture size (because the contaminant flux fraction changes with aperture).

This is not definitive, but it’s a powerful cross-check when difference-image centroiding is unstable.

## Notes / caveats

- This tutorial is intentionally “TESS-realistic”: host attribution can remain ambiguous even after extensive pixel tests. In those cases, the correct next step is resolved ground-based photometry or high-resolution imaging during transit.
- If you want a deterministic first tutorial, start with an LC-only eclipsing binary example (V01 odd/even + V02 secondary) and then come back here.
