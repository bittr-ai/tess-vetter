# Changelog

This project follows semantic versioning (SemVer).

## 0.2.2 (2026-01-20)

- Add static tutorial companion `docs/tutorials/10-toi-5807-check-by-check.md` so plots render in any IDE without running a notebook.
- Add additional pre-rendered tutorial artifacts for V03/V06/V07/V09/V10 under `docs/tutorials/artifacts/10-toi-5807-check-by-check/`.
- Plot polish: more conservative ExoFOP card note wrapping; sensitivity-sweep y-label compaction.

## 0.2.1 (2026-01-20)

- Add optional plotting extra: `pip install 'tess-vetter[plotting]'`.
- Add `tess_vetter.plotting` module with matplotlib-guarded exports and DVR-style summary plotting.
- Add per-check plot functions for V01-V21 plus transit/lightcurve visualization helpers.
- Add plot verification script `scripts/verify_plots.py` (writes to `working_docs/image_support/verification/verification_plots/` by default).

## 0.2.0 (2026-01-16)

- Add V11b `modshift_uniqueness` check: independent ModShift implementation with properly-scaled Fred (~1-10 for TESS, not ~60-96 from exovetter). Includes MS1-MS6 normalized uniqueness metrics, CHASES local uniqueness, and CHI transit depth consistency.
- New API function `modshift_uniqueness()` in `tess_vetter.api.exovetter`.
- Add LC-only false-alarm checks: V13 `data_gaps` and V15 `transit_asymmetry`.
- Extend V04 `depth_stability` metrics with single-event domination and DMM (mean-vs-median depth) diagnostics.
- Check registry now includes 15 checks (V01-V12, V13, V15, plus V11b).

## 0.1.0 (2026-01-14)

- Initial public release.
- Golden-path API for transit detection and vetting under `tess_vetter.api`.
- Extensible vetting pipeline (`VettingPipeline`) with structured results.
