# Changelog

This project follows semantic versioning (SemVer).

## 0.2.0 (2026-01-16)

- Add V11b `modshift_uniqueness` check: independent ModShift implementation with properly-scaled Fred (~1-10 for TESS, not ~60-96 from exovetter). Includes MS1-MS6 normalized uniqueness metrics, CHASES local uniqueness, and CHI transit depth consistency.
- New API function `modshift_uniqueness()` in `bittr_tess_vetter.api.exovetter`.
- Add LC-only false-alarm checks: V13 `data_gaps` and V15 `transit_asymmetry`.
- Extend V04 `depth_stability` metrics with single-event domination and DMM (mean-vs-median depth) diagnostics.
- Check registry now includes 15 checks (V01-V12, V13, V15, plus V11b).

## 0.1.0 (2026-01-14)

- Initial public release.
- Golden-path API for transit detection and vetting under `bittr_tess_vetter.api`.
- Extensible vetting pipeline (`VettingPipeline`) with structured results.
