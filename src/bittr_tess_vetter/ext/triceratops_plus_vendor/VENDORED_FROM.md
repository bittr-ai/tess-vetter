# TRICERATOPS+ Vendor Provenance

- **Upstream:** https://github.com/JGB276/TRICERATOPS-plus
- **Commit SHA:** bc4c4491df30155160252171ddcd83a2b83511b7
- **Date vendored:** 2026-01-08
- **Files copied:** triceratops/*.py, triceratops/data/*

## License
MIT License (OSI Approved - as declared in setup.py)

## Changes from upstream
- Import paths rewritten from `triceratops.X` to
  `bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops.X`
- Data path resolution patched to use `__file__`-relative paths

## Reference
Giacalone, S., et al. 2021, AJ, 161, 24: "Vetting of 384 TESS Objects of
Interest with TRICERATOPS and Statistical Validation of 12 Planet Candidates"

Barrientos et al. 2025, arxiv:2508.02782: TRICERATOPS+ multi-color extension
