```text
                                                                    ·                                         
██    ·           ██            ██            ██                                                      ██      
██▒░              ██▒░          ██▒░  ·       ██▒░                               ✧                    ██▒░·   
██▒░               ▒▒░          ██▒░          ██▒░        ████████                    ██████           ▒▒░    
██▒░             ·          ·   ██▒░          ██▒░        ████████▒░          ✧       ██████▒░                
████████ ·        ██        ██████████    ██████████      ██▒▒▒▒██▒░               ·   ▒▒▒▒▒██        ██      
████████▒░        ██▒░      ██████████▒░  ██████████▒░    ██▒░  ██▒░                        ██▒░      ██▒░    
██      ██        ██▒░       ▒▒▒██▒▒▒▒▒░   ▒▒▒██▒▒▒▒▒░·   ██▒░   ▒▒░                  ████████▒░      ██▒░    
██      ██▒░      ██▒░          ██▒░          ██▒░        ██▒░                        ████████▒░      ██▒░    
██      ██▒░      ██▒░          ██▒░          ██▒░ ·      ██▒░                      ██      ██▒░      ██▒░    
██      ██▒░      ██▒░          ██▒░          ██▒░        ██▒░                      ██      ██▒░      ██▒░    
████████ ▒▒░      ██▒░          ████          ████        ██▒░          ████         ▒████████▒░      ██▒░    
████████▒░        ██▒░          ████▒░        ████▒░·     ██▒░       ·  ████▒░        ████████▒░      ██▒░    
 ▒▒▒▒▒▒▒▒░         ▒▒░           ▒▒▒▒░         ▒▒▒▒░       ▒▒░           ▒▒▒▒░         ▒▒▒▒▒▒▒▒░       ▒▒░    
                                                                                                              
     ·      ·                                                                                              ·  
                                                                                                              
                                                   bittr.ai                                                   
```

# bittr-tess-vetter

Domain library for TESS transit detection + vetting (array-in/array-out).

This package is intentionally “domain-only”: array-in/array-out astronomy algorithms without any platform-specific tooling (stores, manifests, agent frameworks, etc.).

## What’s in here

- Transit detection: TLS/LS periodograms, multi-planet search, candidate merging (`bittr_tess_vetter.api.periodogram`)
- Vetting pipeline: tiered checks (LC-only, optional catalog, optional pixel, optional Exovetter) (`bittr_tess_vetter.api.vet`)
- Pixel diagnostics: centroid shift, difference images, WCS-aware localization, aperture dependence (`bittr_tess_vetter.api.pixel`)
- Transit recovery: detrend + stack + trapezoid fitting for active stars (`bittr_tess_vetter.api.recovery`)
- FPP (optional): TRICERATOPS+ support with a vendored copy under `src/bittr_tess_vetter/ext/` (`bittr_tess_vetter.api.fpp`)
- References: many public API entry points carry machine-readable citations (`src/bittr_tess_vetter/api/REFERENCES.md`)

## Install

Requires Python 3.11–3.12.

Using `uv` (recommended for this repo; uses `uv.lock`):

```bash
uv sync
# Optional extras:
uv sync --all-extras
```

Using `pip`:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[all]"
```

Note: `uv` is configured to use a local editable dependency for `bittr-reason-core` at `../bittr-reason-core` (see `pyproject.toml`).

## Quickstart

```python
import numpy as np
from bittr_tess_vetter.api import Candidate, Ephemeris, LightCurve, run_periodogram, vet_candidate

lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
candidate = Candidate(
    ephemeris=Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5),
    depth_ppm=500,
)

pg = run_periodogram(time=np.asarray(time), flux=np.asarray(flux), flux_err=np.asarray(flux_err))
bundle = vet_candidate(lc, candidate, policy_mode="metrics_only", network=False)

print(pg.best_period_days)
for r in bundle.results:
    print(r.id, r.name, r.passed, r.confidence)
```

## Network behavior

Catalog-backed checks are always opt-in. You must pass `network=True` (and provide metadata like RA/Dec and TIC ID) to enable external queries; otherwise those checks return skipped results.

## Code map

- `src/bittr_tess_vetter/api/`: stable host-facing facade (recommended import surface)
- `src/bittr_tess_vetter/compute/`: core array algorithms (detrending, periodograms, scoring)
- `src/bittr_tess_vetter/validation/`: check implementations and aggregation primitives
- `src/bittr_tess_vetter/pixel/`: pixel-level and WCS-aware diagnostics
- `src/bittr_tess_vetter/recovery/`, `src/bittr_tess_vetter/transit/`, `src/bittr_tess_vetter/activity/`: domain modules
- `src/bittr_tess_vetter/io/`, `src/bittr_tess_vetter/catalogs/`: optional I/O and catalog helpers
- `src/bittr_tess_vetter/ext/triceratops_plus_vendor/`: vendored TRICERATOPS+ (see `THIRD_PARTY_NOTICES.md`)

## Development

```bash
uv run pytest
uv run ruff check .
# Optional (type-checking): install the `dev` extra, then:
# uv sync --extra dev
# uv run mypy src
```

## Docs

Internal working notes live in `working_docs/` (for example `working_docs/api/v1_spec.md`).

## License

MIT.
