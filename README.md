```text
                                                                    ·                               
██    ·           ██            ██            ██                                            ██      
██▒░              ██▒░          ██▒░  ·       ██▒░                               ✧          ██▒░    
██▒░               ▒▒░          ██▒░          ██▒░        ████████          ██████           ▒▒░    
██▒░             ·          ·   ██▒░          ██▒░        ████████▒░        ██████▒░                
████████ ·        ██        ██████████    ██████████      ██▒▒▒▒██▒░         ▒▒▒▒▒██        ██      
████████▒░        ██▒░      ██████████▒░  ██████████▒░    ██▒░  ██▒░              ██▒░      ██▒░    
██      ██        ██▒░       ▒▒▒██▒▒▒▒▒░   ▒▒▒██▒▒▒▒▒░·   ██▒░   ▒▒░        ████████▒░      ██▒░    
██      ██▒░      ██▒░          ██▒░          ██▒░        ██▒░              ████████▒░      ██▒░    
██      ██▒░      ██▒░          ██▒░          ██▒░ ·      ██▒░            ██      ██▒░      ██▒░    
██      ██▒░      ██▒░          ██▒░          ██▒░        ██▒░            ██      ██▒░      ██▒░    
████████ ▒▒░      ██▒░          ████          ████        ██▒░      ██     ▒████████▒░      ██▒░    
████████▒░        ██▒░          ████▒░        ████▒░·     ██▒░      ██▒░    ████████▒░      ██▒░    
 ▒▒▒▒▒▒▒▒░         ▒▒░           ▒▒▒▒░         ▒▒▒▒░       ▒▒░       ▒▒░    ·▒▒▒▒▒▒▒▒░       ▒▒░    
                                                                                                    
     ·      ·                                                                                       
                                                                                                    
                                              bittr.ai                                              
```

# bittr-tess-vetter

Domain library for TESS transit detection + vetting (array-in/array-out).

This package follows a "domain-first" design: array-in/array-out astronomy algorithms with optional platform helpers. The core `api/`, `compute/`, and `validation/` modules are pure functions; the `platform/` module provides opt-in I/O and network clients when needed.

**Package structure:**

- **Pure domain logic** (no I/O, no network): `api/`, `compute/`, `validation/`, `transit/`, `recovery/`, `activity/`
- **Opt-in infrastructure** (network clients, caching, disk I/O): `platform/`

The `platform/` module is entirely optional and only used when explicitly imported.

## What’s in here

- Transit detection: TLS/LS periodograms, multi-planet search, candidate merging (`bittr_tess_vetter.api.periodogram`)
- Vetting pipeline: tiered checks (LC-only, optional catalog, optional pixel, optional Exovetter) (`bittr_tess_vetter.api.vet`)
- Pixel diagnostics: centroid shift, difference images, WCS-aware localization, aperture dependence (`bittr_tess_vetter.api.pixel`)
- Transit recovery: detrend + stack + trapezoid fitting for active stars (`bittr_tess_vetter.api.recovery`)
- FPP (optional): TRICERATOPS+ support with a vendored copy under `src/bittr_tess_vetter/ext/` (`bittr_tess_vetter.api.fpp`)
- References: many public API entry points carry machine-readable citations (`src/bittr_tess_vetter/api/REFERENCES.md`)

## Installation

Requires Python 3.11–3.12.

### Minimal install (basic vetting)
```bash
pip install bittr-tess-vetter
```

### With transit detection (TLS)
```bash
pip install 'bittr-tess-vetter[tls]'
```

### With MCMC fitting
```bash
pip install 'bittr-tess-vetter[fit]'
```

### Full install
```bash
pip install 'bittr-tess-vetter[all]'
```

### Development

Using `uv` (recommended for this repo; uses `uv.lock`):

```bash
uv sync --all-extras --group dev
```

Using `pip`:

```bash
python -m pip install -e ".[all]"
```

**License note for optional extras:** The `[triceratops]` extra includes `pytransit`, which is GPL-2.0 licensed. Installing this extra changes the effective license of your environment. The `[ldtk]` extra is also GPL-2.0. The core package remains BSD-3-Clause.


## Quickstart

The recommended import alias follows patterns from astropy (`import astropy.units as u`):

```python
import bittr_tess_vetter.api as btv
```

Full example:

```python
import numpy as np
import bittr_tess_vetter.api as btv
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
- `src/bittr_tess_vetter/platform/io/`, `src/bittr_tess_vetter/platform/catalogs/`: optional I/O and catalog helpers
- `src/bittr_tess_vetter/ext/triceratops_plus_vendor/`: vendored TRICERATOPS+ (see `THIRD_PARTY_NOTICES.md`)

## Platform Support

- **macOS / Linux**: First-class support. All features work as expected.
- **Windows**: Best-effort support. Some platform-specific features may have limitations:
  - Cache file locking uses `fcntl` (Unix-only); graceful fallback on Windows.
  - Network timeouts use `SIGALRM` which may not work on all platforms.

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
