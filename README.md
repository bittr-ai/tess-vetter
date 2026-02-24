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

# tess-vetter

[![PyPI](https://img.shields.io/pypi/v/tess-vetter.svg)](https://pypi.org/project/tess-vetter/)
[![DOI](https://zenodo.org/badge/1164740674.svg)](https://doi.org/10.5281/zenodo.18743916)

Domain-first Python library for TESS transit detection + vetting (array-in/array-out).

This package follows a "domain-first" design: array-in/array-out astronomy algorithms with optional platform helpers. The core `api/`, `compute/`, and `validation/` modules are pure functions; the `platform/` module provides opt-in I/O and network clients when needed.

**Package structure:**

- **Pure domain logic** (no I/O, no network): `api/`, `compute/`, `validation/`, `transit/`, `recovery/`, `activity/`
- **Opt-in infrastructure** (network clients, caching, disk I/O): `platform/`

The `platform/` module is entirely optional and only used when explicitly imported.

## What's in here

- **Vetting pipeline**: default preset runs 15 checks (V01-V12, V13, V15, V11b); opt-in extended preset adds metrics-only diagnostics (V16-V21), for 21 total checks available via profiles (`tess_vetter.api.vet`)
- **Transit detection**: TLS/LS periodograms, multi-planet search, candidate merging (`tess_vetter.api.periodogram`)
- **Pixel diagnostics**: centroid shift, difference images, WCS-aware localization, aperture dependence (`tess_vetter.api.pixel`)
- **Transit recovery**: detrend + stack + trapezoid fitting for active stars (`tess_vetter.api.recovery`)
- **FPP (optional)**: TRICERATOPS+ support with a vendored copy under `src/tess_vetter/ext/` (`tess_vetter.api.fpp`)
- **Code mode (experimental)**: operation-catalog and MCP tooling under `tess_vetter.code_mode`; this surface is experimental and may change
- **Citations**: many public API entry points carry machine-readable literature references (see `REFERENCES.md` and `tess_vetter.api.references`)

## Installation

Requires Python 3.11–3.12.

### Minimal install (basic vetting)
```bash
pip install tess-vetter
```

CLI entrypoints (equivalent):
- `btv ...`
- `tess-vetter ...`
- `python -m tess_vetter ...`

Preflight your environment before demos/runs:

```bash
btv doctor --profile vet
```

### With transit detection (TLS)
```bash
pip install 'tess-vetter[tls]'
```

### With MCMC fitting
```bash
pip install 'tess-vetter[fit]'
```

### With physical transit model fitting (batman)
```bash
pip install 'tess-vetter[batman]'
```

### With advanced detrending (wotan)
```bash
pip install 'tess-vetter[wotan]'
```

### With limb darkening coefficients (ldtk)
```bash
pip install 'tess-vetter[ldtk]'
```

### With external vetter integration (exovetter)
```bash
pip install 'tess-vetter[exovetter]'
```

### With false positive probability (TRICERATOPS)
```bash
pip install 'tess-vetter[triceratops]'
```

### With Apple Silicon acceleration (MLX, macOS only)
```bash
pip install 'tess-vetter[mlx]'
```

### Full install
```bash
pip install 'tess-vetter[all]'
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
import tess_vetter.api as btv
```

Agent-first walkthroughs and CLI-first examples are in `docs/quickstart.rst`.

### Single candidate vetting

```python
import numpy as np
from tess_vetter.api import Candidate, Ephemeris, LightCurve, vet_candidate

lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
candidate = Candidate(
    ephemeris=Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5),
    depth_ppm=500,
)

bundle = vet_candidate(lc, candidate, network=False)

# Results are structured: status, metrics, flags, citations
for r in bundle.results:
    print(f"{r.id} ({r.name}): {r.status}")
    print(f"  metrics: {r.metrics}")
    print(f"  flags: {r.flags}")
    print(f"  citations: {r.citations}")
```

### Batch vetting (multiple candidates, one light curve)

```python
from tess_vetter.api import vet_many

candidates = [
    Candidate(ephemeris=Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)),
    Candidate(ephemeris=Ephemeris(period_days=7.0, t0_btjd=1852.0, duration_hours=3.0)),
]

bundles, summary = vet_many(lc, candidates, network=False)

# summary is a list of dicts with stable columns for sorting/filtering
for row in summary:
    print(f"P={row['period_days']:.2f}d: {row['n_ok']} ok, {row['n_skipped']} skipped")
```

### Pipeline introspection

```python
from tess_vetter.api import list_checks, describe_checks

# List all registered checks with their requirements
checks = list_checks()
for c in checks:
    req = c["requirements"]
    print(f"{c['id']}: {c['name']} (tier={c['tier']}, needs_network={req['needs_network']})")

# Human-readable summary
print(describe_checks())
```

### Running a subset of checks

```python
from tess_vetter.api import vet_candidate

# Run only LC-only checks (V01-V05)
bundle = vet_candidate(lc, candidate, checks=["V01", "V02", "V03", "V04", "V05"])
```

## Network behavior

Catalog-backed checks are always opt-in. You must pass `network=True` (and provide metadata like RA/Dec and TIC ID) to enable external queries; otherwise those checks return skipped results.

## CLI FPP workflow (agent-safe)

For repeatable FPP runs, use a two-step flow:

1. `btv fpp-prepare` to stage light curves/runtime artifacts and emit a manifest.
2. `btv fpp --prepare-manifest ...` (or `btv fpp-run --prepare-manifest ...`) to compute from the staged manifest.

Example:

```bash
btv fpp-prepare --toi "TOI-5807.01" --network-ok --cache-dir outputs/cache -o outputs/fpp/toi_5807.prepare.json
btv fpp --prepare-manifest outputs/fpp/toi_5807.prepare.json --require-prepared --preset fast --no-network -o outputs/fpp/toi_5807.fast.json
```

Notes:
- `--prepare-manifest` on `btv fpp` uses the same prepared-manifest compute path as `btv fpp-run`.
- `--require-prepared` fails fast if staged artifacts are missing.
- In prepared mode, do not mix direct candidate/staging flags (for example `--tic-id`, `--period-days`, `--cache-dir`) with `--prepare-manifest`.

## Citations

Many public API entry points and vetting checks include a list of literature references in their results. The full reference list lives in `REFERENCES.md` (and is also available programmatically via `tess_vetter.api.references`).

## Code map

- `src/tess_vetter/api/`: stable host-facing facade (recommended import surface)
- `src/tess_vetter/compute/`: core array algorithms (detrending, periodograms, scoring)
- `src/tess_vetter/validation/`: check implementations and aggregation primitives
- `src/tess_vetter/pixel/`: pixel-level and WCS-aware diagnostics
- `src/tess_vetter/recovery/`, `src/tess_vetter/transit/`, `src/tess_vetter/activity/`: domain modules
- `src/tess_vetter/platform/io/`, `src/tess_vetter/platform/catalogs/`: optional I/O and catalog helpers
- `src/tess_vetter/ext/triceratops_plus_vendor/`: vendored TRICERATOPS+ (see `THIRD_PARTY_NOTICES.md`)

## Platform Support

- **macOS / Linux**: First-class support. All features work as expected.
- **Windows**: Best-effort support. Some platform-specific features may have limitations:
  - Cache file locking uses `fcntl` (Unix-only); graceful fallback on Windows.
  - Network timeouts use `SIGALRM` which may not work on all platforms.

## Security

Cache files (light curves, TRICERATOPS results) use pickle serialization for performance. Ensure cache directories have appropriate permissions in shared or multi-user environments.

## Development

Run development commands in the project environment with `uv run`:

```bash
uv run pytest
uv run ruff check .
uv run mypy src/tess_vetter
```

## Docs

User-facing docs are built with Sphinx from `docs/`:

```bash
uv run sphinx-build -b html -W docs docs/_build/html
```

For agent onboarding, start with `docs/quickstart.rst`. For API usage recipes and signatures, see `docs/api.rst`.

Internal working notes live in `working_docs/` and are not part of the stable API.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
