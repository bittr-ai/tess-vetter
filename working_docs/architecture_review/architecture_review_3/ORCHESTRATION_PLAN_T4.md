# Tranche 4: P1 Architecture Improvements + P2 Quick Wins

**Goal:** Address medium-impact architecture issues and quick quality-of-life improvements.

---

## Scope

### From architecture_review_3 (P1):

| ID | Issue | Effort |
|----|-------|--------|
| P1.1 | Base dependencies heavy (numba, emcee, arviz) | 2-3 hrs |
| P1.2 | Runtime setuptools dependency is unusual | 15 min |
| P1.5 | ReadTheDocs config redundant/conflicting | 15 min |

### From CONSOLIDATED_PRIORITIES (P2/P3):

| ID | Issue | Effort |
|----|-------|--------|
| P2.8 | Add GPL notice for triceratops extras | 15 min |
| P2.9 | Add warning when checks are skipped | 1 hr |
| P3.7 | Add stability tier documentation | 1 hr |
| P3.11 | Use platformdirs for cache path default | 30 min |

**Total:** ~6 hours — split into 2 agents for parallelism.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANCHE 4 (2 PARALLEL AGENTS)                        │
├───────────────────────────────────┬─────────────────────────────────────┤
│   Agent 1: DEPS & CONFIG          │   Agent 2: UX & DOCS                │
├───────────────────────────────────┼─────────────────────────────────────┤
│ P1.1  Restructure dependencies    │ P2.9  Skipped-check warnings        │
│       - Define minimal core       │ P3.7  Stability tier docs           │
│       - Create extras: tls, fit   │ P2.8  GPL notice for triceratops    │
│ P1.2  Remove setuptools runtime   │ P3.11 platformdirs cache default    │
│ P1.5  Fix RTD config duplication  │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│ ~3 hours                          │ ~3 hours                            │
└───────────────────────────────────┴─────────────────────────────────────┘
```

---

## Agent 1: Dependencies & Config

### Task 1.1: Restructure Dependencies (P1.1)

Current base dependencies include heavy packages that not all users need:
- `numba` (TLS dependency) — researchers doing simple vetting don't need detection
- `emcee`, `arviz` — MCMC fitting, not needed for basic checks

**New structure:**

```toml
[project]
dependencies = [
  # Minimal core - arrays, models, basic checks
  "numpy>=1.24.0,<2.4.0",
  "scipy>=1.10.0",
  "pydantic>=2.4.0",
  "astropy>=5.0.0",
  "requests>=2.32.4",
]

[project.optional-dependencies]
# Transit detection with TLS (pulls numba)
tls = [
  "transitleastsquares>=1.32",
  "numba>=0.63.0",
]
# MCMC transit fitting
fit = [
  "emcee>=3.1.6",
  "arviz>=0.23.0",
]
# Existing extras unchanged
wotan = ["wotan>=1.1"]
ldtk = ["ldtk>=1.8.5"]
triceratops = [...]
# Full install
all = ["bittr-tess-vetter[tls,fit,wotan,ldtk,triceratops]"]
```

**Code changes required:**
- Add lazy imports for `transitleastsquares`, `numba`, `emcee`, `arviz`
- Raise helpful `ImportError` when optional dep missing
- Update README with new install tiers

### Task 1.2: Remove setuptools Runtime Dep (P1.2)

Remove from `[project.dependencies]`:
```toml
"setuptools>=78.1.1",  # CVE-2025-47273 fix  <-- REMOVE
```

This is a build-time dependency, not runtime. The CVE fix belongs in CI/build images.

### Task 1.3: Fix RTD Config (P1.5)

Current `.readthedocs.yaml` has both:
- `build.jobs.post_install` with uv
- `python.install` with pip

Pick one. Recommend keeping uv approach and removing pip install block.

### Validation

```bash
# Verify minimal install works
uv pip install -e . --no-deps  # then add only core deps
python -c "from bittr_tess_vetter.api import Candidate, Ephemeris, CheckResult"

# Verify optional imports fail gracefully
python -c "from bittr_tess_vetter.api import run_tls"  # should suggest: pip install .[tls]

# Full test suite still passes
uv sync --all-extras && uv run pytest
```

---

## Agent 2: UX & Documentation

### Task 2.1: Skipped-Check Warnings (P2.9)

When `vet_candidate(network=False)` or metadata is missing, checks get skipped silently.

Add warnings to `VettingResult` or logging:
```python
import warnings

if network is False:
    warnings.warn(
        "Catalog checks (V07, V08) skipped because network=False. "
        "Set network=True to enable ExoFOP and Gaia cross-matching.",
        UserWarning,
        stacklevel=2
    )
```

### Task 2.2: Stability Tier Documentation (P3.7)

Add `docs/api-stability.rst` documenting:
- **Stable**: Core types (Candidate, Ephemeris, CheckResult, LightCurve), vet_candidate()
- **Provisional**: Pixel analysis, FPP calculation, periodogram wrappers
- **Internal**: Anything not in `__all__`

Add brief note to `docs/api.rst` linking to stability tiers.

### Task 2.3: GPL Notice for Triceratops (P2.8)

Add note to README and/or pyproject.toml:
```
Note: The `[triceratops]` extra includes `pytransit` which is GPL-2.0 licensed.
Installing this extra changes the effective license of your environment.
```

### Task 2.4: platformdirs Cache Default (P3.11)

Update `src/bittr_tess_vetter/platform/io/cache.py`:

```python
try:
    from platformdirs import user_cache_dir
    DEFAULT_CACHE = Path(user_cache_dir("bittr-tess-vetter"))
except ImportError:
    DEFAULT_CACHE = Path.cwd() / ".bittr-tess-vetter" / "cache"
```

Add `platformdirs` as optional or core dependency (it's tiny).

### Validation

```bash
# Check warnings appear
python -c "
from bittr_tess_vetter.api import vet_candidate, Candidate, Ephemeris
import numpy as np
c = Candidate(tic_id=123, ephemeris=Ephemeris(epoch=0, period=1, duration_hours=2))
# Should warn about network=False
"

# Docs build
cd docs && make html
```

---

## Success Criteria

```bash
# Minimal install possible
pip install bittr-tess-vetter  # No numba, emcee

# Optional deps raise helpful errors
python -c "from bittr_tess_vetter.api import run_tls" 2>&1 | grep -q "pip install"

# Full suite passes
uv sync --all-extras --group dev && uv run pytest && uv run ruff check . && uv run mypy src/

# RTD config is clean (single install method)
grep -c "pip install\|uv pip install" .readthedocs.yaml  # should be 1
```

---

## Deployment

Deploy both agents in parallel. They work on independent areas:
- Agent 1: pyproject.toml, cache.py imports, .readthedocs.yaml
- Agent 2: validation modules, docs/, README
