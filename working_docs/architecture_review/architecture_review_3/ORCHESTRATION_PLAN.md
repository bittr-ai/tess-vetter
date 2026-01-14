# Tranche 3: Architecture Hygiene Fixes

**Goal:** Resolve all P0 blockers from architecture_review_3 before public release.

---

## Scope

| ID | Issue | Effort |
|----|-------|--------|
| P0.1 | Sphinx build artifacts committed (517 files) | 10 min |
| P0.2 | Domain boundary leaky → adopt "honest boundary" docs | 30 min |
| P0.3 | `fcntl` import breaks Windows | 15 min |
| P0.4 | README references non-existent `[dev]` extra | 10 min |
| P0.5 | Version misalignment (0.0.1 vs v0.1.0 tag) | 15 min |

**Total estimated work:** ~1.5 hours — fits in a single Opus agent.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              TRANCHE 3 (SINGLE AGENT)                           │
│                                                                 │
│  Agent 1: P0 HYGIENE FIXES                                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Git cleanup                                                 │
│     - Add docs/_build/ and docs/_autosummary/ to .gitignore     │
│     - Remove tracked build artifacts from git index             │
│     - Verify clone size reduction                               │
│                                                                 │
│  2. Windows compatibility                                       │
│     - Add try/except guard for fcntl import in cache.py         │
│     - Implement fallback (no-op lock or msvcrt on Windows)      │
│     - Add import smoke test for Windows path                    │
│                                                                 │
│  3. Documentation alignment                                     │
│     - Fix README install instructions ([dev] → dev group)       │
│     - Update README to reflect "domain-first" (not domain-only) │
│     - Clarify which modules have I/O side effects               │
│                                                                 │
│  4. Version alignment                                           │
│     - Bump version to 0.1.0 in pyproject.toml                   │
│     - Bump version to 0.1.0 in __init__.py                      │
│     - Update CITATION.cff version field                         │
│                                                                 │
│  5. Validation                                                  │
│     - Run ruff check                                            │
│     - Run mypy                                                  │
│     - Run pytest                                                │
│     - Verify git status is clean                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Task Specifications

### Task 1: Git Cleanup (P0.1)

```bash
# Add to .gitignore
docs/_build/
docs/_autosummary/

# Remove from git index (keeps local files)
git rm -r --cached docs/_build/
git rm -r --cached docs/_autosummary/  # if tracked

# Verify
git ls-files docs/_build | wc -l  # should be 0
```

### Task 2: Windows Compatibility (P0.3)

Edit `src/bittr_tess_vetter/platform/io/cache.py`:

```python
# Before (line 6)
import fcntl

# After
import sys

if sys.platform != "win32":
    import fcntl
else:
    fcntl = None  # type: ignore[assignment]
```

Update lock functions to handle `fcntl is None`:

```python
def _acquire_lock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)

def _release_lock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
```

### Task 3: Documentation Alignment (P0.2, P0.4)

**README.md fixes:**
- Replace `pip install -e ".[dev]"` with `uv sync --group dev` or document the actual extras
- Update architecture description from "domain-only" to "domain-first"
- Add note that `platform/` contains I/O helpers, and some validation checks use network

**Boundary documentation (Option B - honest boundary):**
- Keep code where it is (no refactoring)
- Update docs to accurately describe what does/doesn't do I/O
- Document `network=False` flag for offline operation

### Task 4: Version Alignment (P0.5)

Files to update:
1. `pyproject.toml`: `version = "0.1.0"`
2. `src/bittr_tess_vetter/__init__.py`: `__version__ = "0.1.0"`
3. `CITATION.cff`: `version: 0.1.0`

### Task 5: Validation

```bash
uv run ruff check .
uv run mypy src/bittr_tess_vetter
uv run pytest
git status  # should show only intended changes
```

---

## Success Criteria

```bash
# No build artifacts tracked
git ls-files docs/_build | wc -l  # 0

# Windows import works (simulate by checking guard exists)
grep -q "sys.platform" src/bittr_tess_vetter/platform/io/cache.py

# README has correct install instructions
! grep -q '\[dev\]' README.md

# Version aligned
grep -q 'version = "0.1.0"' pyproject.toml
grep -q '__version__ = "0.1.0"' src/bittr_tess_vetter/__init__.py

# All checks pass
uv run ruff check . && uv run mypy src/bittr_tess_vetter && uv run pytest
```

---

## Deployment Command

```
Deploy single Opus agent with full P0 scope.
Agent has 200k context - can handle all tasks sequentially with validation.
```

---

## Post-Agent Checklist

- [ ] Review git diff before committing
- [ ] Commit with message: `fix: resolve P0 architecture hygiene issues`
- [ ] Push and verify CI passes
- [ ] Tag `v0.1.0` after CI green
