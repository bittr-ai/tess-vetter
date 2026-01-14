# CI/CD Configuration Research Report

**Date:** 2026-01-14
**Package:** bittr-tess-vetter
**Status:** No CI/CD currently configured

---

## Executive Summary

This repository has **zero CI/CD automation**. No GitHub Actions, no pre-commit hooks, no automated dependency updates, no documentation builds. For a production open-source astronomy library, this represents a significant gap requiring immediate attention.

---

## Current State Analysis

### 1. GitHub Workflows

| Item | Status |
|------|--------|
| `.github/workflows/` | **Does not exist** |
| Any workflow files | None |

### 2. Alternative CI Systems

| System | Status |
|--------|--------|
| Travis CI (`.travis.yml`) | Not configured |
| CircleCI (`.circleci/`) | Not configured |
| Azure Pipelines (`azure-pipelines.yml`) | Not configured |
| GitLab CI (`.gitlab-ci.yml`) | Not configured |

### 3. pyproject.toml Automation

**Build System:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Development Dependencies (defined but not CI-integrated):**
```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.0.0",
  "ruff>=0.1.0",
  "mypy>=1.0.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "ruff>=0.14.10",
]
```

**Tool Configuration Present:**
- `[tool.pytest.ini_options]` - Test markers for `slow` tests
- `[tool.ruff]` - Linting rules configured
- `[tool.mypy]` - Strict type checking configured

**Scripts/Hooks:** None defined

### 4. Pre-commit Hooks

| Item | Status |
|------|--------|
| `.pre-commit-config.yaml` | **Does not exist** |
| Any git hooks | None configured |

### 5. Automated Dependency Updates

| Service | Status |
|---------|--------|
| Dependabot (`.github/dependabot.yml`) | **Not configured** |
| Renovate (`renovate.json`) | **Not configured** |

### 6. Documentation Build

| Item | Status |
|------|--------|
| `.readthedocs.yaml` | **Does not exist** |
| `docs/` directory | **Does not exist** |
| Sphinx/mkdocs config | None |

### 7. Other Automation

| Item | Status |
|------|--------|
| Makefile | None |
| tox.ini | None |
| noxfile.py | None |
| Justfile | None |

---

## Gap Analysis for Production Open-Source

### Critical Gaps

1. **No automated testing** - 33,000 lines of test code exist but never run automatically
2. **No linting enforcement** - Ruff configured but not enforced on PRs
3. **No type checking in CI** - mypy configured (strict) but not automated
4. **No dependency security scanning** - Vulnerable packages could go unnoticed
5. **No release automation** - Manual PyPI publishing is error-prone

### Important Gaps

6. **No pre-commit hooks** - Code quality checks only at developer discretion
7. **No documentation build** - API reference not generated automatically
8. **No code coverage tracking** - Test effectiveness unknown
9. **No dependency freshness** - No Dependabot/Renovate for updates

### Nice-to-Have Gaps

10. **No benchmark/performance CI** - Astronomy code is performance-sensitive
11. **No integration test with real TESS data** - Data pipeline testing
12. **No multi-platform testing** - macOS/Linux/Windows matrix

---

## Recommended CI/CD Setup

### Tier 1: Essential (Implement Immediately)

#### `.github/workflows/ci.yml`
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --extra dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --extra dev
      - run: uv run mypy src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --frozen --all-extras
      - run: uv run pytest -m "not slow" --tb=short
      - run: uv run pytest -m slow --tb=short
        if: github.event_name == 'push'
```

#### `.github/workflows/release.yml`
```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

#### `.github/dependabot.yml`
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      astronomy:
        patterns: ["astropy", "lightkurve", "transitleastsquares"]
      dev:
        patterns: ["pytest", "ruff", "mypy"]
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
```

#### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

### Tier 2: Production Quality

#### Coverage Reporting
Add to test job:
```yaml
- run: uv run pytest --cov=src/bittr_tess_vetter --cov-report=xml
- uses: codecov/codecov-action@v4
  with:
    files: coverage.xml
```

#### Security Scanning
```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
    - run: uv pip compile pyproject.toml -o requirements.txt
    - uses: pyupio/safety@v3
```

### Tier 3: Astronomy-Specific

#### Data Integration Tests
```yaml
integration:
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
    - run: uv sync --frozen --all-extras
    - run: uv run pytest tests/test_integration/ -v
      env:
        MAST_API_TOKEN: ${{ secrets.MAST_API_TOKEN }}
```

#### Documentation Build (ReadTheDocs)
`.readthedocs.yaml`:
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.12"
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
sphinx:
  configuration: docs/conf.py
```

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | `.github/workflows/ci.yml` | 2h | Critical |
| P0 | `.pre-commit-config.yaml` | 30min | High |
| P1 | `.github/dependabot.yml` | 15min | Medium |
| P1 | Coverage integration | 1h | Medium |
| P2 | Release automation | 1h | High |
| P2 | Security scanning | 30min | Medium |
| P3 | Documentation build | 4h | Medium |
| P3 | Multi-platform matrix | 1h | Low |

---

## Comparison: Astronomy Library Standards

| Feature | lightkurve | astropy | This Repo |
|---------|------------|---------|-----------|
| GitHub Actions | Yes | Yes | No |
| Pre-commit | Yes | Yes | No |
| Dependabot | Yes | Yes | No |
| codecov | Yes | Yes | No |
| ReadTheDocs | Yes | Yes | No |
| PyPI release CI | Yes | Yes | No |

---

## Recommendations

1. **Immediate:** Add basic CI workflow with lint/type/test matrix
2. **This week:** Configure pre-commit hooks and Dependabot
3. **This month:** Set up release automation and documentation build
4. **Ongoing:** Add integration tests for MAST API interactions

The 33,000 lines of existing tests represent significant untapped value. Automating these tests would immediately improve code quality confidence.
