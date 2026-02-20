# Releasing `tess-vetter`

## Local preflight

Run the full local smoke suite:

```bash
./scripts/release_smoke.sh
```

## Versioning

1. Bump `version` in `pyproject.toml`.
2. Bump `__version__` in `src/tess_vetter/__init__.py` to match.
3. Run the packaging guardrail test:

```bash
uv run pytest -q tests/test_packaging_version_integrity.py
```

4. Update `CHANGELOG.md`.

## Tag and push

```bash
git tag -a "vX.Y.Z" -m "vX.Y.Z"
git push origin "vX.Y.Z"
```

## What happens next (CI)

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

- Builds sdist + wheel
- Installs from the wheel on Python 3.11/3.12 and runs import smoke checks
- Publishes to PyPI via trusted publishing (OIDC)
- Creates a GitHub Release with the build artifacts attached

## First-time setup checklist (maintainers)

- Configure PyPI trusted publishing for this repo (see the comment in `release.yml`).
- Decide whether you want coverage reporting and configure Codecov (optional).
