# Releasing `tess-vetter`

## Local preflight

Run the full local smoke suite:

```bash
./scripts/release_smoke.sh
```

Use `uv run ...` for all release/test commands so tooling runs in the project
environment (`.venv`). Avoid running bare `pytest` from another active venv.

## Coverage gate (required)

Before tagging a release:

1. Coverage reporting in Codecov must be enabled for the repository.
2. The release commit must pass the configured Codecov/status-check coverage gate on the default branch.
3. The README coverage badge must resolve to a live coverage value (not `unknown`).

## Versioning

1. Bump `version` in `pyproject.toml`.
2. Bump `__version__` in `src/tess_vetter/__init__.py` to match.
3. Run the packaging guardrail test:

```bash
uv run pytest -q tests/test_packaging_version_integrity.py
```

4. Update `CHANGELOG.md`.

## Zenodo DOI readiness (no minting in this repo)

After Zenodo has minted the DOI for the tagged release, update these exact locations before finalizing release comms:

1. `CITATION.cff`
   - Set `identifiers[0].value` to the minted version DOI (example: `10.5281/zenodo.18743917`).
2. `README.md`
   - In `Project DOI (Zenodo)`, use the concept DOI (latest release DOI) for the badge/link (example: `10.5281/zenodo.18743916`).

Release-order note:
1. Create/push tag and let GitHub Release publish artifacts.
2. Wait for Zenodo to ingest the release and mint DOI.
3. Apply DOI updates above in a follow-up commit (no new tag required unless policy says otherwise).

## Tag and push

From a clean `main` that is synced with `origin/main`:

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
- Configure Codecov coverage reporting and enforce a required coverage status check.
