# Contributing to tess-vetter

Thank you for your interest in contributing to tess-vetter!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/bittr-ai/tess-vetter.git
cd tess-vetter

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (including dev tools and optional extras used in CI/release checks)
uv sync --all-extras --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run broad regression coverage gate (matches CI modest gate)
uv run pytest --cov=tess_vetter

# Run core-library coverage gate (>90, excludes adapter/vendor surfaces)
uv run coverage run --rcfile=coverage.core.ini -m pytest -q -m "not slow"
uv run coverage report --rcfile=coverage.core.ini

# Run specific test file
uv run pytest tests/test_api/test_periodogram_wrappers.py
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Type Checking

```bash
uv run mypy src/tess_vetter
```

## Pull Request Process

1. Fork the repository and create a branch from `main`
2. Make your changes and add tests if applicable
3. Ensure all tests pass and linting is clean
4. Update documentation if needed
5. Submit a pull request

## Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for test additions/changes

## Questions?

Open an issue or discussion on GitHub.
