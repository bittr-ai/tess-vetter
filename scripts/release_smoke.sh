#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Lint / format"
uv run ruff check .
uv run ruff format --check .

echo "==> Unit tests"
uv run pytest -q

echo "==> Build docs"
uv run sphinx-build -b html docs docs/_build/html >/dev/null
uv run python -m bittr_tess_vetter.report._ui_meta --out docs/_build/html/report_ui_meta.json

echo "==> Build artifacts"
uv run hatch build -t sdist -t wheel

echo "==> Inspect sdist for unwanted payload"
python - <<'PY'
import tarfile
from pathlib import Path

sdist = sorted(Path("dist").glob("*.tar.gz"))[-1]
with tarfile.open(sdist, "r:gz") as tf:
    names = tf.getnames()

root = names[0].split("/")[0] + "/"
bad_prefixes = [
    root + ".uv-cache/",
    root + "working_docs/",
    root + ".github/",
    root + "uv.lock",
    root + "dist/",
    root + "build/",
]
hits = [n for n in names if any(n.startswith(p) for p in bad_prefixes)]
if hits:
    raise SystemExit(f"sdist contains unexpected files (showing up to 20): {hits[:20]}")
print("sdist ok:", sdist)
PY

echo "==> Import smoke from wheel"
python - <<'PY'
import sys
import subprocess
from pathlib import Path

wheel = sorted(Path("dist").glob("*.whl"))[-1]
subprocess.check_call([sys.executable, "-m", "venv", ".release_smoke_venv"])
pip = Path(".release_smoke_venv") / ("Scripts" if sys.platform.startswith("win") else "bin") / "pip"
py = Path(".release_smoke_venv") / ("Scripts" if sys.platform.startswith("win") else "bin") / "python"
subprocess.check_call([str(pip), "install", "--quiet", str(wheel)])
subprocess.check_call([str(py), "-c", "import bittr_tess_vetter.api as btv; print(btv.__name__)"])
print("wheel ok:", wheel)
PY

echo "==> Done"
