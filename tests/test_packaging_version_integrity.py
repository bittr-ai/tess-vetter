from __future__ import annotations

import tomllib
from pathlib import Path

import bittr_tess_vetter


def _load_project_table() -> dict[str, object]:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return pyproject["project"]


def test_package_version_matches_pyproject() -> None:
    project = _load_project_table()
    assert bittr_tess_vetter.__version__ == project["version"]


def test_all_extra_does_not_self_reference_package_name() -> None:
    project = _load_project_table()
    package_name = project["name"]
    all_extra = project["optional-dependencies"]["all"]
    assert not any(
        dep == package_name or dep.startswith(f"{package_name}[") for dep in all_extra
    )
