from __future__ import annotations

import tomllib
from pathlib import Path


def test_project_scripts_include_package_name_alias() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["btv"] == "tess_vetter.cli.enrich_cli:main"
    assert scripts["tess-vetter"] == "tess_vetter.cli.enrich_cli:main"

