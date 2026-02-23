from __future__ import annotations

import ast
from pathlib import Path

REPORT_SRC = Path(__file__).resolve().parents[2] / "src" / "tess_vetter" / "report"


def _forbidden_imports(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "tess_vetter.api" or name.startswith("tess_vetter.api."):
                    violations.append(f"import {name}")

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "tess_vetter.api" or module.startswith("tess_vetter.api."):
                violations.append(f"from {module} import ...")
            if node.level > 0 and (module == "api" or module.startswith("api.")):
                violations.append(f"from {'.' * node.level}{module} import ...")

    return violations


def test_report_layer_does_not_import_api() -> None:
    offenders: list[str] = []

    for path in sorted(REPORT_SRC.rglob("*.py")):
        for imp in _forbidden_imports(path):
            rel = path.relative_to(REPORT_SRC.parent.parent.parent)
            offenders.append(f"{rel}: {imp}")

    assert not offenders, "Forbidden import(s) from report layer to api layer:\n" + "\n".join(offenders)
