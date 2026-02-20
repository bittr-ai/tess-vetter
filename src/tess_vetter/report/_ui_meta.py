"""Build-time UI metadata artifact builder for report payloads."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tess_vetter.report.field_catalog import FIELD_CATALOG, FieldKey
from tess_vetter.report.schema import CheckSummaryModel, ReferenceEntryModel

UI_META_VERSION = "1"


def _field_entry(key: FieldKey) -> dict[str, Any]:
    spec = FIELD_CATALOG[key]
    entry = asdict(spec)
    entry["key"] = key.value
    return entry


def _model_property_types(model_schema: dict[str, Any]) -> dict[str, str]:
    props = model_schema.get("properties", {})
    result: dict[str, str] = {}
    for name, prop in props.items():
        type_name = prop.get("type")
        if isinstance(type_name, list):
            result[name] = "|".join(sorted(str(v) for v in type_name))
        elif isinstance(type_name, str):
            result[name] = type_name
        elif "$ref" in prop:
            result[name] = "object"
        elif "anyOf" in prop:
            result[name] = "union"
        else:
            result[name] = "unknown"
    return result


def build_ui_meta_artifact() -> dict[str, Any]:
    """Return deterministic field/check/reference metadata for UI build tooling."""
    sorted_keys = sorted(FIELD_CATALOG.keys(), key=lambda k: k.value)
    fields = [_field_entry(key) for key in sorted_keys]

    check_schema = CheckSummaryModel.model_json_schema()
    reference_schema = ReferenceEntryModel.model_json_schema()

    return {
        "version": UI_META_VERSION,
        "fields": fields,
        "checks": {
            "path": "summary.checks[*]",
            "property_types": _model_property_types(check_schema),
            "schema": check_schema,
        },
        "references": {
            "path": "summary.references[*]",
            "property_types": _model_property_types(reference_schema),
            "schema": reference_schema,
        },
    }


def write_ui_meta_artifact(path: str | Path) -> Path:
    """Write deterministic UI metadata JSON artifact to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_ui_meta_artifact()
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write report UI metadata JSON artifact.")
    parser.add_argument(
        "--out",
        default="docs/_build/html/report_ui_meta.json",
        help="Output path for report UI metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = write_ui_meta_artifact(args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()


__all__ = ["UI_META_VERSION", "build_ui_meta_artifact", "write_ui_meta_artifact"]
