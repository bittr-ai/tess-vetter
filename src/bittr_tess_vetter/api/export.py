"""Export helpers for vetting results (researcher-facing, policy-free)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from bittr_tess_vetter.api.vetting_report import render_validation_report_markdown
from bittr_tess_vetter.validation.result_schema import VettingBundleResult

ExportFormat = Literal["json", "csv", "md"]


def _bundle_to_jsonable(bundle: VettingBundleResult, *, include_raw: bool) -> dict[str, Any]:
    data = bundle.model_dump()
    if include_raw:
        return data
    # Strip raw payloads (can be large/unstructured); keep all metrics/flags/notes.
    for r in data.get("results", []) or []:
        if isinstance(r, dict):
            r.pop("raw", None)
    return data


def export_bundle(
    bundle: VettingBundleResult,
    *,
    format: ExportFormat,
    path: str | Path | None = None,
    include_raw: bool = False,
    title: str = "Vetting Report",
) -> str | None:
    """Export a vetting bundle to a stable interchange format.

    This helper is policy-free: it does not impose thresholds or verdicts.

    Args:
        bundle: Vetting bundle result.
        format: "json", "csv", or "md".
        path: If provided, writes to this path and returns None. Otherwise returns the string.
        include_raw: If True, include check-level `raw` payloads in JSON (and CSV `raw_json`).
        title: Markdown title when format="md".
    """
    fmt = str(format).lower()
    out_path = Path(path).expanduser().resolve() if path is not None else None

    if fmt == "json":
        payload = {"schema_version": 1, "bundle": _bundle_to_jsonable(bundle, include_raw=include_raw)}
        text = json.dumps(payload, indent=2, sort_keys=True)
        if out_path is not None:
            out_path.write_text(text + "\n")
            return None
        return text + "\n"

    if fmt == "md":
        md = render_validation_report_markdown(title=title, bundle=bundle)
        if out_path is not None:
            out_path.write_text(md)
            return None
        return md

    if fmt == "csv":
        rows: list[dict[str, Any]] = []
        for r in bundle.results:
            row: dict[str, Any] = {
                "id": r.id,
                "name": r.name,
                "status": r.status,
                "confidence": r.confidence,
                "metrics_json": json.dumps(r.metrics, sort_keys=True),
                "flags_json": json.dumps(list(r.flags), sort_keys=True),
                "notes_json": json.dumps(list(r.notes), sort_keys=True),
                "provenance_json": json.dumps(r.provenance, sort_keys=True),
            }
            if include_raw:
                row["raw_json"] = json.dumps(r.raw, sort_keys=True, default=str)
            rows.append(row)

        fieldnames = list(rows[0].keys()) if rows else [
            "id",
            "name",
            "status",
            "confidence",
            "metrics_json",
            "flags_json",
            "notes_json",
            "provenance_json",
        ]

        if out_path is not None:
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in rows:
                    w.writerow(row)
            return None

        # Return CSV as string
        from io import StringIO

        buf = StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        return buf.getvalue()

    raise ValueError(f"Unsupported export format: {format!r}")


__all__ = ["ExportFormat", "export_bundle"]

