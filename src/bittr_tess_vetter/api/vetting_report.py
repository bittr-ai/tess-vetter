"""Vetting result reporting helpers (researcher-facing).

These helpers convert structured vetting outputs into easy-to-consume formats:
- plain-text console tables
- markdown reports
- compact JSON-serializable summaries

All helpers are policy-free: they do not impose new thresholds. They only
reformat and select metrics already produced by checks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VettingTableOptions:
    """Formatting options for :func:`format_vetting_table`."""

    include_header: bool = True
    include_counts: bool = True
    include_provenance: bool = False
    include_inputs_summary: bool = False
    max_name_chars: int = 30
    max_metrics_per_row: int = 0
    metric_keys: tuple[str, ...] = ()


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    return str(x)


def _format_confidence(v: float | None) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.3f}"
    except Exception:
        return ""


def _select_metrics_for_row(
    metrics: dict[str, float | int | str | bool | None] | None,
    *,
    metric_keys: Sequence[str],
    max_metrics_per_row: int,
) -> list[tuple[str, str]]:
    if not metrics:
        return []
    pairs: list[tuple[str, str]] = []

    if metric_keys:
        for k in metric_keys:
            if k in metrics:
                pairs.append((str(k), _safe_str(metrics.get(k))))
    else:
        # deterministic ordering
        for k in sorted(metrics.keys()):
            v = metrics.get(k)
            if isinstance(v, (float, int, str, bool, type(None))):
                pairs.append((str(k), _safe_str(v)))

    if max_metrics_per_row > 0:
        pairs = pairs[: int(max_metrics_per_row)]
    return pairs


def _table(rows: list[list[str]], headers: list[str] | None = None) -> str:
    if headers is not None:
        rows = [headers, *rows]

    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    norm: list[list[str]] = []
    for r in rows:
        r2 = list(r) + [""] * (n_cols - len(r))
        norm.append([_safe_str(c) for c in r2])

    widths = [0] * n_cols
    for r in norm:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    out_lines: list[str] = []
    for idx, r in enumerate(norm):
        line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))
        out_lines.append(line.rstrip())
        if headers is not None and idx == 0:
            sep = "  ".join("-" * w for w in widths).rstrip()
            out_lines.append(sep)
    return "\n".join(out_lines)


def format_vetting_table(
    bundle: Any,
    *,
    options: VettingTableOptions | None = None,
) -> str:
    """Format a :class:`~bittr_tess_vetter.validation.result_schema.VettingBundleResult` as a console table."""
    opt = options or VettingTableOptions()

    results = list(getattr(bundle, "results", []) or [])
    n_ok = int(getattr(bundle, "n_passed", 0))
    n_err = int(getattr(bundle, "n_failed", 0))
    n_skip = int(getattr(bundle, "n_unknown", 0))

    lines: list[str] = []
    if opt.include_header:
        lines.append("Vetting Results")
        lines.append("=" * 60)

    if opt.include_counts:
        lines.append(f"Checks: {len(results)}  ok: {n_ok}  error: {n_err}  skipped: {n_skip}")
        lines.append("")

    headers = ["ID", "Name", "Status", "Confidence"]
    if opt.max_metrics_per_row > 0 or opt.metric_keys:
        headers.append("Metrics")

    rows: list[list[str]] = []
    for r in results:
        rid = _safe_str(getattr(r, "id", ""))
        name = _safe_str(getattr(r, "name", ""))
        status = _safe_str(getattr(r, "status", ""))
        conf = _format_confidence(getattr(r, "confidence", None))

        if opt.max_name_chars > 0 and len(name) > opt.max_name_chars:
            name = name[: max(0, opt.max_name_chars - 1)] + "…"

        row = [rid, name, status, conf]

        metrics = getattr(r, "metrics", None)
        if opt.max_metrics_per_row > 0 or opt.metric_keys:
            pairs = _select_metrics_for_row(
                metrics,
                metric_keys=opt.metric_keys,
                max_metrics_per_row=int(opt.max_metrics_per_row),
            )
            metrics_str = ", ".join(f"{k}={v}" for k, v in pairs)
            row.append(metrics_str)

        rows.append(row)

    lines.append(_table(rows, headers=headers))

    if opt.include_inputs_summary:
        inputs = getattr(bundle, "inputs_summary", None)
        if isinstance(inputs, dict) and inputs:
            lines.append("")
            lines.append("Inputs Summary")
            lines.append("-" * 60)
            lines.append(_table([[k, _safe_str(v)] for k, v in sorted(inputs.items())], headers=None))

    if opt.include_provenance:
        prov = getattr(bundle, "provenance", None)
        if isinstance(prov, dict) and prov:
            lines.append("")
            lines.append("Provenance")
            lines.append("-" * 60)
            lines.append(_table([[k, _safe_str(v)] for k, v in sorted(prov.items())], headers=None))

    return "\n".join(lines).rstrip() + "\n"


def summarize_bundle(
    bundle: Any,
    *,
    check_ids: Sequence[str] | None = None,
    include_metrics: bool = True,
    metric_keys: Sequence[str] | None = None,
    include_flags: bool = True,
    include_notes: bool = False,
    include_provenance: bool = True,
    include_inputs_summary: bool = True,
) -> dict[str, Any]:
    """Return a compact JSON-serializable summary of a vetting bundle."""
    results = list(getattr(bundle, "results", []) or [])
    if check_ids is not None:
        wanted = {str(x) for x in check_ids}
        results = [r for r in results if str(getattr(r, "id", "")) in wanted]

    by_id: dict[str, Any] = {}
    for r in results:
        rid = str(getattr(r, "id", ""))
        entry: dict[str, Any] = {
            "id": rid,
            "name": getattr(r, "name", None),
            "status": getattr(r, "status", None),
            "confidence": getattr(r, "confidence", None),
        }

        if include_metrics:
            metrics = getattr(r, "metrics", None) or {}
            if metric_keys:
                entry["metrics"] = {k: metrics.get(k) for k in metric_keys if k in metrics}
            else:
                entry["metrics"] = dict(metrics)

        if include_flags:
            entry["flags"] = list(getattr(r, "flags", []) or [])

        if include_notes:
            entry["notes"] = list(getattr(r, "notes", []) or [])

        by_id[rid] = entry

    out: dict[str, Any] = {
        "counts": {
            "checks": int(len(getattr(bundle, "results", []) or [])),
            "ok": int(getattr(bundle, "n_passed", 0)),
            "error": int(getattr(bundle, "n_failed", 0)),
            "skipped": int(getattr(bundle, "n_unknown", 0)),
        },
        "results_by_id": by_id,
    }

    if include_inputs_summary:
        inputs = getattr(bundle, "inputs_summary", None)
        if isinstance(inputs, dict):
            out["inputs_summary"] = dict(inputs)

    if include_provenance:
        prov = getattr(bundle, "provenance", None)
        if isinstance(prov, dict):
            out["provenance"] = dict(prov)

    return out


def render_validation_report_markdown(
    *,
    title: str,
    bundle: Any,
    include_table: bool = True,
    table_options: VettingTableOptions | None = None,
    extra_sections: Iterable[tuple[str, str]] | None = None,
) -> str:
    """Render a lightweight Markdown report for a vetting run."""
    md: list[str] = [f"# {title}", ""]

    if include_table:
        table = format_vetting_table(bundle, options=table_options).rstrip()
        md.extend(["```", table, "```", ""])

    if extra_sections:
        for section_title, body in extra_sections:
            md.extend([f"## {section_title}", "", body.rstrip(), ""])

    return "\n".join(md).rstrip() + "\n"


__all__ = [
    "VettingTableOptions",
    "format_vetting_table",
    "summarize_bundle",
    "render_validation_report_markdown",
]

