#!/usr/bin/env python3
"""Sync per-cell "Expected Output" blocks from executed notebook outputs.

Usage:
  uv run python scripts/sync_notebook_expected_outputs.py docs/tutorials/09-toi-5807-validation-walkthrough.ipynb

Behavior:
- Finds markdown cells containing markers like: <!-- EXPECTED:<cell_id> -->
- Replaces the contents of the surrounding fenced block with the executed output
  extracted from the referenced code cell (<cell_id>).

Notes:
- Output is extracted from:
  - stream stdout/stderr
  - execute_result / display_data text/plain
  - error tracebacks
- The notebook must already be executed (outputs present) for this to have effect.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MARKER_RE = re.compile(r"<!--\s*EXPECTED:(?P<cell_id>[-a-zA-Z0-9_]+)\s*-->")
FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(?P<body>.*)\n```", re.DOTALL)


def _collect_text_output(cell: dict[str, Any]) -> str:
    parts: list[str] = []
    for out in cell.get("outputs", []) or []:
        otype = out.get("output_type")
        if otype == "stream":
            txt = out.get("text", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            parts.append(str(txt))
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {}) or {}
            txt = data.get("text/plain")
            if txt is None:
                continue
            if isinstance(txt, list):
                txt = "".join(txt)
            parts.append(str(txt) + "\n")
        elif otype == "error":
            tb = out.get("traceback", []) or []
            if isinstance(tb, list):
                parts.append("\n".join(str(x) for x in tb) + "\n")
            else:
                parts.append(str(tb) + "\n")

    text = "".join(parts).strip("\n")
    return text + "\n" if text else ""


def _replace_fenced_body(md: str, new_body: str) -> str:
    m = FENCE_RE.search(md)
    if not m:
        return md
    start, end = m.span("body")
    return md[:start] + new_body.rstrip("\n") + md[end:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=str, help="Path to executed .ipynb")
    args = ap.parse_args()

    path = Path(args.notebook).expanduser().resolve()
    nb = json.loads(path.read_text())

    by_id: dict[str, dict[str, Any]] = {}
    for cell in nb.get("cells", []) or []:
        cid = cell.get("id")
        if isinstance(cid, str):
            by_id[cid] = cell

    changed = False
    for cell in nb.get("cells", []) or []:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        md = "".join(src) if isinstance(src, list) else str(src or "")
        mm = MARKER_RE.search(md)
        if not mm:
            continue
        target_id = mm.group("cell_id")
        target = by_id.get(target_id)
        if not target or target.get("cell_type") != "code":
            continue

        out_text = _collect_text_output(target)
        if not out_text:
            out_text = "(no output)\n"

        new_md = _replace_fenced_body(md, out_text)
        if new_md != md:
            cell["source"] = [new_md]
            changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

