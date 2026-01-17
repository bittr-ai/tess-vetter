#!/usr/bin/env python3
"""Execute a notebook incrementally up to a specific cell (by id or index).

This is intended for tutorial authoring: run one step at a time, inspect the
outputs, then (manually) hardcode "Expected Output" blocks.

Examples:
  # Execute from the start through the code cell with id 'run-baseline'
  uv run python scripts/exec_notebook_until.py docs/tutorials/09-toi-5807-validation-walkthrough.ipynb --to-id run-baseline

  # Execute the first 10 cells (0-based index inclusive)
  uv run python scripts/exec_notebook_until.py docs/tutorials/09-toi-5807-validation-walkthrough.ipynb --to-index 10

Notes:
  - Executes in a fresh kernel each run.
  - Writes outputs back into the same notebook by default (`--inplace`).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=str)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--to-id", type=str, help="Execute through the code cell with this id")
    g.add_argument("--to-index", type=int, help="Execute through this cell index (0-based, inclusive)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument(
        "--inplace",
        dest="inplace",
        action="store_true",
        default=True,
        help="Write executed outputs back into the notebook file (default: enabled).",
    )
    ap.add_argument(
        "--no-inplace",
        dest="inplace",
        action="store_false",
        help="Do not modify the notebook file in-place.",
    )
    args = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient

    path = Path(args.notebook).expanduser().resolve()
    nb = nbformat.read(path, as_version=4)

    if args.to_index is not None:
        stop_idx = int(args.to_index)
    else:
        stop_idx = None
        for i, cell in enumerate(nb.cells):
            if cell.get("cell_type") == "code" and cell.get("id") == str(args.to_id):
                stop_idx = i
                break
        if stop_idx is None:
            raise SystemExit(f"Could not find code cell with id={args.to_id!r}")

    # Execute cells from 0..stop_idx (inclusive); clear outputs beyond stop.
    for i, cell in enumerate(nb.cells):
        if i > stop_idx and cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # nbclient executes the whole notebook; for incremental execution we run a
    # truncated copy and then copy outputs back.
    nb_trunc = copy.deepcopy(nb)
    nb_trunc.cells = nb_trunc.cells[: stop_idx + 1]

    client = NotebookClient(nb_trunc, timeout=int(args.timeout), kernel_name="python3", allow_errors=False)
    client.execute()

    for i in range(stop_idx + 1):
        if nb.cells[i].get("cell_type") == "code":
            nb.cells[i]["outputs"] = nb_trunc.cells[i].get("outputs", [])
            nb.cells[i]["execution_count"] = nb_trunc.cells[i].get("execution_count")

    if args.inplace:
        nbformat.write(nb, path)

    print({"notebook": str(path), "executed_through_index": stop_idx})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
