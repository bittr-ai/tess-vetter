# TOI-5807 incremental tutorial

This folder contains small, step-by-step notebooks for vetting **TOI-5807.01 / TIC 188646744**.

Design:
- Each notebook is runnable on its own.
- Each notebook includes **Expected Output** blocks and (when relevant) embeds **pre-rendered plots** committed under `docs/tutorials/artifacts/tutorial_toi-5807-incremental/`.
- A tiny shared helper lives in `toi5807_shared.py` to keep constants and setup consistent.

Run notebooks from the repo root so relative paths resolve (e.g. `docs/tutorials/data/...`).

Plotting requires `matplotlib` (or install the package extra: `pip install 'tess-vetter[plotting]'`).
