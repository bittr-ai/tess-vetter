"""Researcher-facing helpers for FPP workflows (policy-free).

These helpers reduce notebook glue around TRICERATOPS(+):
- hydrating a PersistentCache from local per-sector light curves
- parsing common high-resolution imaging contrast-curve files (ExoFOP .tbl)

They do not impose thresholds or make any validation verdicts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from bittr_tess_vetter.api.datasets import LocalDataset
from bittr_tess_vetter.api.fpp import ContrastCurve
from bittr_tess_vetter.api.io import PersistentCache
from bittr_tess_vetter.api.lightcurve import make_data_ref
from bittr_tess_vetter.domain.lightcurve import LightCurveData


def hydrate_cache_from_dataset(
    *,
    dataset: LocalDataset,
    tic_id: int,
    flux_type: str = "pdcsap",
    cache_dir: str | Path | None = None,
    cadence_seconds: float = 120.0,
    sectors: list[int] | None = None,
) -> PersistentCache:
    """Hydrate a PersistentCache with per-sector LightCurveData for TRICERATOPS.

    This matches the pattern used in `04-real-candidate-validation.ipynb`:
    store one LightCurveData per sector under a `make_data_ref(...)` key so the
    FPP engine can load per-sector light curves from cache.

    Args:
        dataset: LocalDataset containing `lc_by_sector`.
        tic_id: TIC identifier to embed in cached LightCurveData and cache keys.
        flux_type: Flux type component of the cache key (e.g. "pdcsap").
        cache_dir: Optional directory for the on-disk cache.
        cadence_seconds: Cadence to record in LightCurveData (informational).
        sectors: Optional subset of sectors to cache (default: all in dataset).

    Returns:
        A PersistentCache instance populated with the requested sectors.
    """
    if not dataset.lc_by_sector:
        raise ValueError("dataset.lc_by_sector is empty")
    if int(tic_id) <= 0:
        raise ValueError("tic_id must be positive")

    cache = PersistentCache(cache_dir=cache_dir)
    wanted = {int(s) for s in sectors} if sectors is not None else None

    for sector, lc in sorted(dataset.lc_by_sector.items()):
        sec = int(sector)
        if wanted is not None and sec not in wanted:
            continue

        t = np.asarray(lc.time, dtype=np.float64)
        f = np.asarray(lc.flux, dtype=np.float64)
        e = np.asarray(lc.flux_err, dtype=np.float64) if lc.flux_err is not None else np.zeros_like(f)

        q = np.zeros(len(t), dtype=np.int32)
        valid = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)

        lc_data = LightCurveData(
            time=t,
            flux=f,
            flux_err=e,
            quality=q,
            valid_mask=valid,
            tic_id=int(tic_id),
            sector=sec,
            cadence_seconds=float(cadence_seconds),
        )

        key = make_data_ref(int(tic_id), sec, str(flux_type))
        cache.put(key, lc_data)

    return cache


_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def load_contrast_curve_exofop_tbl(
    path: str | Path,
    *,
    filter: str | None = None,
) -> ContrastCurve:
    """Parse an ExoFOP-format contrast curve `.tbl` file into a ContrastCurve.

    The common format is:

    - comment / metadata header lines
    - a numeric table with at least 2 columns: separation_arcsec, delta_mag
      (an optional third column like dmag_rms is ignored)

    Parsing behavior is intentionally forgiving:
    - accepts comma- or whitespace-delimited rows
    - ignores non-numeric rows

    Args:
        path: Path to the `.tbl` file.
        filter: Optional imaging band label to store in the returned ContrastCurve.
            If None, defaults to "Vis". (TRICERATOPS normalization is applied later.)

    Returns:
        ContrastCurve with separation_arcsec and delta_mag sorted by separation.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Contrast curve file not found: {p}")

    seps: list[float] = []
    dmags: list[float] = []

    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        if not (_FLOAT_RE.match(parts[0]) and _FLOAT_RE.match(parts[1])):
            continue

        try:
            sep = float(parts[0])
            dmag = float(parts[1])
        except Exception:
            continue

        if np.isfinite(sep) and np.isfinite(dmag):
            seps.append(sep)
            dmags.append(dmag)

    if len(seps) < 2:
        raise ValueError(f"Contrast curve parse failed: found {len(seps)} numeric rows in {p}")

    sep_arr = np.asarray(seps, dtype=np.float64)
    dmag_arr = np.asarray(dmags, dtype=np.float64)
    order = np.argsort(sep_arr)

    return ContrastCurve(
        separation_arcsec=sep_arr[order],
        delta_mag=dmag_arr[order],
        filter=str(filter) if filter is not None else "Vis",
    )


__all__ = ["hydrate_cache_from_dataset", "load_contrast_curve_exofop_tbl"]
