"""Local dataset loading helpers (researcher-facing, optional IO).

This module provides small, dependency-light utilities for loading common
bittr-style dataset folders (including this repo's tutorial datasets) into
in-memory API types (`LightCurve`, `TPFStamp`).

Design:
- No network access.
- No pandas dependency (uses stdlib `csv`).
- Produces metrics-ready arrays; does not apply vetting policy thresholds.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tess_vetter.api.types import LightCurve, TPFStamp


@dataclass(frozen=True)
class LocalDataset:
    """In-memory dataset assembled from a local folder."""

    schema_version: int
    root: Path
    lc_by_sector: dict[int, LightCurve] = field(default_factory=dict)
    tpf_by_sector: dict[int, TPFStamp] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "root": str(self.root),
            "sectors_lc": sorted(self.lc_by_sector.keys()),
            "sectors_tpf": sorted(self.tpf_by_sector.keys()),
            "artifacts": sorted(self.artifacts.keys()),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable metadata summary (does not embed array data)."""
        return self.summary()


def _read_csv_no_pandas(path: Path) -> dict[str, np.ndarray]:
    """Read a tutorial-format CSV with comment header lines."""
    time: list[float] = []
    flux: list[float] = []
    flux_err: list[float] = []
    quality: list[int] = []

    with path.open(newline="") as f:
        def _non_comment_lines() -> Any:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                yield line

        reader = csv.DictReader(_non_comment_lines())
        for row in reader:
            time.append(float(row["time_btjd"]))
            flux.append(float(row["flux"]))
            flux_err.append(float(row["flux_err"]))
            quality.append(int(row.get("quality", "0")))

    return {
        "time_btjd": np.asarray(time, dtype=np.float64),
        "flux": np.asarray(flux, dtype=np.float64),
        "flux_err": np.asarray(flux_err, dtype=np.float64),
        "quality": np.asarray(quality, dtype=np.int32),
    }


def _maybe_build_wcs(wcs_header: Any) -> Any:
    """Best-effort WCS construction.

    Returns either an astropy WCS object (if available) or the raw header dict.
    """
    try:
        from astropy.wcs import WCS  # type: ignore[import-not-found]

        return WCS(wcs_header)
    except Exception:
        return wcs_header


def load_local_dataset(
    path: str | Path,
    *,
    pattern_overrides: dict[str, str] | None = None,
) -> LocalDataset:
    """Load a local dataset folder into API types.

    Supported inputs (by default patterns):

    - Light curves: `sector{sector}_pdcsap.csv` with columns:
      `time_btjd,flux,flux_err,quality` (tutorial format)
    - TPF stamps: `sector{sector}_tpf.npz` with keys:
      `time,flux,flux_err,wcs_header,aperture_mask,quality`

    Args:
        path: Dataset folder.
        pattern_overrides: Optional overrides for filename patterns:
            - "lc_csv": e.g. "sector{sector}_pdcsap.csv"
            - "tpf_npz": e.g. "sector{sector}_tpf.npz"

    Returns:
        LocalDataset with `lc_by_sector` and `tpf_by_sector` populated.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    patterns = {
        "lc_csv": "sector{sector}_pdcsap.csv",
        "tpf_npz": "sector{sector}_tpf.npz",
    }
    if pattern_overrides:
        patterns.update({str(k): str(v) for k, v in pattern_overrides.items()})

    lc_by_sector: dict[int, LightCurve] = {}
    tpf_by_sector: dict[int, TPFStamp] = {}
    artifacts: dict[str, Any] = {}

    # Light curves: scan matching CSVs
    lc_glob = patterns["lc_csv"].format(sector="*")
    for csv_path in sorted(root.glob(lc_glob)):
        name = csv_path.name
        try:
            sector_str = name.split("_")[0].replace("sector", "")
            sector = int(sector_str)
        except Exception:
            continue

        data = _read_csv_no_pandas(csv_path)
        ok = data["quality"] == 0
        lc_by_sector[int(sector)] = LightCurve(
            time=data["time_btjd"][ok],
            flux=data["flux"][ok],
            flux_err=data["flux_err"][ok],
        )

    # TPF stamps: scan matching NPZs
    tpf_glob = patterns["tpf_npz"].format(sector="*")
    for npz_path in sorted(root.glob(tpf_glob)):
        name = npz_path.name
        try:
            sector_str = name.split("_")[0].replace("sector", "")
            sector = int(sector_str)
        except Exception:
            continue

        d = np.load(npz_path, allow_pickle=True)
        wcs_header = d["wcs_header"].item() if "wcs_header" in d else None
        tpf_by_sector[int(sector)] = TPFStamp(
            time=np.asarray(d["time"], dtype=np.float64),
            flux=np.asarray(d["flux"], dtype=np.float64),
            flux_err=np.asarray(d["flux_err"], dtype=np.float64) if "flux_err" in d else None,
            wcs=_maybe_build_wcs(wcs_header) if wcs_header is not None else None,
            aperture_mask=np.asarray(d["aperture_mask"], dtype=bool)
            if "aperture_mask" in d
            else None,
            quality=np.asarray(d["quality"], dtype=np.int32) if "quality" in d else None,
        )

    # Store any non-parsed artifacts for discovery (researchers may load manually).
    artifacts["files"] = sorted(p.name for p in root.iterdir() if p.is_file())

    return LocalDataset(
        schema_version=1,
        root=root,
        lc_by_sector=lc_by_sector,
        tpf_by_sector=tpf_by_sector,
        artifacts=artifacts,
    )


def load_tutorial_target(name: str) -> LocalDataset:
    """Load a bundled tutorial target dataset by name.

    This is a repo convenience wrapper around :func:`load_local_dataset`.
    """
    base = Path(__file__).resolve().parents[3] / "docs" / "tutorials" / "data"
    return load_local_dataset(base / name)


__all__ = ["LocalDataset", "load_local_dataset", "load_tutorial_target"]
