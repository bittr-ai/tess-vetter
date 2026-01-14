"""Pixel-level localization diagnostics for TPF-based vetting.

This module is a lightweight helper for diagnosing whether difference-image and
centroid offsets are driven by an astrophysical off-target source or by a
reference-coordinate assumption (e.g., assuming the target is at the TPF center).

It intentionally does not require WCS; it reports multiple reasonable reference
points derived from the pixel data itself (OOT brightest pixel, OOT flux centroid).

References:
    - Twicken et al. 2018 (2018PASP..130f4502T): difference images and centroid-offset diagnostics
    - Bryson et al. 2013 (2013PASP..125..889B): pixel-level localization diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bittr_tess_vetter.pixel.aperture import TransitParams
from bittr_tess_vetter.pixel.cadence_mask import default_cadence_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _compute_transit_mask(
    time: NDArray[np.floating],
    transit_params: TransitParams,
) -> NDArray[np.bool_]:
    # Keep consistent with bittr_tess_vetter.pixel.difference._compute_transit_mask
    phase = ((time - transit_params.t0) % transit_params.period) / transit_params.period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half_duration_phase = (transit_params.duration / 2) / transit_params.period
    result: NDArray[np.bool_] = np.abs(phase) <= half_duration_phase
    return result


def _flux_centroid_rc(image: NDArray[np.floating]) -> tuple[float, float]:
    img = np.asarray(image, dtype=np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    tot = float(np.sum(img))
    if tot <= 0:
        return (float("nan"), float("nan"))
    rows, cols = img.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    r = float(np.sum(rr * img) / tot)
    c = float(np.sum(cc * img) / tot)
    return (r, c)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _localization_score(distance_pixels: float, n_rows: int, n_cols: int) -> float:
    # Matches the spirit of bittr_tess_vetter.pixel.difference.compute_difference_image
    max_distance = float(np.sqrt((n_rows / 2) ** 2 + (n_cols / 2) ** 2))
    if max_distance <= 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (float(distance_pixels) / max_distance))))


@dataclass(frozen=True)
class LocalizationDiagnosticsResult:
    tic_id: int | None
    sector: int | None
    shape: tuple[int, int, int]

    n_in_transit: int
    n_out_of_transit: int

    stamp_center_rc: tuple[int, int]
    oot_brightest_rc: tuple[int, int]
    oot_centroid_rc: tuple[float, float]

    diff_brightest_rc: tuple[int, int]
    in_transit_centroid_rc: tuple[float, float]

    dist_diff_to_stamp_center_px: float
    dist_ootbright_to_stamp_center_px: float
    dist_diff_to_ootbright_px: float
    dist_diff_to_ootcentroid_px: float

    localization_score_to_stamp_center: float
    localization_score_to_oot_brightest: float
    localization_score_to_oot_centroid: float

    note: str
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tic_id": self.tic_id,
            "sector": self.sector,
            "shape": list(self.shape),
            "n_in_transit": int(self.n_in_transit),
            "n_out_of_transit": int(self.n_out_of_transit),
            "stamp_center_rc": [int(self.stamp_center_rc[0]), int(self.stamp_center_rc[1])],
            "oot_brightest_rc": [int(self.oot_brightest_rc[0]), int(self.oot_brightest_rc[1])],
            "oot_centroid_rc": [float(self.oot_centroid_rc[0]), float(self.oot_centroid_rc[1])],
            "diff_brightest_rc": [int(self.diff_brightest_rc[0]), int(self.diff_brightest_rc[1])],
            "in_transit_centroid_rc": [
                float(self.in_transit_centroid_rc[0]),
                float(self.in_transit_centroid_rc[1]),
            ],
            "distances_pixels": {
                "diff_to_stamp_center": float(self.dist_diff_to_stamp_center_px),
                "oot_brightest_to_stamp_center": float(self.dist_ootbright_to_stamp_center_px),
                "diff_to_oot_brightest": float(self.dist_diff_to_ootbright_px),
                "diff_to_oot_centroid": float(self.dist_diff_to_ootcentroid_px),
            },
            "localization_scores": {
                "to_stamp_center": float(self.localization_score_to_stamp_center),
                "to_oot_brightest": float(self.localization_score_to_oot_brightest),
                "to_oot_centroid": float(self.localization_score_to_oot_centroid),
            },
            "note": self.note,
            "extra": self.extra,
        }


def compute_localization_diagnostics(
    *,
    tpf_data: NDArray[np.floating],
    time: NDArray[np.floating],
    transit_params: TransitParams,
) -> tuple[LocalizationDiagnosticsResult, dict[str, NDArray[np.floating]]]:
    """Compute localization diagnostics and return small images for optional export."""
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D, got shape {tpf_data.shape}")
    n_times, n_rows, n_cols = tpf_data.shape
    if time.ndim != 1 or time.size != n_times:
        raise ValueError("time must be 1D and match tpf_data first axis")
    if n_rows == 0 or n_cols == 0:
        raise ValueError("tpf_data has invalid spatial dimensions")

    cadence_mask = default_cadence_mask(
        time=time,
        flux=tpf_data,
        quality=np.zeros(int(time.shape[0]), dtype=np.int32),
        require_finite_pixels=True,
    )
    tpf_data = tpf_data[cadence_mask]
    time = time[cadence_mask]
    n_times = int(time.size)
    if n_times < 3:
        raise ValueError("Insufficient valid cadences after masking.")

    in_mask = _compute_transit_mask(time, transit_params)
    out_mask = ~in_mask
    n_in = int(np.sum(in_mask))
    n_out = int(np.sum(out_mask))
    if n_in == 0 or n_out == 0:
        raise ValueError(f"Need both in- and out-of-transit cadences (n_in={n_in}, n_out={n_out}).")

    in_img = np.nanmedian(tpf_data[in_mask], axis=0)
    out_img = np.nanmedian(tpf_data[out_mask], axis=0)
    diff = out_img - in_img

    diff_flat = int(np.argmax(diff))
    diff_r, diff_c = np.unravel_index(diff_flat, diff.shape)
    diff_rc = (int(diff_r), int(diff_c))

    oot_flat = int(np.argmax(out_img))
    oot_r, oot_c = np.unravel_index(oot_flat, out_img.shape)
    oot_rc = (int(oot_r), int(oot_c))

    stamp_center = (int(n_rows // 2), int(n_cols // 2))
    oot_cent = _flux_centroid_rc(out_img)
    in_cent = _flux_centroid_rc(in_img)

    d_diff_center = _dist((diff_rc[0], diff_rc[1]), (stamp_center[0], stamp_center[1]))
    d_oot_center = _dist((oot_rc[0], oot_rc[1]), (stamp_center[0], stamp_center[1]))
    d_diff_oot = _dist((diff_rc[0], diff_rc[1]), (oot_rc[0], oot_rc[1]))
    d_diff_ootcent = _dist((diff_rc[0], diff_rc[1]), oot_cent)

    res = LocalizationDiagnosticsResult(
        tic_id=None,
        sector=None,
        shape=(int(n_times), int(n_rows), int(n_cols)),
        n_in_transit=n_in,
        n_out_of_transit=n_out,
        stamp_center_rc=stamp_center,
        oot_brightest_rc=oot_rc,
        oot_centroid_rc=oot_cent,
        diff_brightest_rc=diff_rc,
        in_transit_centroid_rc=in_cent,
        dist_diff_to_stamp_center_px=float(d_diff_center),
        dist_ootbright_to_stamp_center_px=float(d_oot_center),
        dist_diff_to_ootbright_px=float(d_diff_oot),
        dist_diff_to_ootcentroid_px=float(d_diff_ootcent),
        localization_score_to_stamp_center=_localization_score(d_diff_center, n_rows, n_cols),
        localization_score_to_oot_brightest=_localization_score(d_diff_oot, n_rows, n_cols),
        localization_score_to_oot_centroid=_localization_score(d_diff_ootcent, n_rows, n_cols),
        note="Distances to stamp center can be misleading if the target is not centered in the cutout. "
        "Prefer distances/scores relative to OOT-derived target proxies (oot_brightest_rc / oot_centroid_rc).",
        extra={
            "phase_mask_method": "difference_like",
        },
    )

    images: dict[str, NDArray[np.floating]] = {
        "in_transit_median": in_img.astype(np.float32),
        "out_of_transit_median": out_img.astype(np.float32),
        "difference_image": diff.astype(np.float32),
    }
    return res, images
