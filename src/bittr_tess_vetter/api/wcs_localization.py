"""WCS-aware localization via difference imaging for the public API.

This facade exists to provide a stable import surface for host applications.
It delegates computation to `bittr_tess_vetter.pixel.wcs_localization`.

References:
    - Twicken et al. 2018 (2018PASP..130f4502T): difference image centroid offsets (Kepler DV)
    - Bryson et al. 2013 (2013PASP..125..889B): background false positive localization diagnostics
    - Greisen & Calabretta 2002 (2002A&A...395.1061G), Calabretta & Greisen 2002 (2002A&A...395.1077C):
      FITS WCS conventions used for sky↔pixel transforms (via astropy.wcs)
    - Astropy Collaboration 2013 (2013A&A...558A..33A): astropy.wcs implementation
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np

from bittr_tess_vetter.api.references import (
    ASTROPY_COLLAB_2013,
    BRYSON_2013,
    CALABRETTA_GREISEN_2002,
    GREISEN_CALABRETTA_2002,
    TWICKEN_2018,
    cite,
    cites,
)
from bittr_tess_vetter.pixel.cadence_mask import default_cadence_mask
from bittr_tess_vetter.pixel.wcs_localization import (
    LocalizationResult,
    LocalizationVerdict,
)
from bittr_tess_vetter.pixel.wcs_localization import (
    _compute_out_of_transit_mask as _compute_out_of_transit_mask_windowed,
)
from bittr_tess_vetter.pixel.wcs_localization import (
    _compute_transit_mask as _compute_transit_mask_windowed,
)
from bittr_tess_vetter.pixel.wcs_localization import (
    compute_difference_image_centroid as _compute_difference_image_centroid,
)
from bittr_tess_vetter.pixel.wcs_localization import (
    localize_transit_source as _localize_transit_source,
)

if False:  # TYPE_CHECKING
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData


class ReferenceSource(TypedDict, total=False):
    """Reference sky position used for host-hypothesis localization.

    Keys are intentionally minimal and JSON-friendly so host apps can pass
    Gaia/TIC neighbors without depending on internal models.
    """

    name: str
    ra: float
    dec: float
    meta: dict[str, Any]


class DifferenceImageCentroidDiagnostics(TypedDict):
    n_cadences_total: int
    n_cadences_used: int
    n_cadences_dropped: int
    n_in_transit: int
    n_out_of_transit: int
    baseline_mode: str
    warnings: list[str]


REFERENCES = [
    ref.to_dict()
    for ref in [
        TWICKEN_2018,
        BRYSON_2013,
        GREISEN_CALABRETTA_2002,
        CALABRETTA_GREISEN_2002,
        ASTROPY_COLLAB_2013,
    ]
]


@cites(
    cite(TWICKEN_2018, "difference images and centroid offsets (Kepler DV)"),
    cite(BRYSON_2013, "difference-image localization diagnostics"),
    cite(GREISEN_CALABRETTA_2002, "FITS WCS framework (Paper I)"),
    cite(CALABRETTA_GREISEN_2002, "celestial WCS conventions (Paper II)"),
    cite(ASTROPY_COLLAB_2013, "astropy.wcs implementation"),
)
def localize_transit_source(
    *,
    tpf_fits: TPFFitsData,
    period: float,
    t0: float,
    duration_hours: float,
    reference_sources: list[ReferenceSource] | None = None,
    method: str = "difference_image",
    bootstrap_draws: int = 500,
    bootstrap_seed: int = 42,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
) -> LocalizationResult:
    """Localize the transit source using WCS-aware difference imaging."""
    # TypedDict is runtime-compatible with the underlying list[dict] contract.
    return _localize_transit_source(
        tpf_fits=tpf_fits,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        reference_sources=reference_sources,  # type: ignore[arg-type]
        method=method,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        oot_margin_mult=oot_margin_mult,
        oot_window_mult=oot_window_mult,
    )


@cites(
    cite(TWICKEN_2018, "difference image computation and centroid offsets (Kepler DV)"),
    cite(BRYSON_2013, "difference-image localization diagnostics"),
)
def compute_difference_image_centroid_diagnostics(
    *,
    tpf_fits: TPFFitsData,
    period: float,
    t0: float,
    duration_hours: float,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    method: Literal["centroid", "gaussian_fit"] = "centroid",
) -> tuple[tuple[float, float], np.ndarray, DifferenceImageCentroidDiagnostics]:
    """Compute a windowed difference image + centroid and return small diagnostics.

    This is a low-level helper for host apps that want to run PRF-lite scoring
    on a consistent difference image while also reporting basic cadence/mask
    bookkeeping for auditability.

    Delegates to `bittr_tess_vetter.pixel.wcs_localization.compute_difference_image_centroid`.
    """
    cadence_mask = default_cadence_mask(time=tpf_fits.time, flux=tpf_fits.flux, quality=tpf_fits.quality)
    time = np.asarray(tpf_fits.time[cadence_mask], dtype=np.float64)
    duration_days = float(duration_hours) / 24.0

    in_transit_mask = _compute_transit_mask_windowed(time, float(period), float(t0), duration_days)
    out_of_transit_mask = _compute_out_of_transit_mask_windowed(
        time,
        float(period),
        float(t0),
        duration_days,
        float(oot_margin_mult),
        oot_window_mult,
    )

    centroid_rc, diff_image = _compute_difference_image_centroid(
        tpf_fits=tpf_fits,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        oot_margin_mult=oot_margin_mult,
        oot_window_mult=oot_window_mult,
        method=method,
    )

    n_total = int(tpf_fits.time.shape[0])
    n_used = int(np.sum(cadence_mask))

    diagnostics: DifferenceImageCentroidDiagnostics = {
        "n_cadences_total": n_total,
        "n_cadences_used": n_used,
        "n_cadences_dropped": n_total - n_used,
        "n_in_transit": int(np.sum(in_transit_mask)),
        "n_out_of_transit": int(np.sum(out_of_transit_mask)),
        "baseline_mode": "global" if oot_window_mult is None else "local",
        "warnings": [f"dropped_bad_cadences:{n_total - n_used}"] if n_used < n_total else [],
    }
    return centroid_rc, diff_image, diagnostics


__all__ = [
    "ReferenceSource",
    "DifferenceImageCentroidDiagnostics",
    "LocalizationResult",
    "LocalizationVerdict",
    "compute_difference_image_centroid_diagnostics",
    "localize_transit_source",
]
