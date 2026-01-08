"""WCS-aware localization via difference imaging for the public API.

This facade exists to provide a stable import surface for host applications.
It delegates computation to `bittr_tess_vetter.pixel.wcs_localization`.
"""

from __future__ import annotations

from typing import Any, TypedDict

from bittr_tess_vetter.pixel.wcs_localization import (
    LocalizationResult,
    LocalizationVerdict,
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


def localize_transit_source(
    *,
    tpf_fits: "TPFFitsData",
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


__all__ = [
    "ReferenceSource",
    "LocalizationResult",
    "LocalizationVerdict",
    "localize_transit_source",
]
