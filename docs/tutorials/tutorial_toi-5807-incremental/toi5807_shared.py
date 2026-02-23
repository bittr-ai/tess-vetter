from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Prefer local source tree over any installed package so the tutorial stays in-sync
# with the repo checkout.
_repo_root = Path(__file__).resolve().parents[3]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import tess_vetter.api as btv  # noqa: E402

# Stable identifiers
TIC_ID = 188646744
TOI_LABEL = 'TOI-5807.01'

# Stable ephemeris defaults (avoid depending on ExoFOP changing over time)
PERIOD_DAYS = 14.2423724
T0_BTJD = 3540.26317
DURATION_HOURS = 4.046

# Stable stellar parameters (used by duration-consistency checks)
STELLAR = btv.StellarParams(radius=1.65, mass=1.47, teff=6816.0, logg=4.17)

# Coordinates used for catalog checks (V06)
# These match ExoFOP (RA=20:34:16.13, Dec=43:21:38.48) and the bundled TPF WCS header.
RA_DEG = 308.5672083333333
DEC_DEG = 43.36068888888889


def load_dataset() -> btv.LocalDataset:
    return btv.load_tutorial_target('tic188646744')


def stitch_pdcsap(ds: btv.LocalDataset) -> btv.LightCurve:
    stitched = btv.stitch_lightcurves(
        [
            {
                'time': np.asarray(lc.time, dtype=np.float64),
                'flux': np.asarray(lc.flux, dtype=np.float64),
                'flux_err': np.asarray(lc.flux_err, dtype=np.float64),
                'sector': int(sector),
                'quality': np.asarray(
                    lc.quality if getattr(lc, 'quality', None) is not None else np.zeros(len(lc.time)),
                    dtype=np.int32,
                ),
            }
            for sector, lc in sorted(ds.lc_by_sector.items())
        ]
    )
    return btv.LightCurve(
        time=stitched.time,
        flux=stitched.flux,
        flux_err=stitched.flux_err,
        quality=stitched.quality,
    )


def stitch_pdcsap_sectors(ds: btv.LocalDataset, sectors: list[int]) -> btv.LightCurve:
    """Stitch PDCSAP light curves for a chosen subset of sectors."""
    stitched = btv.stitch_lightcurves(
        [
            {
                'time': np.asarray(ds.lc_by_sector[int(sector)].time, dtype=np.float64),
                'flux': np.asarray(ds.lc_by_sector[int(sector)].flux, dtype=np.float64),
                'flux_err': np.asarray(ds.lc_by_sector[int(sector)].flux_err, dtype=np.float64),
                'sector': int(sector),
                'quality': np.asarray(
                    ds.lc_by_sector[int(sector)].quality
                    if getattr(ds.lc_by_sector[int(sector)], 'quality', None) is not None
                    else np.zeros(len(ds.lc_by_sector[int(sector)].time)),
                    dtype=np.int32,
                ),
            }
            for sector in list(sectors)
        ]
    )
    return btv.LightCurve(
        time=stitched.time,
        flux=stitched.flux,
        flux_err=stitched.flux_err,
        quality=stitched.quality,
    )


def detrend_transit_masked_bin_median(
    lc: btv.LightCurve,
    *,
    bin_hours: float = 12.0,
    duration_buffer_factor: float = 1.5,
    sigma_clip: float | None = 5.0,
) -> btv.LightCurve:
    """Lightweight transit-masked detrend using piecewise median bins.

    This is a deterministic, dependency-light alternative to wotan-based
    detrending. It estimates a long-term trend using only out-of-transit
    points, then divides it out while preserving the overall normalization.
    """
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    flux_err = np.asarray(lc.flux_err, dtype=np.float64)

    if len(time) < 10:
        return lc

    # Mask transits with a modest buffer (in phase/time) so the trend estimate
    # isn't biased by the transit signal.
    in_tr = btv.get_in_transit_mask(
        time,
        PERIOD_DAYS,
        T0_BTJD,
        DURATION_HOURS * float(duration_buffer_factor),
    )
    oot = ~in_tr

    # Optional outlier rejection on the OOT points.
    if sigma_clip is not None and float(sigma_clip) > 0:
        good = btv.sigma_clip(flux, sigma=float(sigma_clip))
        oot = oot & np.asarray(good, dtype=bool)

    bin_width_days = float(bin_hours) / 24.0
    if bin_width_days <= 0:
        raise ValueError("bin_hours must be > 0")

    t_min = float(np.nanmin(time))
    t_max = float(np.nanmax(time))
    n_bins = int(np.ceil((t_max - t_min) / bin_width_days))
    n_bins = max(3, min(n_bins, 5000))

    edges = t_min + np.arange(n_bins + 1, dtype=np.float64) * bin_width_days
    centers = (edges[:-1] + edges[1:]) / 2.0

    medians = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        m = oot & (time >= edges[i]) & (time < edges[i + 1])
        if int(np.sum(m)) < 10:
            continue
        medians[i] = float(np.nanmedian(flux[m]))

    ok = np.isfinite(medians)
    if int(np.sum(ok)) < 3:
        # Fallback: time-based running median without masking.
        flat = btv.flatten(time, flux, window_length=float(bin_hours) / 24.0)
        return btv.LightCurve(
            time=time,
            flux=np.asarray(flat, dtype=np.float64),
            flux_err=flux_err,
            quality=lc.quality,
            valid_mask=getattr(lc, "valid_mask", None),
        )

    trend = np.interp(time, centers[ok], medians[ok])
    trend = np.where(np.isfinite(trend), trend, float(np.nanmedian(flux)))
    trend_median = float(np.nanmedian(trend))
    if not np.isfinite(trend_median) or abs(trend_median) < 1e-12:
        trend_median = 1.0

    detrended_flux = flux / trend * trend_median
    detrended_err = flux_err / trend * trend_median

    return btv.LightCurve(
        time=time,
        flux=detrended_flux.astype(np.float64),
        flux_err=detrended_err.astype(np.float64),
        quality=lc.quality,
        valid_mask=getattr(lc, "valid_mask", None),
    )


def estimate_depth_ppm(stitched: btv.LightCurve) -> tuple[float, float]:
    in_tr = btv.get_in_transit_mask(stitched.time, PERIOD_DAYS, T0_BTJD, DURATION_HOURS)
    out_tr = btv.get_out_of_transit_mask(
        stitched.time,
        PERIOD_DAYS,
        T0_BTJD,
        DURATION_HOURS,
        buffer_factor=3.0,
    )
    depth_hat, depth_err = btv.measure_transit_depth(stitched.flux, in_tr, out_tr)
    return float(depth_hat * 1e6), float(depth_err * 1e6)


def make_candidate(depth_ppm: float) -> btv.Candidate:
    ephem = btv.Ephemeris(period_days=PERIOD_DAYS, t0_btjd=T0_BTJD, duration_hours=DURATION_HOURS)
    return btv.Candidate(ephemeris=ephem, depth_ppm=float(depth_ppm))


def estimate_planet_radius_earth(*, depth_ppm: float, stellar_radius_rsun: float = STELLAR.radius) -> float:
    """Back-of-the-envelope Rp estimate assuming depth≈(Rp/Rs)^2 (no dilution)."""
    rp_rs = float(np.sqrt(float(depth_ppm) / 1e6))
    rp_rsun = rp_rs * float(stellar_radius_rsun)
    rsun_to_rearth = 109.076
    return float(rp_rsun * rsun_to_rearth)


def make_session(
    *,
    stitched: btv.LightCurve,
    candidate: btv.Candidate,
    network: bool,
    preset: str = 'default',
    stellar: btv.StellarParams = STELLAR,
    tpf: btv.TPFStamp | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    context: dict[str, Any] | None = None,
) -> btv.VettingSession:
    return btv.VettingSession.from_api(
        lc=btv.LightCurve(
            time=stitched.time,
            flux=stitched.flux,
            flux_err=stitched.flux_err,
            quality=stitched.quality,
        ),
        candidate=candidate,
        stellar=stellar,
        tpf=tpf,
        network=bool(network),
        preset=str(preset),
        tic_id=TIC_ID,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        context=context,
    )


def print_context(*, network: bool) -> None:
    if hasattr(STELLAR, "model_dump"):
        stellar_payload: Any = STELLAR.model_dump()
    elif is_dataclass(STELLAR):
        stellar_payload = asdict(STELLAR)
    else:
        stellar_payload = {
            "radius": getattr(STELLAR, "radius", None),
            "mass": getattr(STELLAR, "mass", None),
            "teff": getattr(STELLAR, "teff", None),
            "logg": getattr(STELLAR, "logg", None),
        }

    payload: dict[str, Any] = {
        'network': bool(network),
        'tic_id': TIC_ID,
        'toi': TOI_LABEL,
        'ephemeris': {
            'period_days': PERIOD_DAYS,
            't0_btjd': T0_BTJD,
            'duration_hours': DURATION_HOURS,
        },
        'stellar': stellar_payload,
        'coords_deg': {'ra': RA_DEG, 'dec': DEC_DEG},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def artifact_dirs(*, step_id: str) -> tuple[Path, Path | None]:
    """Return (run_out_dir, docs_out_dir_if_writable)."""
    run_out_dir = Path('persistent_cache') / 'tutorial_toi-5807-incremental' / step_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    docs_out_dir = Path('docs') / 'tutorials' / 'artifacts' / 'tutorial_toi-5807-incremental' / step_id
    try:
        docs_out_dir.mkdir(parents=True, exist_ok=True)
        return run_out_dir, docs_out_dir
    except Exception:
        return run_out_dir, None
