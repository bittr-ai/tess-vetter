from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from bittr_tess_vetter.api import compute_aperture_family_depth_curve, localize_transit_source
from bittr_tess_vetter.api.catalogs import query_gaia_by_position_sync
from bittr_tess_vetter.api.stellar_dilution import (
    build_host_hypotheses_from_profile,
    compute_dilution_scenarios,
)
from bittr_tess_vetter.api.wcs_localization import ReferenceSource
from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef


@dataclass(frozen=True)
class CandidateEphemeris:
    period_days: float
    t0_btjd: float
    duration_hours: float
    depth_ppm: float


def _require_lightkurve() -> Any:
    try:
        import lightkurve as lk  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "This example requires lightkurve to download TPFs.\n"
            "Install with: pip install lightkurve astroquery"
        ) from e
    return lk


def _download_tpf_fits_path(
    *, tic_id: int, sector: int, author: str, exptime_seconds: int | None
) -> Path:
    lk = _require_lightkurve()
    author_norm = author.upper()
    search = lk.search_targetpixelfile(
        f"TIC {tic_id}",
        sector=sector,
        author=author_norm,
        exptime=exptime_seconds,
    )
    if len(search) < 1:
        raise SystemExit(f"No TPF found for TIC {tic_id} sector {sector} author={author_norm}")

    tpf = search.download()
    path = getattr(tpf, "path", None)
    if not path:
        raise SystemExit("lightkurve TPF download did not expose a local FITS path")
    return Path(str(path))


def _tpf_fits_data_from_path(*, fits_path: Path, ref: TPFFitsRef) -> TPFFitsData:
    with fits.open(fits_path) as hdu_list:
        primary_header = hdu_list[0].header
        if len(hdu_list) < 2:
            raise ValueError("Invalid TPF FITS structure: missing data extension")

        data_hdu = hdu_list[1]
        data = data_hdu.data
        if data is None:
            raise ValueError("Invalid TPF FITS structure: data extension has no table")

        time = np.asarray(data["TIME"], dtype=np.float64)
        flux = np.asarray(data["FLUX"], dtype=np.float64)
        flux_err = None
        if "FLUX_ERR" in data.columns.names:
            flux_err = np.asarray(data["FLUX_ERR"], dtype=np.float64)

        quality = np.asarray(data["QUALITY"], dtype=np.int32)

        aperture_hdu = None
        if len(hdu_list) > 2 and hdu_list[2].name == "APERTURE":
            aperture_hdu = hdu_list[2]
            aperture_mask = np.asarray(aperture_hdu.data, dtype=np.int32)
        else:
            aperture_mask = np.ones(flux.shape[1:], dtype=np.int32)

        # SPOC TPFs store celestial WCS on the APERTURE HDU header.
        wcs = WCS(aperture_hdu.header if aperture_hdu is not None else data_hdu.header)

        camera = int(primary_header.get("CAMERA", 1))
        ccd = int(primary_header.get("CCD", 1))

        meta: dict[str, Any] = {
            "RA_OBJ": primary_header.get("RA_OBJ"),
            "DEC_OBJ": primary_header.get("DEC_OBJ"),
            "SECTOR": primary_header.get("SECTOR"),
        }

        return TPFFitsData(
            ref=ref,
            time=time,
            flux=flux,
            flux_err=flux_err,
            wcs=wcs,
            aperture_mask=aperture_mask,
            quality=quality,
            camera=camera,
            ccd=ccd,
            meta=meta,
        )


def _build_reference_sources_from_gaia(
    *, ra: float, dec: float, radius_arcsec: float
) -> list[ReferenceSource]:
    gaia = query_gaia_by_position_sync(ra, dec, radius_arcsec=radius_arcsec, timeout=90)
    refs: list[ReferenceSource] = [{"name": "target", "ra": float(ra), "dec": float(dec)}]
    for n in gaia.neighbors:
        refs.append(
            {"name": f"gaia_dr3:{int(n.source_id)}", "ra": float(n.ra), "dec": float(n.dec)}
        )
    return refs


def _print_dilution_plausibility(
    *, tic_id: int, ra: float, dec: float, radius_arcsec: float, observed_depth_ppm: float
) -> None:
    gaia = query_gaia_by_position_sync(ra, dec, radius_arcsec=radius_arcsec, timeout=90)
    primary_g = gaia.source.phot_g_mean_mag if gaia.source is not None else None
    companions = [
        (int(n.source_id), float(n.separation_arcsec), float(n.phot_g_mean_mag), n.delta_mag, None)
        for n in gaia.neighbors
        if n.phot_g_mean_mag is not None
    ]
    primary, companion_hypotheses = build_host_hypotheses_from_profile(
        tic_id=tic_id,
        primary_g_mag=float(primary_g) if primary_g is not None else None,
        primary_radius_rsun=None,
        close_bright_companions=companions,
    )

    scenarios = compute_dilution_scenarios(
        observed_depth_ppm=float(observed_depth_ppm),
        primary=primary,
        companions=companion_hypotheses,
    )

    print("\nDilution plausibility (depth-only):")
    for s in scenarios[:12]:
        frac = s.host.estimated_flux_fraction
        true_ppm = s.true_depth_ppm
        impossible = true_ppm > 1_000_000.0
        tag = "IMPOSSIBLE (>100%)" if impossible else ""
        print(
            f'- {s.host.name} sep={s.host.separation_arcsec:5.1f}" '
            f"G={s.host.g_mag if s.host.g_mag is not None else float('nan'):.2f} "
            f"flux_frac={frac:.3e} true_depth={true_ppm:9.0f} ppm {tag}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blend/host-ambiguity demo: WCS localization + Gaia hypotheses + dilution plausibility."
    )
    parser.add_argument("--tic-id", type=int, default=188646744)
    parser.add_argument("--sectors", type=str, default="55,75,82,83")
    parser.add_argument("--author", type=str, default="SPOC")
    parser.add_argument("--exptime-seconds", type=int, default=120)
    parser.add_argument("--period-days", type=float, default=14.2423724)
    parser.add_argument("--t0-btjd", type=float, default=3540.263170)
    parser.add_argument("--duration-hours", type=float, default=4.56)
    parser.add_argument("--depth-ppm", type=float, default=232.0)
    parser.add_argument("--gaia-radius-arcsec", type=float, default=60.0)
    args = parser.parse_args()

    tic_id = int(args.tic_id)
    sectors = [int(s.strip()) for s in str(args.sectors).split(",") if s.strip()]
    eph = CandidateEphemeris(
        period_days=float(args.period_days),
        t0_btjd=float(args.t0_btjd),
        duration_hours=float(args.duration_hours),
        depth_ppm=float(args.depth_ppm),
    )

    print(f"TIC {tic_id} blend/localization demo")
    print(f"Ephemeris: P={eph.period_days} d, t0={eph.t0_btjd} BTJD, dur={eph.duration_hours} h")
    print(f"Observed depth: {eph.depth_ppm} ppm")
    print(f"Sectors: {sectors}")

    first_ra_dec: tuple[float, float] | None = None

    for sector in sectors:
        fits_path = _download_tpf_fits_path(
            tic_id=tic_id,
            sector=sector,
            author=str(args.author),
            exptime_seconds=int(args.exptime_seconds) if args.exptime_seconds else None,
        )
        ref = TPFFitsRef(
            tic_id=tic_id,
            sector=sector,
            author=str(args.author).lower(),
            exptime_seconds=int(args.exptime_seconds) if args.exptime_seconds else None,
        )
        tpf_fits = _tpf_fits_data_from_path(fits_path=fits_path, ref=ref)

        ra_obj = tpf_fits.meta.get("RA_OBJ")
        dec_obj = tpf_fits.meta.get("DEC_OBJ")
        if isinstance(ra_obj, (int, float)) and isinstance(dec_obj, (int, float)):
            first_ra_dec = (float(ra_obj), float(dec_obj))

        reference_sources: list[ReferenceSource] | None = None
        if first_ra_dec is not None:
            reference_sources = _build_reference_sources_from_gaia(
                ra=first_ra_dec[0],
                dec=first_ra_dec[1],
                radius_arcsec=float(args.gaia_radius_arcsec),
            )

        loc = localize_transit_source(
            tpf_fits=tpf_fits,
            period=eph.period_days,
            t0=eph.t0_btjd,
            duration_hours=eph.duration_hours,
            reference_sources=reference_sources,
        )

        target_sep_arcsec = loc.distances_to_sources.get("target")
        nearest_name: str | None = None
        nearest_sep_arcsec: float | None = None
        if loc.distances_to_sources:
            nearest_name, nearest_sep_arcsec = min(
                loc.distances_to_sources.items(), key=lambda kv: float(kv[1])
            )
        nearest_non_target_name: str | None = None
        nearest_non_target_sep_arcsec: float | None = None
        non_target = {k: v for k, v in loc.distances_to_sources.items() if k != "target"}
        if non_target:
            nearest_non_target_name, nearest_non_target_sep_arcsec = min(
                non_target.items(), key=lambda kv: float(kv[1])
            )

        print(f"\nSector {sector} localization:")
        print(f"- verdict: {loc.verdict}")
        if isinstance(target_sep_arcsec, (int, float)) and np.isfinite(float(target_sep_arcsec)):
            print(f"- target_separation_arcsec: {float(target_sep_arcsec):.2f}")
        if nearest_name is not None and nearest_sep_arcsec is not None:
            print(f"- nearest_reference: {nearest_name}")
            print(f"- nearest_separation_arcsec: {float(nearest_sep_arcsec):.2f}")
        if nearest_non_target_name is not None and nearest_non_target_sep_arcsec is not None:
            print(f"- nearest_non_target: {nearest_non_target_name}")
            print(
                f"- nearest_non_target_separation_arcsec: {float(nearest_non_target_sep_arcsec):.2f}"
            )
        print(f"- warnings: {len(loc.warnings)}")
        if loc.verdict_rationale:
            for line in loc.verdict_rationale[:2]:
                print(f"- rationale: {line}")

        ap = compute_aperture_family_depth_curve(
            tpf_fits=tpf_fits,
            period=eph.period_days,
            t0=eph.t0_btjd,
            duration_hours=eph.duration_hours,
        )
        print("Aperture-family check:")
        print(f"- blend_indicator: {ap.blend_indicator}")
        print(f"- slope_significance: {ap.depth_slope_significance:.2f}σ")
        print(f"- recommended_aperture_px: {ap.recommended_aperture_px}")

    if first_ra_dec is not None:
        _print_dilution_plausibility(
            tic_id=tic_id,
            ra=first_ra_dec[0],
            dec=first_ra_dec[1],
            radius_arcsec=float(args.gaia_radius_arcsec),
            observed_depth_ppm=eph.depth_ppm,
        )
    else:
        print("\nSkipping Gaia/dilution step (could not read RA_OBJ/DEC_OBJ from TPF headers).")

    print("\nInterpretation guide:")
    print(
        "- OFF_TARGET to a very faint Gaia source is often a red flag, but can also be a failure mode"
    )
    print("  for saturated targets; use dilution plausibility to rule out impossible hosts.")
    print(
        "- If multiple sectors disagree, treat host attribution as ambiguous and plan resolved follow-up."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
