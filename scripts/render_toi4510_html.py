"""Render an HTML report for TOI-4510.01 using real MAST data."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from bittr_tess_vetter.platform.io.mast_client import MASTClient
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.report import build_report, render_html

# TOI-4510.01 parameters (ExoFOP)
TIC_ID = 340458804
TOI = "4510.01"
PERIOD_DAYS = 194.243
T0_BTJD = 2039.7625
DURATION_HOURS = 9.256
DEPTH_PPM = 811.0

def main() -> None:
    # Download all sectors
    print(f"Downloading light curves for TIC {TIC_ID}...")
    client = MASTClient()
    lc_list = client.download_all_sectors(tic_id=TIC_ID)
    print(f"Downloaded {len(lc_list)} sectors, stitching...")

    # Stitch into single LightCurve
    time = np.concatenate([lc.time for lc in lc_list])
    flux = np.concatenate([lc.flux for lc in lc_list])
    flux_err = np.concatenate([lc.flux_err for lc in lc_list])

    # Sort by time
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    print(f"Total: {len(time)} points, {time[-1] - time[0]:.1f} days baseline")

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=PERIOD_DAYS, t0_btjd=T0_BTJD, duration_hours=DURATION_HOURS)
    candidate = Candidate(ephemeris=eph, depth_ppm=DEPTH_PPM)

    # Build report
    print("Building report...")
    report = build_report(lc, candidate, tic_id=TIC_ID, toi=TOI)
    print(f"SNR: {report.lc_summary.snr:.1f}, transits: {report.lc_summary.n_transits}")

    # Render HTML
    html = render_html(report)
    out = Path(__file__).parent.parent / "working_docs" / "report" / "toi4510_report.html"
    out.write_text(html)
    print(f"Wrote {out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
