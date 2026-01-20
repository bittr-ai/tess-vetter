"""Style presets and constants for plotting.

This module defines:
- STYLES: matplotlib rcParams presets ("default", "paper", "presentation")
- COLORS: Standard colors for transit/model data
- COLORMAPS: Recommended colormaps for image plots
- LABELS: Standard axis labels with units
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Style Presets
# =============================================================================

STYLES: dict[str, dict[str, Any]] = {
    "default": {
        # Figure size and resolution
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        # Font sizes
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Line widths
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        # Grid
        "axes.grid": False,
        # Legend
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
    },
    "paper": {
        # Publication-ready: small figures, high resolution
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,
        # Smaller fonts for dense figures
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # Thinner lines for high DPI
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        # Minimal styling
        "axes.grid": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "none",
        # Use serif font for publications
        "font.family": "serif",
    },
    "presentation": {
        # Large figures for slides
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        # Large fonts for readability
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        # Thicker lines for visibility
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.5,
        # Grid for easier reading
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        # Bold legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.5",
    },
}


# =============================================================================
# Color Definitions
# =============================================================================

COLORS: dict[str, str] = {
    # Transit data points
    "transit": "#1f77b4",  # Blue - in-transit data
    "out_of_transit": "#7f7f7f",  # Gray - out-of-transit baseline
    # Odd/even comparison
    "odd": "#d62728",  # Red - odd transits
    "even": "#2ca02c",  # Green - even transits
    # Model and fit
    "model": "#ff7f0e",  # Orange - fitted model
    "residual": "#9467bd",  # Purple - residuals
    # Secondary eclipse
    "secondary": "#8c564b",  # Brown - secondary eclipse
    # Confidence intervals
    "ci_1sigma": "#aec7e8",  # Light blue - 1-sigma region
    "ci_2sigma": "#c7c7c7",  # Light gray - 2-sigma region
    # Threshold lines
    "threshold": "#e377c2",  # Pink - threshold markers
    "reference": "#17becf",  # Cyan - reference lines
    # Pixel/centroid
    "target": "#d62728",  # Red - target position
    "in_transit_centroid": "#1f77b4",  # Blue - in-transit centroid
    "out_transit_centroid": "#2ca02c",  # Green - out-of-transit centroid
    "shift_vector": "#ff7f0e",  # Orange - centroid shift vector
    # Anomaly/outlier
    "outlier": "#e41a1c",  # Bright red - outlier points
    "flagged": "#ffff33",  # Yellow - flagged data
}


# =============================================================================
# Colormaps
# =============================================================================

COLORMAPS: dict[str, str] = {
    # Flux images (diverging around zero is useful for difference images)
    "flux": "viridis",
    "difference": "RdBu_r",  # Red-Blue reversed: red=positive, blue=negative
    "residual": "coolwarm",
    # Centroid/position images
    "centroid": "viridis",
    # SNR/significance
    "snr": "plasma",
    "significance": "magma",
    # Aperture masks (binary)
    "mask": "Greys",
    # Phase diagrams
    "phase": "twilight",
}


# =============================================================================
# Axis Labels
# =============================================================================

LABELS: dict[str, str] = {
    # Time
    "time_btjd": "Time (BTJD)",
    "time_days": "Time (days)",
    "phase": "Orbital Phase",
    "phase_folded": "Phase (folded)",
    # Flux
    "flux": "Relative Flux",
    "flux_normalized": "Normalized Flux",
    "flux_ppm": "Flux (ppm from median)",
    "flux_electrons": r"Flux (e$^-$/s)",
    # Depth
    "depth_ppm": "Transit Depth (ppm)",
    "depth_percent": "Transit Depth (%)",
    # Position
    "pixel_x": "Column (pixels)",
    "pixel_y": "Row (pixels)",
    "offset_arcsec": "Offset (arcsec)",
    "ra": "RA (deg)",
    "dec": "Dec (deg)",
    # Duration
    "duration_hours": "Duration (hours)",
    "duration_days": "Duration (days)",
    # Period
    "period_days": "Period (days)",
    # Epoch
    "epoch": "Epoch Number",
    "transit_number": "Transit Number",
    # Significance
    "sigma": r"Significance ($\sigma$)",
    "snr": "SNR",
    # Sector
    "sector": "TESS Sector",
}
