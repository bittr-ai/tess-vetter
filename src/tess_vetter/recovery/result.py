"""Domain models for transit recovery.

This module provides dataclasses for representing transit recovery results:
- StackedTransit: Stacked transit light curve data
- TrapezoidFit: Trapezoid model fit parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class StackedTransit:
    """Stacked transit light curve.

    Contains phase-folded and binned transit data for improved SNR analysis.

    Attributes:
        phase: Phase bin centers (0.0 to 1.0, transit at 0.5)
        flux: Binned flux values (normalized)
        flux_err: Propagated flux uncertainties
        n_points_per_bin: Number of data points contributing to each bin
        n_transits: Number of distinct transits stacked
    """

    phase: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_err: NDArray[np.float64]
    n_points_per_bin: NDArray[np.int32]
    n_transits: int


@dataclass(frozen=True)
class TrapezoidFit:
    """Trapezoid model fit result.

    Contains parameters from fitting a trapezoid transit model to stacked data.

    Attributes:
        depth: Fractional transit depth (e.g., 0.004 for 0.4%)
        depth_err: Uncertainty on depth
        duration_phase: Total transit duration in phase units
        ingress_ratio: Ingress/egress as fraction of duration (0.1 to 0.4)
        chi2: Chi-squared statistic of the fit
        reduced_chi2: Reduced chi-squared (chi2 / dof)
        converged: Whether the fit converged successfully
    """

    depth: float
    depth_err: float
    duration_phase: float
    ingress_ratio: float
    chi2: float
    reduced_chi2: float
    converged: bool
