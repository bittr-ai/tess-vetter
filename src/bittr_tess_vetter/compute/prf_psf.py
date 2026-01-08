"""PRF/PSF model interface and implementations.

This module provides a layered PRF/PSF model system for TESS pixel-level analysis:
- Protocol (interface) for PRF models
- Default parametric PSF implementation (always available)
- Factory function to select backends

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve at module level
- ONLY numpy and scipy dependencies

Backend strategy (per D3 decision):
1. Default: parametric PSF (elliptical Gaussian) - always available
2. Optional: instrument PRF grids (future-proof hook)
3. Optional: third-party (lightkurve/oktopus) when installed
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from .prf_schemas import BackgroundParams, PRFFitResult, PRFParams

if TYPE_CHECKING:
    pass

__all__ = [
    "PRFModel",
    "ParametricPSF",
    "get_prf_model",
    "AVAILABLE_BACKENDS",
]


# =============================================================================
# PRF Model Protocol (Interface)
# =============================================================================


@runtime_checkable
class PRFModel(Protocol):
    """Protocol for PRF/PSF models.

    All PRF model implementations must satisfy this interface.
    The protocol defines methods for evaluating the PRF on pixel grids
    and fitting to observed images.
    """

    @property
    def backend_name(self) -> str:
        """Return the name of the backend (for provenance tracking)."""
        ...

    @property
    def params(self) -> PRFParams:
        """Return current PRF parameters."""
        ...

    def evaluate(
        self,
        center_row: float,
        center_col: float,
        shape: tuple[int, int],
        *,
        background: tuple[float, float, float] | None = None,
        jitter_sigma: float | None = None,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Evaluate PRF weights on a pixel grid.

        Parameters
        ----------
        center_row : float
            Row coordinate of the PRF center (0-indexed, can be fractional).
        center_col : float
            Column coordinate of the PRF center (0-indexed, can be fractional).
        shape : tuple[int, int]
            Shape of the output array as (n_rows, n_cols).
        background : tuple[float, float, float] | None, optional
            Background gradient as (b0, bx, by). If None, no background is added.
        jitter_sigma : float | None, optional
            Isotropic jitter to add in quadrature (pixels).
        normalize : bool, optional
            If True, normalize PRF weights to sum to 1.0. Default True.

        Returns
        -------
        NDArray[np.float64]
            2D array of shape `shape` containing the PRF model.
            If normalize=True, the PRF portion (excluding background) sums to 1.0.
        """
        ...

    def evaluate_at_positions(
        self,
        positions: list[tuple[float, float]],
        shape: tuple[int, int],
        *,
        jitter_sigma: float | None = None,
        normalize: bool = True,
    ) -> list[NDArray[np.float64]]:
        """Evaluate PRF for multiple source positions.

        Parameters
        ----------
        positions : list[tuple[float, float]]
            List of (row, col) positions for each source.
        shape : tuple[int, int]
            Shape of the output arrays as (n_rows, n_cols).
        jitter_sigma : float | None, optional
            Isotropic jitter to add in quadrature (pixels).
        normalize : bool, optional
            If True, normalize each PRF to sum to 1.0. Default True.

        Returns
        -------
        list[NDArray[np.float64]]
            List of 2D arrays, one per position.
        """
        ...

    def fit_to_image(
        self,
        image: NDArray[np.float64],
        *,
        mask: NDArray[np.bool_] | None = None,
        uncertainty: NDArray[np.float64] | None = None,
        initial_center: tuple[float, float] | None = None,
        fit_background: bool = True,
        fit_shape: bool = True,
    ) -> PRFFitResult:
        """Fit the PRF model to an observed image.

        Parameters
        ----------
        image : NDArray[np.float64]
            2D array of observed pixel values.
        mask : NDArray[np.bool_] | None, optional
            Boolean mask where True indicates pixels to include in fit.
            If None, all pixels are included.
        uncertainty : NDArray[np.float64] | None, optional
            1-sigma uncertainties for each pixel. Used for chi-squared.
        initial_center : tuple[float, float] | None, optional
            Initial guess for (row, col) center. If None, uses image centroid.
        fit_background : bool, optional
            Whether to fit background gradient. Default True.
        fit_shape : bool, optional
            Whether to fit PSF shape (sigma, theta). Default True.

        Returns
        -------
        PRFFitResult
            Fitted parameters, center, and fit quality metrics.
        """
        ...


# =============================================================================
# Parametric PSF Implementation
# =============================================================================


class ParametricPSF:
    """Parametric PSF model using elliptical Gaussian.

    This is the default backend (always available, no external dependencies).
    Implements an elliptical Gaussian with optional rotation and background gradient.

    Parameters
    ----------
    params : PRFParams, optional
        Initial PRF parameters. Default is isotropic with sigma=1.5.
    """

    def __init__(self, params: PRFParams | None = None) -> None:
        """Initialize parametric PSF model."""
        self._params = params if params is not None else PRFParams()

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "parametric"

    @property
    def params(self) -> PRFParams:
        """Return current PRF parameters."""
        return self._params

    def evaluate(
        self,
        center_row: float,
        center_col: float,
        shape: tuple[int, int],
        *,
        background: tuple[float, float, float] | None = None,
        jitter_sigma: float | None = None,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Evaluate elliptical Gaussian PRF on a pixel grid.

        Parameters
        ----------
        center_row : float
            Row coordinate of the PRF center (0-indexed).
        center_col : float
            Column coordinate of the PRF center (0-indexed).
        shape : tuple[int, int]
            Shape of output array as (n_rows, n_cols).
        background : tuple[float, float, float] | None, optional
            Background gradient as (b0, bx, by).
        jitter_sigma : float | None, optional
            Isotropic jitter to add in quadrature.
        normalize : bool, optional
            If True, normalize PRF to sum to 1.0.

        Returns
        -------
        NDArray[np.float64]
            2D PRF model array.
        """
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"shape dimensions must be positive, got {shape}")

        # Get effective parameters (with jitter if specified)
        params = self._params
        if jitter_sigma is not None and jitter_sigma > 0:
            params = params.with_jitter(jitter_sigma)

        n_rows, n_cols = shape

        # Create coordinate grids
        row_coords, col_coords = np.mgrid[0:n_rows, 0:n_cols]
        row_coords = row_coords.astype(np.float64)
        col_coords = col_coords.astype(np.float64)

        # Compute rotated coordinates relative to center
        drow = row_coords - center_row
        dcol = col_coords - center_col

        cos_theta = np.cos(params.theta)
        sin_theta = np.sin(params.theta)

        # Rotate to principal axes
        x_rot = cos_theta * dcol + sin_theta * drow
        y_rot = -sin_theta * dcol + cos_theta * drow

        # Elliptical Gaussian
        exponent = (x_rot / params.sigma_col) ** 2 + (y_rot / params.sigma_row) ** 2
        prf = params.amplitude * np.exp(-0.5 * exponent)

        # Normalize if requested
        if normalize:
            prf_sum = prf.sum()
            if prf_sum > 0:
                prf = prf / prf_sum

        # Add background if specified
        if background is not None:
            bg_params = BackgroundParams.from_tuple(background)
            bg = bg_params.evaluate(row_coords, col_coords, center_row, center_col)
            prf = prf + bg

        result: NDArray[np.float64] = prf.astype(np.float64)
        return result

    def evaluate_at_positions(
        self,
        positions: list[tuple[float, float]],
        shape: tuple[int, int],
        *,
        jitter_sigma: float | None = None,
        normalize: bool = True,
    ) -> list[NDArray[np.float64]]:
        """Evaluate PRF for multiple source positions.

        Parameters
        ----------
        positions : list[tuple[float, float]]
            List of (row, col) positions.
        shape : tuple[int, int]
            Shape of output arrays.
        jitter_sigma : float | None, optional
            Isotropic jitter to add.
        normalize : bool, optional
            If True, normalize each PRF.

        Returns
        -------
        list[NDArray[np.float64]]
            PRF model for each position.
        """
        return [
            self.evaluate(
                center_row=row,
                center_col=col,
                shape=shape,
                jitter_sigma=jitter_sigma,
                normalize=normalize,
            )
            for row, col in positions
        ]

    def fit_to_image(
        self,
        image: NDArray[np.float64],
        *,
        mask: NDArray[np.bool_] | None = None,
        uncertainty: NDArray[np.float64] | None = None,
        initial_center: tuple[float, float] | None = None,
        fit_background: bool = True,
        fit_shape: bool = True,
    ) -> PRFFitResult:
        """Fit PRF model to an observed image using least-squares.

        Uses scipy.optimize.minimize with Nelder-Mead for robust fitting.

        Parameters
        ----------
        image : NDArray[np.float64]
            2D image to fit.
        mask : NDArray[np.bool_] | None, optional
            Pixels to include (True = include).
        uncertainty : NDArray[np.float64] | None, optional
            Per-pixel uncertainties.
        initial_center : tuple[float, float] | None, optional
            Initial (row, col) guess.
        fit_background : bool, optional
            Whether to fit background.
        fit_shape : bool, optional
            Whether to fit PSF shape.

        Returns
        -------
        PRFFitResult
            Fitted parameters and quality metrics.
        """
        shape = image.shape
        if mask is None:
            mask = np.ones(shape, dtype=bool)

        # Initial center: use centroid if not provided
        if initial_center is None:
            row_coords, col_coords = np.mgrid[0 : shape[0], 0 : shape[1]]
            flux_sum = np.sum(image[mask])
            if flux_sum > 0:
                center_row = float(np.sum(row_coords[mask] * image[mask]) / flux_sum)
                center_col = float(np.sum(col_coords[mask] * image[mask]) / flux_sum)
            else:
                center_row = shape[0] / 2.0
                center_col = shape[1] / 2.0
        else:
            center_row, center_col = initial_center

        # Build parameter vector and bounds
        # Order: [center_row, center_col, amplitude, sigma_row, sigma_col, theta, b0, bx, by]
        params_init = self._params

        x0: list[float] = [center_row, center_col, float(np.max(image[mask]))]

        if fit_shape:
            x0.extend([params_init.sigma_row, params_init.sigma_col, params_init.theta])
        if fit_background:
            x0.extend(
                [params_init.background.b0, params_init.background.bx, params_init.background.by]
            )

        def objective(x: NDArray[np.float64]) -> float:
            """Compute sum of squared residuals."""
            c_row, c_col, amp = x[0], x[1], x[2]
            idx = 3

            if fit_shape:
                s_row, s_col, th = x[idx], x[idx + 1], x[idx + 2]
                idx += 3
            else:
                s_row = params_init.sigma_row
                s_col = params_init.sigma_col
                th = params_init.theta

            if fit_background:
                b0, bx, by = x[idx], x[idx + 1], x[idx + 2]
            else:
                b0, bx, by = 0.0, 0.0, 0.0

            # Enforce positive sigmas
            if s_row <= 0.1 or s_col <= 0.1 or amp <= 0:
                return 1e30

            # Build model
            temp_params = PRFParams(
                sigma_row=s_row,
                sigma_col=s_col,
                theta=th,
                amplitude=amp,
                background=BackgroundParams(b0=b0, bx=bx, by=by),
            )
            temp_psf = ParametricPSF(temp_params)
            model = temp_psf.evaluate(
                c_row,
                c_col,
                shape,
                background=(b0, bx, by) if fit_background else None,
                normalize=False,
            )

            residuals = image - model
            if uncertainty is not None:
                residuals = residuals / uncertainty

            return float(np.sum(residuals[mask] ** 2))

        # Run optimization
        result = optimize.minimize(
            objective,
            np.array(x0),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8},
        )

        # Extract fitted parameters
        x_fit = result.x
        c_row_fit, c_col_fit, amp_fit = x_fit[0], x_fit[1], x_fit[2]
        idx = 3

        if fit_shape:
            s_row_fit, s_col_fit, th_fit = x_fit[idx], x_fit[idx + 1], x_fit[idx + 2]
            idx += 3
        else:
            s_row_fit = params_init.sigma_row
            s_col_fit = params_init.sigma_col
            th_fit = params_init.theta

        if fit_background:
            b0_fit, bx_fit, by_fit = x_fit[idx], x_fit[idx + 1], x_fit[idx + 2]
        else:
            b0_fit, bx_fit, by_fit = 0.0, 0.0, 0.0

        # Build fitted model and compute residuals
        fitted_params = PRFParams(
            sigma_row=max(0.1, s_row_fit),
            sigma_col=max(0.1, s_col_fit),
            theta=th_fit,
            amplitude=max(0.01, amp_fit),
            background=BackgroundParams(b0=b0_fit, bx=bx_fit, by=by_fit),
        )

        fitted_psf = ParametricPSF(fitted_params)
        fitted_model = fitted_psf.evaluate(
            c_row_fit,
            c_col_fit,
            shape,
            background=(b0_fit, bx_fit, by_fit) if fit_background else None,
            normalize=False,
        )

        residuals = image - fitted_model
        residual_rms = float(np.sqrt(np.mean(residuals[mask] ** 2)))

        # Compute chi-squared if uncertainty provided
        chi_squared = None
        if uncertainty is not None:
            chi_squared = float(np.sum((residuals[mask] / uncertainty[mask]) ** 2))

        return PRFFitResult(
            params=fitted_params,
            center_row=float(c_row_fit),
            center_col=float(c_col_fit),
            residual_rms=residual_rms,
            converged=result.success,
            n_iterations=int(result.nit) if hasattr(result, "nit") else 0,
            chi_squared=chi_squared,
            covariance=None,  # Nelder-Mead doesn't provide covariance
        )


# =============================================================================
# Backend Registry and Factory
# =============================================================================

AVAILABLE_BACKENDS: dict[str, bool] = {
    "parametric": True,  # Always available
    "instrument": False,  # Future: load from calibration grids
    "lightkurve": False,  # Future: use lightkurve/oktopus
}


def _check_lightkurve_available() -> bool:
    """Check if lightkurve is available for PRF modeling."""
    try:
        import lightkurve  # noqa: F401

        return True
    except ImportError:
        return False


def get_prf_model(
    backend: str = "parametric",
    *,
    params: PRFParams | None = None,
    camera: int | None = None,
    ccd: int | None = None,
    sector: int | None = None,
) -> PRFModel:
    """Get a PRF model instance.

    Factory function to create PRF models from different backends.

    Parameters
    ----------
    backend : str, optional
        Backend to use. Options:
        - "parametric": Elliptical Gaussian (always available, default)
        - "instrument": Load from calibration grids (raises if not available)
        - "lightkurve": Use lightkurve/oktopus (raises if not installed)
    params : PRFParams | None, optional
        Initial PRF parameters (for parametric backend).
    camera : int | None, optional
        TESS camera number (1-4). Used by instrument backend.
    ccd : int | None, optional
        TESS CCD number (1-4). Used by instrument backend.
    sector : int | None, optional
        TESS sector number. Used by instrument backend.

    Returns
    -------
    PRFModel
        PRF model instance implementing the PRFModel protocol.

    Raises
    ------
    ValueError
        If the requested backend is not available.
    """
    backend = backend.lower()

    if backend == "parametric":
        return ParametricPSF(params=params)

    elif backend == "instrument":
        # Future hook for instrument PRF grids
        raise ValueError(
            "Instrument PRF backend not yet available. "
            "Use 'parametric' backend or wait for calibration grid support."
        )

    elif backend == "lightkurve":
        if not _check_lightkurve_available():
            raise ValueError(
                "lightkurve backend requires lightkurve package. "
                "Install with: pip install lightkurve"
            )
        # Future hook for lightkurve PRF
        raise ValueError("lightkurve PRF backend not yet implemented. Use 'parametric' backend.")

    else:
        available = [k for k, v in AVAILABLE_BACKENDS.items() if v]
        raise ValueError(f"Unknown backend '{backend}'. Available backends: {available}")
