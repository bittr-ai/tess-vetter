"""PRF/PSF parameter schemas and data structures.

This module defines dataclasses for PRF/PSF model parameters, fitting results,
and serialization helpers for JSON round-tripping.

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve
- ONLY numpy and standard library dependencies
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "PRFParams",
    "PRFFitResult",
    "BackgroundParams",
    "prf_params_to_dict",
    "prf_params_from_dict",
    "fit_result_to_dict",
    "fit_result_from_dict",
]


@dataclass(frozen=True)
class BackgroundParams:
    """Background gradient parameters.

    The background model is: b0 + bx * (col - col_center) + by * (row - row_center)

    Attributes
    ----------
    b0 : float
        Constant background level.
    bx : float
        Background gradient in the column direction.
    by : float
        Background gradient in the row direction.
    """

    b0: float = 0.0
    bx: float = 0.0
    by: float = 0.0

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to (b0, bx, by) tuple."""
        return (self.b0, self.bx, self.by)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float]) -> BackgroundParams:
        """Create from (b0, bx, by) tuple."""
        return cls(b0=t[0], bx=t[1], by=t[2])

    def evaluate(
        self,
        row_coords: np.ndarray,
        col_coords: np.ndarray,
        row_center: float,
        col_center: float,
    ) -> np.ndarray:
        """Evaluate background model on a coordinate grid.

        Parameters
        ----------
        row_coords : np.ndarray
            Row coordinates (can be 1D or 2D grid).
        col_coords : np.ndarray
            Column coordinates (same shape as row_coords).
        row_center : float
            Row coordinate of the PRF center.
        col_center : float
            Column coordinate of the PRF center.

        Returns
        -------
        np.ndarray
            Background values at each coordinate (same shape as inputs).
        """
        return self.b0 + self.bx * (col_coords - col_center) + self.by * (row_coords - row_center)


@dataclass(frozen=True)
class PRFParams:
    """Parameters for a PRF/PSF model.

    Represents an elliptical Gaussian PSF with optional rotation and background.

    Attributes
    ----------
    sigma_row : float
        Standard deviation along the row axis (pixels).
    sigma_col : float
        Standard deviation along the column axis (pixels).
    theta : float
        Rotation angle in radians (counter-clockwise from col axis).
    amplitude : float
        Peak amplitude of the PSF (before normalization).
    background : BackgroundParams
        Background gradient parameters.
    """

    sigma_row: float = 1.5
    sigma_col: float = 1.5
    theta: float = 0.0
    amplitude: float = 1.0
    background: BackgroundParams = field(default_factory=BackgroundParams)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.sigma_row <= 0:
            raise ValueError(f"sigma_row must be positive, got {self.sigma_row}")
        if self.sigma_col <= 0:
            raise ValueError(f"sigma_col must be positive, got {self.sigma_col}")
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")

    @property
    def is_isotropic(self) -> bool:
        """Check if the PSF is isotropic (circular)."""
        return bool(np.isclose(self.sigma_row, self.sigma_col) and np.isclose(self.theta, 0.0))

    @property
    def effective_sigma(self) -> float:
        """Geometric mean of sigma_row and sigma_col."""
        return float(np.sqrt(self.sigma_row * self.sigma_col))

    def with_jitter(self, jitter_sigma: float) -> PRFParams:
        """Return new params with added isotropic jitter.

        Jitter is added in quadrature: sigma_new = sqrt(sigma^2 + jitter^2)

        Parameters
        ----------
        jitter_sigma : float
            Isotropic jitter to add (in pixels).

        Returns
        -------
        PRFParams
            New PRFParams with jitter added.
        """
        if jitter_sigma <= 0:
            return self
        new_sigma_row = float(np.sqrt(self.sigma_row**2 + jitter_sigma**2))
        new_sigma_col = float(np.sqrt(self.sigma_col**2 + jitter_sigma**2))
        return PRFParams(
            sigma_row=new_sigma_row,
            sigma_col=new_sigma_col,
            theta=self.theta,
            amplitude=self.amplitude,
            background=self.background,
        )


@dataclass(frozen=True)
class PRFFitResult:
    """Result of fitting a PRF/PSF model to image data.

    Attributes
    ----------
    params : PRFParams
        The fitted PRF parameters.
    center_row : float
        Fitted row center of the PSF.
    center_col : float
        Fitted column center of the PSF.
    residual_rms : float
        RMS of fit residuals (normalized by image flux).
    converged : bool
        Whether the optimization converged.
    n_iterations : int
        Number of iterations used by the optimizer.
    chi_squared : float | None
        Chi-squared statistic if uncertainty was provided.
    covariance : np.ndarray | None
        Covariance matrix of fitted parameters (if available).
    """

    params: PRFParams
    center_row: float
    center_col: float
    residual_rms: float
    converged: bool
    n_iterations: int
    chi_squared: float | None = None
    covariance: np.ndarray | None = None

    @property
    def center(self) -> tuple[float, float]:
        """Return (row, col) center as tuple."""
        return (self.center_row, self.center_col)


# =============================================================================
# JSON Serialization Helpers
# =============================================================================


def prf_params_to_dict(params: PRFParams) -> dict[str, Any]:
    """Serialize PRFParams to a JSON-compatible dictionary.

    Parameters
    ----------
    params : PRFParams
        The PRF parameters to serialize.

    Returns
    -------
    dict[str, Any]
        Dictionary representation suitable for JSON serialization.
    """
    return {
        "sigma_row": params.sigma_row,
        "sigma_col": params.sigma_col,
        "theta": params.theta,
        "amplitude": params.amplitude,
        "background": {
            "b0": params.background.b0,
            "bx": params.background.bx,
            "by": params.background.by,
        },
    }


def prf_params_from_dict(d: dict[str, Any]) -> PRFParams:
    """Deserialize PRFParams from a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary containing PRF parameter values.

    Returns
    -------
    PRFParams
        Reconstructed PRFParams object.
    """
    bg_dict = d.get("background", {})
    background = BackgroundParams(
        b0=float(bg_dict.get("b0", 0.0)),
        bx=float(bg_dict.get("bx", 0.0)),
        by=float(bg_dict.get("by", 0.0)),
    )
    return PRFParams(
        sigma_row=float(d["sigma_row"]),
        sigma_col=float(d["sigma_col"]),
        theta=float(d.get("theta", 0.0)),
        amplitude=float(d.get("amplitude", 1.0)),
        background=background,
    )


def fit_result_to_dict(result: PRFFitResult) -> dict[str, Any]:
    """Serialize PRFFitResult to a JSON-compatible dictionary.

    Parameters
    ----------
    result : PRFFitResult
        The fit result to serialize.

    Returns
    -------
    dict[str, Any]
        Dictionary representation suitable for JSON serialization.
    """
    d: dict[str, Any] = {
        "params": prf_params_to_dict(result.params),
        "center_row": result.center_row,
        "center_col": result.center_col,
        "residual_rms": result.residual_rms,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
    }
    if result.chi_squared is not None:
        d["chi_squared"] = result.chi_squared
    if result.covariance is not None:
        d["covariance"] = result.covariance.tolist()
    return d


def fit_result_from_dict(d: dict[str, Any]) -> PRFFitResult:
    """Deserialize PRFFitResult from a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary containing fit result values.

    Returns
    -------
    PRFFitResult
        Reconstructed PRFFitResult object.
    """
    covariance = None
    if "covariance" in d and d["covariance"] is not None:
        covariance = np.array(d["covariance"], dtype=np.float64)

    return PRFFitResult(
        params=prf_params_from_dict(d["params"]),
        center_row=float(d["center_row"]),
        center_col=float(d["center_col"]),
        residual_rms=float(d["residual_rms"]),
        converged=bool(d["converged"]),
        n_iterations=int(d["n_iterations"]),
        chi_squared=float(d["chi_squared"]) if d.get("chi_squared") is not None else None,
        covariance=covariance,
    )


def prf_params_to_json(params: PRFParams) -> str:
    """Serialize PRFParams to a JSON string.

    Parameters
    ----------
    params : PRFParams
        The PRF parameters to serialize.

    Returns
    -------
    str
        JSON string representation.
    """
    return json.dumps(prf_params_to_dict(params))


def prf_params_from_json(s: str) -> PRFParams:
    """Deserialize PRFParams from a JSON string.

    Parameters
    ----------
    s : str
        JSON string containing PRF parameter values.

    Returns
    -------
    PRFParams
        Reconstructed PRFParams object.
    """
    return prf_params_from_dict(json.loads(s))
