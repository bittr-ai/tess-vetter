"""Exovetter-based vetting checks (Modshift, SWEET).

This module integrates the exovetter library's Modshift and SWEET tests
into a validation pipeline.

Modshift: Detects eccentric eclipsing binaries where the secondary eclipse
occurs at an unexpected phase (not 0.5). This catches EBs that would be
missed by the standard secondary eclipse search at phase 0.5.

SWEET: Sine Wave Evaluation for Ephemeris Transits - Detects stellar
variability (e.g., rotation) that could masquerade as planetary transits.
If a sine wave at the transit period fits the data well, the signal may
be stellar variability rather than a transit.

References:
- Coughlin et al. (2014) - Modshift technique for EB detection
- Thompson et al. (2018) - Kepler DR25 vetting methodology
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult
from bittr_tess_vetter.validation.base import CheckConfig, VetterCheck

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters

logger = logging.getLogger(__name__)


# =============================================================================
# V11: Modshift Check
# =============================================================================


class ModshiftCheck(VetterCheck):
    """V11: Modshift test for secondary eclipse detection at arbitrary phase.

    Astronomical Significance:
    --------------------------
    The standard secondary eclipse search looks at phase 0.5, which is correct
    for circular orbits. However, eccentric orbits can have secondary eclipses
    at phases significantly different from 0.5.

    The Modshift test (Coughlin et al. 2014) searches for the strongest
    secondary signal at any phase by computing:
    - Primary signal strength (pri): The main transit/eclipse signal
    - Secondary signal strength (sec): Any secondary signal
    - Tertiary signal strength (ter): Third strongest signal
    - Positive signal strength (pos): Largest positive (brightening) event
    - Fred: Red noise level (convolution std / lightcurve std)

    Key metrics for EB detection:
    - If sec > false_alarm_threshold: Significant secondary eclipse detected
    - If sec / pri > 0.1: Secondary is substantial fraction of primary
    - If ter is significant: Multiple eclipse-like events suggest EB
    - High Fred indicates correlated noise that could produce false signals

    Pass Criteria:
    - secondary/primary ratio < threshold (default 0.5)
    - secondary < false_alarm_threshold OR no significant secondary

    Confidence Calculation:
    - High confidence (0.95) when sec/pri << threshold
    - Lower confidence near threshold or with high Fred (red noise)
    """

    id = "V11"
    name = "modshift"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default Modshift configuration."""
        return CheckConfig(
            enabled=True,
            threshold=0.5,  # Max secondary/primary ratio before flagging as EB
            additional={
                "fred_warning_threshold": 2.0,  # Fred > 2 indicates significant red noise
                "tertiary_warning_threshold": 0.3,  # If ter/pri > this, warn
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Run Modshift test to detect secondary eclipses at arbitrary phase.

        Args:
            candidate: Transit candidate with period, t0, duration, depth.
            lightcurve: Light curve data (required).
            stellar: Not used for this check.

        Returns:
            VetterCheckResult with Modshift metrics and pass/fail status.
        """
        if lightcurve is None:
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.30,
                details={
                    "status": "skipped",
                    "reason": "Light curve data required for Modshift test",
                },
            )

        threshold = self.config.threshold or 0.5
        additional = self.config.additional or {}
        fred_warn = additional.get("fred_warning_threshold", 2.0)
        ter_warn = additional.get("tertiary_warning_threshold", 0.3)

        try:
            # Import exovetter components
            from exovetter.tce import Tce
            from exovetter.vetters import ModShift

            # Create a lightkurve-like object for exovetter
            # exovetter expects a lightkurve object with time, flux attributes
            lk_obj = _create_lightkurve_like(lightcurve)

            # Create TCE (Threshold Crossing Event) object
            import astropy.units as u
            from exovetter import const as exo_const

            tce = Tce(
                period=candidate.period * u.day,
                epoch=candidate.t0 * u.day,
                epoch_offset=exo_const.btjd,
                depth=candidate.depth * 1e6 * exo_const.ppm,  # Convert to ppm
                duration=candidate.duration_hours * u.hour,
            )

            # Run Modshift
            vetter = ModShift(lc_name="flux")
            metrics = vetter.run(tce, lk_obj, plot=False)

        except ImportError as e:
            logger.warning(f"Exovetter import failed: {e}")
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.20,
                details={
                    "status": "error",
                    "reason": f"Exovetter import failed: {e}",
                },
            )
        except Exception as e:
            logger.warning(f"Modshift test failed: {e}")
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.20,
                details={
                    "status": "error",
                    "reason": f"Modshift test failed: {e}",
                },
            )

        # Extract metrics
        pri = float(metrics.get("pri", 0))
        sec = float(metrics.get("sec", 0))
        ter = float(metrics.get("ter", 0))
        pos = float(metrics.get("pos", 0))
        fred = float(metrics.get("Fred", 1.0))
        fa_thresh = float(metrics.get("false_alarm_threshold", 0))

        # Calculate ratios
        sec_pri_ratio = sec / pri if pri > 0 else 0.0
        ter_pri_ratio = ter / pri if pri > 0 else 0.0

        # Determine pass/fail
        # Fail if secondary is significant fraction of primary
        significant_secondary = sec_pri_ratio > threshold
        # Also check if secondary exceeds false alarm threshold
        sec_above_fa = sec > fa_thresh and sec > 0

        passed = not (significant_secondary and sec_above_fa)

        # Calculate confidence
        if not passed:
            # Failed - high confidence it's an EB
            if sec_pri_ratio > 0.8:
                confidence = 0.98
            elif sec_pri_ratio > threshold:
                confidence = 0.90
            else:
                confidence = 0.85
        else:
            # Passed
            if fred > fred_warn:
                # High red noise - lower confidence
                confidence = 0.70
            elif ter_pri_ratio > ter_warn:
                # Tertiary signal present - moderate concern
                confidence = 0.80
            elif sec_pri_ratio < threshold / 2:
                # Well below threshold
                confidence = 0.95
            else:
                confidence = 0.85

        # Build interpretation
        interpretation = self._build_interpretation(
            passed=passed,
            sec_pri_ratio=sec_pri_ratio,
            ter_pri_ratio=ter_pri_ratio,
            fred=fred,
            sec_above_fa=sec_above_fa,
            threshold=threshold,
            fred_warn=fred_warn,
            ter_warn=ter_warn,
        )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=round(confidence, 3),
            details={
                "primary_signal": round(pri, 4),
                "secondary_signal": round(sec, 4),
                "tertiary_signal": round(ter, 4),
                "positive_signal": round(pos, 4),
                "fred": round(fred, 4),
                "false_alarm_threshold": round(fa_thresh, 4),
                "secondary_primary_ratio": round(sec_pri_ratio, 4),
                "tertiary_primary_ratio": round(ter_pri_ratio, 4),
                "threshold": threshold,
                "significant_secondary": significant_secondary,
                "secondary_above_fa": sec_above_fa,
                "interpretation": interpretation,
            },
        )

    def _build_interpretation(
        self,
        passed: bool,
        sec_pri_ratio: float,
        ter_pri_ratio: float,
        fred: float,
        sec_above_fa: bool,
        threshold: float,
        fred_warn: float,
        ter_warn: float,
    ) -> str:
        """Build human-readable interpretation of Modshift results."""
        parts = []

        if not passed:
            parts.append(
                f"Significant secondary eclipse detected (sec/pri = {sec_pri_ratio:.2f}, "
                f"threshold = {threshold}). This suggests an eclipsing binary system."
            )
        else:
            parts.append(
                f"No significant secondary eclipse (sec/pri = {sec_pri_ratio:.2f} < {threshold})."
            )

        if fred > fred_warn:
            parts.append(
                f"Warning: High red noise level (Fred = {fred:.2f}). "
                "Results may be affected by correlated noise."
            )

        if ter_pri_ratio > ter_warn and passed:
            parts.append(
                f"Note: Tertiary signal is {ter_pri_ratio * 100:.1f}% of primary. "
                "Multiple eclipse-like features detected."
            )

        return " ".join(parts)


# =============================================================================
# V12: SWEET Check
# =============================================================================


class SWEETCheck(VetterCheck):
    """V12: SWEET test for stellar variability masquerading as transits.

    Astronomical Significance:
    --------------------------
    The SWEET (Sine Wave Evaluation for Ephemeris Transits) test checks
    whether the observed signal could be explained by stellar variability
    (rotation, pulsation) rather than a planetary transit.

    Many stars exhibit sinusoidal brightness variations due to:
    - Stellar rotation with starspots
    - Ellipsoidal variations in close binaries
    - Pulsations (delta Scuti, RR Lyrae, etc.)

    If a sinusoidal signal at the transit period (or harmonics) has high
    amplitude relative to uncertainties, the "transit" may actually be
    stellar variability.

    The test fits sinusoids at:
    - Half the transit period (P/2) - checks for even harmonics
    - The transit period (P) - direct variability at transit period
    - Twice the transit period (2P) - subharmonic variability

    Key metric: amplitude-to-uncertainty ratio (amp_ratio)
    - If amp_ratio > threshold at P, signal may be stellar variability

    Pass Criteria:
    - amplitude-to-uncertainty ratio at period < threshold (default 3.0)

    Confidence Calculation:
    - High confidence (0.95) when amp_ratio << threshold
    - Lower confidence when amp_ratio approaches threshold
    """

    id = "V12"
    name = "sweet"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default SWEET configuration."""
        return CheckConfig(
            enabled=True,
            threshold=3.0,  # amplitude-to-uncertainty ratio threshold
            additional={
                "half_period_threshold": 4.0,  # Higher threshold for P/2
                "double_period_threshold": 4.0,  # Higher threshold for 2P
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Run SWEET test to detect stellar variability at transit period.

        Args:
            candidate: Transit candidate with period, t0, duration, depth.
            lightcurve: Light curve data (required).
            stellar: Not used for this check.

        Returns:
            VetterCheckResult with SWEET metrics and pass/fail status.
        """
        if lightcurve is None:
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.30,
                details={
                    "status": "skipped",
                    "reason": "Light curve data required for SWEET test",
                },
            )

        threshold = self.config.threshold or 3.0
        additional = self.config.additional or {}
        half_p_thresh = additional.get("half_period_threshold", 4.0)
        double_p_thresh = additional.get("double_period_threshold", 4.0)

        try:
            # Import exovetter components
            from exovetter.tce import Tce
            from exovetter.vetters import Sweet

            # Create a lightkurve-like object for exovetter
            lk_obj = _create_lightkurve_like(lightcurve)

            # Create TCE object
            import astropy.units as u
            from exovetter import const as exo_const

            tce = Tce(
                period=candidate.period * u.day,
                epoch=candidate.t0 * u.day,
                epoch_offset=exo_const.btjd,
                depth=candidate.depth * 1e6 * exo_const.ppm,
                duration=candidate.duration_hours * u.hour,
            )

            # Run SWEET
            vetter = Sweet(lc_name="flux", threshold_sigma=int(threshold))
            metrics = vetter.run(tce, lk_obj, plot=False)

        except ImportError as e:
            logger.warning(f"Exovetter import failed: {e}")
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.20,
                details={
                    "status": "error",
                    "reason": f"Exovetter import failed: {e}",
                },
            )
        except Exception as e:
            logger.warning(f"SWEET test failed: {e}")
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.20,
                details={
                    "status": "error",
                    "reason": f"SWEET test failed: {e}",
                },
            )

        # Extract metrics
        # amp is a dict with 'half_period', 'period', 'double_period'
        # Each contains (amplitude, uncertainty, ratio)
        amp = metrics.get("amp", {})
        msg = metrics.get("msg", "")

        # Parse amplitude results
        amp_results: dict[str, dict[str, float]] = {}
        for key in ["half_period", "period", "double_period"]:
            if key in amp:
                val = amp[key]
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    amp_results[key] = {
                        "amplitude": float(val[0]),
                        "uncertainty": float(val[1]),
                        "ratio": float(val[2]),
                    }
                else:
                    amp_results[key] = {
                        "amplitude": 0.0,
                        "uncertainty": 1.0,
                        "ratio": 0.0,
                    }

        # Get ratios for each period
        half_p_ratio = amp_results.get("half_period", {}).get("ratio", 0.0)
        period_ratio = amp_results.get("period", {}).get("ratio", 0.0)
        double_p_ratio = amp_results.get("double_period", {}).get("ratio", 0.0)

        # Determine pass/fail
        # Primary concern is the period itself
        fails_at_period = period_ratio > threshold
        fails_at_half = half_p_ratio > half_p_thresh
        fails_at_double = double_p_ratio > double_p_thresh

        # Fail if significant variability at the transit period
        passed = not fails_at_period

        # Calculate confidence
        if not passed:
            # Failed - high confidence it's stellar variability
            if period_ratio > threshold * 2:
                confidence = 0.95
            elif period_ratio > threshold:
                confidence = 0.85
            else:
                confidence = 0.80
        else:
            # Passed
            if fails_at_half or fails_at_double:
                # Variability at harmonics - moderate concern
                confidence = 0.75
            elif period_ratio < threshold / 2:
                # Well below threshold
                confidence = 0.95
            elif period_ratio < threshold:
                confidence = 0.85
            else:
                confidence = 0.80

        # Build interpretation
        interpretation = self._build_interpretation(
            passed=passed,
            period_ratio=period_ratio,
            half_p_ratio=half_p_ratio,
            double_p_ratio=double_p_ratio,
            threshold=threshold,
            half_p_thresh=half_p_thresh,
            double_p_thresh=double_p_thresh,
            msg=msg,
        )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=round(confidence, 3),
            details={
                "period_amplitude_ratio": round(period_ratio, 4),
                "half_period_amplitude_ratio": round(half_p_ratio, 4),
                "double_period_amplitude_ratio": round(double_p_ratio, 4),
                "threshold": threshold,
                "amplitude_details": amp_results,
                "fails_at_period": fails_at_period,
                "fails_at_half_period": fails_at_half,
                "fails_at_double_period": fails_at_double,
                "exovetter_message": msg,
                "interpretation": interpretation,
            },
        )

    def _build_interpretation(
        self,
        passed: bool,
        period_ratio: float,
        half_p_ratio: float,
        double_p_ratio: float,
        threshold: float,
        half_p_thresh: float,
        double_p_thresh: float,
        msg: str,
    ) -> str:
        """Build human-readable interpretation of SWEET results."""
        parts = []

        if not passed:
            parts.append(
                f"Significant sinusoidal signal detected at transit period "
                f"(amp/sigma = {period_ratio:.2f}, threshold = {threshold}). "
                "This signal may be stellar variability rather than a planet transit."
            )
        else:
            parts.append(
                f"No significant sinusoidal variability at transit period "
                f"(amp/sigma = {period_ratio:.2f} < {threshold})."
            )

        if half_p_ratio > half_p_thresh and passed:
            parts.append(
                f"Note: Significant signal at half-period (amp/sigma = {half_p_ratio:.2f}). "
                "May indicate even-harmonic variability."
            )

        if double_p_ratio > double_p_thresh and passed:
            parts.append(
                f"Note: Significant signal at double-period (amp/sigma = {double_p_ratio:.2f}). "
                "May indicate subharmonic variability."
            )

        if msg:
            parts.append(f"Exovetter note: {msg}")

        return " ".join(parts)


# =============================================================================
# Helper Functions
# =============================================================================


class _LightkurveLike:
    """Minimal lightkurve-like object for exovetter compatibility.

    Exovetter expects a lightkurve object with time and flux attributes.
    This class wraps our LightCurveData to provide that interface.
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err if flux_err is not None else np.ones_like(flux) * 0.001
        self.time_format = "btjd"  # TESS uses BTJD


def _create_lightkurve_like(lightcurve: LightCurveData) -> _LightkurveLike:
    """Create a lightkurve-like object from our LightCurveData.

    Args:
        lightcurve: Our domain LightCurveData object.

    Returns:
        A minimal object with time, flux, flux_err attributes.
    """
    # Get valid data only
    mask = lightcurve.valid_mask
    time = lightcurve.time[mask]
    flux = lightcurve.flux[mask]
    flux_err = lightcurve.flux_err[mask] if lightcurve.flux_err is not None else None

    return _LightkurveLike(time, flux, flux_err)


# =============================================================================
# Convenience Functions
# =============================================================================


def run_modshift_check(
    candidate: TransitCandidate,
    lightcurve: LightCurveData,
    config: CheckConfig | None = None,
) -> VetterCheckResult:
    """Run Modshift check on a transit candidate.

    Args:
        candidate: Transit candidate parameters.
        lightcurve: Light curve data.
        config: Optional configuration override.

    Returns:
        VetterCheckResult from Modshift test.
    """
    check = ModshiftCheck(config)
    return check.run(candidate, lightcurve)


def run_sweet_check(
    candidate: TransitCandidate,
    lightcurve: LightCurveData,
    config: CheckConfig | None = None,
) -> VetterCheckResult:
    """Run SWEET check on a transit candidate.

    Args:
        candidate: Transit candidate parameters.
        lightcurve: Light curve data.
        config: Optional configuration override.

    Returns:
        VetterCheckResult from SWEET test.
    """
    check = SWEETCheck(config)
    return check.run(candidate, lightcurve)


def run_exovetter_checks(
    candidate: TransitCandidate,
    lightcurve: LightCurveData,
    modshift_config: CheckConfig | None = None,
    sweet_config: CheckConfig | None = None,
) -> list[VetterCheckResult]:
    """Run all exovetter checks (Modshift and SWEET) on a candidate.

    Args:
        candidate: Transit candidate parameters.
        lightcurve: Light curve data.
        modshift_config: Optional Modshift configuration.
        sweet_config: Optional SWEET configuration.

    Returns:
        List of VetterCheckResults [Modshift, SWEET].
    """
    results = []
    results.append(run_modshift_check(candidate, lightcurve, modshift_config))
    results.append(run_sweet_check(candidate, lightcurve, sweet_config))
    return results


def get_exovetter_checks(
    config: dict[str, CheckConfig] | None = None,
) -> list[VetterCheck]:
    """Get all exovetter check instances.

    Args:
        config: Optional dict mapping check IDs to configs.

    Returns:
        List of [ModshiftCheck, SWEETCheck] instances.
    """
    config = config or {}
    return [
        ModshiftCheck(config.get("V11")),
        SWEETCheck(config.get("V12")),
    ]
