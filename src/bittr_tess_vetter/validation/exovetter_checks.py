"""Exovetter-based vetting checks (Modshift, SWEET).

This module integrates the exovetter library's Modshift and SWEET tests
into a validation pipeline.

V11 - Modshift: Detects eccentric eclipsing binaries where the secondary
eclipse occurs at an unexpected phase (not 0.5). This catches EBs that would
be missed by the standard secondary eclipse search at phase 0.5.

Algorithm: Phase-folds data, convolves with transit template, measures
primary/secondary/tertiary signal strengths and red noise (Fred).

Key Metrics:
- pri: Primary (transit) signal strength
- sec: Secondary eclipse signal at any phase
- ter: Tertiary signal (third strongest dip)
- pos: Positive (brightening) signal
- Fred: Red noise level = std(convolution) / std(lightcurve)
- false_alarm_threshold: 1-sigma noise floor

V12 - SWEET: Sine Wave Evaluation for Ephemeris Transits - Detects stellar
variability (e.g., rotation) that could masquerade as planetary transits.
If a sine wave at the transit period fits the data well, the signal may
be stellar variability rather than a transit.

References:
    [1] Thompson et al. 2018, ApJS 235, 38 (arXiv:1710.06758)
        Section 3.2.3: ModShift in DR25 Robovetter; Table 3 metric summary
        Section 3.2.4: SWEET test for stellar variability detection
    [2] Coughlin et al. 2016, ApJS 224, 12 (arXiv:1512.06149)
        Section 4.3: ModShift false positive identification in DR24
        Section 4.4: Original SWEET implementation methodology
    [3] Santerne et al. 2013, A&A 557, A139 (arXiv:1307.2003)
        Secondary eclipse phase offset for eccentric orbits
    [4] Twicken et al. 2018, PASP 130, 064502 (arXiv:1803.04526)
        DV pipeline architecture context
    [5] McQuillan et al. 2014, ApJS 211, 24 (arXiv:1402.5694)
        Stellar rotation periods establishing expected variability timescales
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult
from bittr_tess_vetter.validation.base import CheckConfig, VetterCheck

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters

logger = logging.getLogger(__name__)


# =============================================================================
# Fred Regime Classification
# =============================================================================


def _classify_fred_regime(fred: float) -> str:
    """Classify Fred value into reliability regime.

    Fred ("Fraction of Red noise") quantifies correlated noise impact.
    Higher values indicate more correlated noise.

    Args:
        fred: Fred value from ModShift

    Returns:
        Regime string: "low", "standard", "high", or "critical"
    """
    if fred < 1.5:
        return "low"
    elif fred < 2.5:
        return "standard"
    elif fred < 3.5:
        return "high"
    else:
        return "critical"


def _is_likely_folded(time: np.ndarray, period: float) -> bool:
    """Detect if input appears to be phase-folded.

    ModShift requires unfolded time series to work properly. If the input
    is already phase-folded, the algorithm cannot properly identify
    secondary eclipse phases.

    Args:
        time: Time array
        period: Orbital period in days

    Returns:
        True if the input appears to be phase-folded
    """
    if len(time) < 2:
        return False

    baseline = float(time.max() - time.min())

    # If baseline < 1.5 periods, likely folded
    if baseline < 1.5 * period:
        return True

    # If time is normalized to [0, period], definitely folded
    return bool(time.min() >= 0 and time.max() <= period * 1.1)


def _compute_inputs_summary(
    lightcurve: LightCurveData,
    candidate: TransitCandidate,
    is_folded: bool,
) -> dict[str, Any]:
    """Compute input data summary for transparency.

    Args:
        lightcurve: Light curve data
        candidate: Transit candidate parameters
        is_folded: Whether input appears phase-folded

    Returns:
        Dictionary with input summary fields
    """
    mask = lightcurve.valid_mask
    time = lightcurve.time[mask]

    if len(time) < 2:
        return {
            "n_points": len(time),
            "n_transits_expected": 0,
            "cadence_median_min": 0.0,
            "baseline_days": 0.0,
            "snr": candidate.snr if candidate.snr > 0 else None,
            "is_folded": is_folded,
            "flux_err_available": lightcurve.flux_err is not None,
        }

    baseline_days = float(time.max() - time.min())
    n_transits = int(baseline_days / candidate.period) if candidate.period > 0 else 0

    # Compute median cadence
    time_diff = np.diff(time)
    cadence_median = float(np.median(time_diff)) if len(time_diff) > 0 else 0.0
    cadence_median_min = cadence_median * 24.0 * 60.0  # Convert to minutes

    return {
        "n_points": int(np.sum(mask)),
        "n_transits_expected": n_transits,
        "cadence_median_min": round(cadence_median_min, 2),
        "baseline_days": round(baseline_days, 2),
        "snr": candidate.snr if candidate.snr > 0 else None,
        "is_folded": is_folded,
        "flux_err_available": lightcurve.flux_err is not None,
    }


# =============================================================================
# V11: Modshift Check
# =============================================================================


class ModshiftCheck(VetterCheck):
    """V11: Modshift test for secondary eclipse detection at arbitrary phase.

    Astronomical Significance:
    --------------------------
    The standard secondary eclipse search looks at phase 0.5, which is correct
    for circular orbits. However, eccentric orbits can have secondary eclipses
    at phases significantly different from 0.5 (Santerne et al. 2013).

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
    - passed=True: "No strong evidence of eclipsing binary or systematic"
    - passed=False: "Strong evidence of secondary eclipse suggesting EB/blend"

    Confidence Calculation:
    - Regime-based on Fred value and sec/pri ratio
    - Degraded when fred > 2.0 (red noise)
    - Degraded when n_transits < 5

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 (arXiv:1710.06758)
            Section 3.2.3: ModShift in DR25 Robovetter
        [2] Coughlin et al. 2016, ApJS 224, 12 (arXiv:1512.06149)
            Section 4.3: ModShift false positive identification
        [3] Santerne et al. 2013, A&A 557, A139 (arXiv:1307.2003)
            Secondary eclipse phase offset for eccentric orbits
    """

    id = "V11"
    name = "modshift"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default Modshift configuration.

        Note: Threshold fields are DEPRECATED. Threshold interpretation has been
        moved to astro-arc-tess guardrails. By default, this check returns
        passed=None (metrics-only mode). Set legacy_mode=True to compute
        passed based on thresholds.
        """
        return CheckConfig(
            enabled=True,
            threshold=0.5,  # Max secondary/primary ratio before flagging as EB (DEPRECATED)
            additional={
                "fred_warning_threshold": 2.0,  # Fred > 2 indicates significant red noise (DEPRECATED)
                "fred_critical_threshold": 3.5,  # Fred > 3.5 makes result unreliable (DEPRECATED)
                "tertiary_warning_threshold": 0.3,  # If ter/pri > this, warn (DEPRECATED)
                "marginal_secondary_threshold": 0.3,  # Warn if sec/pri > this (DEPRECATED)
                # Metrics-only mode (default): passed=None
                # Set legacy_mode=True to compute passed based on thresholds
                "legacy_mode": False,
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
                    "warnings": ["NO_LIGHTCURVE_DATA"],
                    "passed_meaning": "no_strong_eb_evidence",
                },
            )

        threshold = self.config.threshold or 0.5
        additional = self.config.additional or {}
        fred_warn = additional.get("fred_warning_threshold", 2.0)
        fred_critical = additional.get("fred_critical_threshold", 3.5)
        ter_warn = additional.get("tertiary_warning_threshold", 0.3)
        marginal_sec = additional.get("marginal_secondary_threshold", 0.3)

        # Check for folded input
        mask = lightcurve.valid_mask
        time = lightcurve.time[mask]
        is_folded = _is_likely_folded(time, candidate.period)

        # Compute inputs summary
        inputs_summary = _compute_inputs_summary(lightcurve, candidate, is_folded)

        # If input is folded, return early with warning
        if is_folded:
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,
                confidence=0.10,
                details={
                    "status": "invalid",
                    "reason": "ModShift requires unfolded time series",
                    "warnings": ["FOLDED_INPUT_DETECTED"],
                    "inputs_summary": inputs_summary,
                    "fred_regime": "unknown",
                    "passed_meaning": "no_strong_eb_evidence",
                    "interpretation": (
                        "ModShift requires unfolded time series; result is invalid. "
                        "The input appears to be phase-folded."
                    ),
                },
            )

        try:
            # Import exovetter components
            from exovetter.tce import Tce
            from exovetter.vetters import ModShift

            # Create a lightkurve-like object for exovetter
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
                    "warnings": ["EXOVETTER_IMPORT_ERROR"],
                    "inputs_summary": inputs_summary,
                    "passed_meaning": "no_strong_eb_evidence",
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
                    "warnings": ["MODSHIFT_EXECUTION_ERROR"],
                    "inputs_summary": inputs_summary,
                    "passed_meaning": "no_strong_eb_evidence",
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

        # Classify Fred regime
        fred_regime = _classify_fred_regime(fred)

        # Build warnings list
        warnings: list[str] = []
        if fred > fred_critical:
            warnings.append("FRED_UNRELIABLE")
        elif fred > fred_warn:
            warnings.append("HIGH_RED_NOISE")
        if ter_pri_ratio > ter_warn:
            warnings.append("TERTIARY_SIGNAL")
        if pos > pri:
            warnings.append("POSITIVE_SIGNAL_HIGH")
        if pri < 5 * fa_thresh and fa_thresh > 0:
            warnings.append("LOW_PRIMARY_SNR")
        n_transits = inputs_summary.get("n_transits_expected", 0)
        if n_transits < 5:
            warnings.append("LOW_TRANSIT_COUNT")
        if sec_pri_ratio > marginal_sec and sec_pri_ratio <= threshold:
            warnings.append("MARGINAL_SECONDARY")

        # Get legacy_mode setting
        legacy_mode = self.config.additional.get("legacy_mode", False)

        # Compute threshold-based flags (for reference, even in metrics-only mode)
        significant_secondary = sec_pri_ratio > threshold
        sec_above_fa = sec > fa_thresh * 0.8 and sec > 0
        sec_above_fa_strict = sec > fa_thresh and sec > 0

        # Determine if Fred makes result unreliable
        reliable_result = fred < fred_critical

        # Determine passed value based on mode
        if legacy_mode:
            # Legacy mode: compute passed based on thresholds
            if fred >= fred_critical:
                passed: bool | None = True  # Cannot reliably assess, default to pass
            else:
                # Fail if secondary is significant fraction of primary
                passed = not (significant_secondary and sec_above_fa)
        else:
            # Metrics-only mode: return passed=None, let caller make policy decisions
            passed = None

        # Calculate confidence with regime-based logic
        # For metrics-only mode, use True as placeholder for confidence calculation
        confidence_passed = passed if passed is not None else True
        confidence = self._compute_confidence(
            passed=confidence_passed,
            sec_pri_ratio=sec_pri_ratio,
            fred=fred,
            fred_warn=fred_warn,
            fred_critical=fred_critical,
            ter_pri_ratio=ter_pri_ratio,
            ter_warn=ter_warn,
            n_transits=n_transits,
            snr=inputs_summary.get("snr"),
            reliable_result=reliable_result,
        )

        # Build interpretation
        interpretation = self._build_interpretation(
            passed=confidence_passed,
            sec_pri_ratio=sec_pri_ratio,
            ter_pri_ratio=ter_pri_ratio,
            fred=fred,
            sec_above_fa=sec_above_fa_strict,
            threshold=threshold,
            fred_warn=fred_warn,
            ter_warn=ter_warn,
            fred_regime=fred_regime,
            reliable_result=reliable_result,
        )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=round(confidence, 3),
            details={
                # Metrics-only mode marker
                "_metrics_only": not legacy_mode,
                # Legacy keys (preserved for backward compatibility)
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
                "secondary_above_fa": sec_above_fa_strict,
                "interpretation": interpretation,
                # New keys
                "warnings": warnings,
                "inputs_summary": inputs_summary,
                "fred_regime": fred_regime,
                "passed_meaning": "no_strong_eb_evidence",
                "reliable_result": reliable_result,
            },
        )

    def _compute_confidence(
        self,
        passed: bool,
        sec_pri_ratio: float,
        fred: float,
        fred_warn: float,
        fred_critical: float,
        ter_pri_ratio: float,
        ter_warn: float,
        n_transits: int | None,
        snr: float | None,
        reliable_result: bool,
    ) -> float:
        """Compute confidence based on result quality indicators.

        Args:
            passed: Whether the check passed
            sec_pri_ratio: Secondary/primary signal ratio
            fred: Fred (red noise) value
            fred_warn: Fred warning threshold
            fred_critical: Fred critical threshold
            ter_pri_ratio: Tertiary/primary ratio
            ter_warn: Tertiary warning threshold
            n_transits: Number of expected transits
            snr: Signal-to-noise ratio
            reliable_result: Whether Fred is below critical

        Returns:
            Confidence value between 0.1 and 0.98
        """
        if not reliable_result:
            # Fred is critical - result is unreliable
            return 0.35

        if not passed:
            # Failed check - confidence in the EB detection
            if sec_pri_ratio > 0.8:
                base = 0.95
            elif sec_pri_ratio > 0.6:
                base = 0.90
            else:
                base = 0.85
        else:
            # Passed check - confidence that it's NOT an EB
            if fred > 3.0:
                base = 0.35  # High red noise undermines result
            elif fred > fred_warn:
                base = 0.60
            elif sec_pri_ratio < 0.2:
                base = 0.90  # Well below threshold
            elif sec_pri_ratio < 0.35:
                base = 0.80
            else:
                base = 0.70  # Near threshold

        # Modifiers
        if ter_pri_ratio > ter_warn:
            base -= 0.10
        if n_transits is not None and n_transits > 5:
            base += 0.05
        if snr is not None and snr > 10:
            base += 0.05

        return max(0.1, min(0.98, base))

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
        fred_regime: str,
        reliable_result: bool,
    ) -> str:
        """Build human-readable interpretation of Modshift results."""
        parts = []

        if not reliable_result:
            parts.append(
                f"Warning: Fred={fred:.2f} is in the '{fred_regime}' regime. "
                "Red noise is too high for reliable ModShift analysis. "
                "Defaulting to pass with very low confidence."
            )
            return " ".join(parts)

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
                f"Warning: High red noise level (Fred = {fred:.2f}, regime = '{fred_regime}'). "
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
    - Stellar rotation with starspots (~40% of planet host stars, McQuillan 2013)
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

    Harmonic Aliasing Detection:
    - P/2 aliasing: Stellar rotation at 2*P can create transit-like dips
      when phase-folded at P. The amplitude at P/2 creates depth ~ 2*A.
    - 2P aliasing: Can cause odd/even depth differences

    Pass Criteria:
    - passed=True: "No strong evidence that variability explains transit"
    - passed=False: "Strong evidence of stellar variability at transit period"

    Confidence Calculation:
    - Degraded when n_cycles < 5
    - Degraded when n_transits < 5
    - Degraded when harmonic variability detected

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 (arXiv:1710.06758)
            Section 3.2.4: SWEET test for stellar variability in DR25
        [2] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            Section 4.4: Original SWEET implementation in DR24
        [3] McQuillan et al. 2014, ApJS 211, 24 (arXiv:1402.5694)
            Stellar rotation periods establishing expected variability
        [4] Basri et al. 2013, ApJ 769, 37 (arXiv:1304.0136)
            Typical stellar variability amplitudes (0.1-10 mmag)
    """

    id = "V12"
    name = "sweet"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default SWEET configuration.

        Note: Threshold fields are DEPRECATED. Threshold interpretation has been
        moved to astro-arc-tess guardrails. By default, this check returns
        passed=None (metrics-only mode). Set legacy_mode=True to compute
        passed based on thresholds.
        """
        return CheckConfig(
            enabled=True,
            threshold=3.5,  # amplitude-to-uncertainty ratio threshold at P (DEPRECATED)
            additional={
                "half_period_threshold": 3.5,  # Threshold for P/2 (DEPRECATED)
                "double_period_threshold": 4.0,  # Higher threshold for 2P (DEPRECATED)
                "min_cycles_for_fit": 2.0,  # Minimum baseline/period ratio
                "variability_depth_threshold": 0.5,  # Fraction of depth explainable
                "confidence_floor": 0.30,  # Minimum confidence for degraded checks
                "include_harmonic_analysis": True,  # Enable harmonic failure logic
                # Metrics-only mode (default): passed=None
                # Set legacy_mode=True to compute passed based on thresholds
                "legacy_mode": False,
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
                    "warnings": ["NO_LIGHTCURVE_DATA"],
                },
            )

        threshold = self.config.threshold or 3.5
        additional = self.config.additional or {}
        half_p_thresh = additional.get("half_period_threshold", 3.5)
        double_p_thresh = additional.get("double_period_threshold", 4.0)
        min_cycles = additional.get("min_cycles_for_fit", 2.0)
        var_depth_thresh = additional.get("variability_depth_threshold", 0.5)
        conf_floor = additional.get("confidence_floor", 0.30)
        include_harmonic = additional.get("include_harmonic_analysis", True)

        # Compute inputs summary
        inputs_summary = self._compute_inputs_summary(lightcurve, candidate)
        n_cycles = inputs_summary.get("n_cycles_observed", 0)

        # Check minimum data requirements
        warnings: list[str] = []
        if n_cycles < min_cycles:
            warnings.append("LOW_BASELINE_CYCLES")
        if n_cycles < 4.0:
            # Cannot reliably detect 2P variability
            inputs_summary["can_detect_2p"] = False
            warnings.append("CANNOT_DETECT_2P")
        else:
            inputs_summary["can_detect_2p"] = True

        n_transits = inputs_summary.get("n_transits", 0)
        if n_transits < 3:
            warnings.append("INSUFFICIENT_TRANSITS")

        snr = inputs_summary.get("snr")
        if snr is not None and snr < 7:
            warnings.append("LOW_SNR")

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
                    "warnings": ["EXOVETTER_IMPORT_ERROR"],
                    "inputs_summary": inputs_summary,
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
                    "warnings": ["SWEET_EXECUTION_ERROR"],
                    "inputs_summary": inputs_summary,
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

        # Compute harmonic analysis
        transit_depth_ppm = candidate.depth * 1e6
        harmonic_analysis = self._analyze_harmonics(
            amp_results, transit_depth_ppm, candidate.period
        )

        # Compute aliasing flags
        aliasing_flags = self._compute_aliasing_flags(
            amp_results, transit_depth_ppm, half_p_thresh, var_depth_thresh
        )

        # Add harmonic warnings
        if half_p_ratio > 2.0 or double_p_ratio > 2.0:
            warnings.append("HARMONIC_VARIABILITY_DETECTED")
        var_explains = harmonic_analysis.get("variability_explains_depth_fraction", 0.0)
        if var_explains > 0.3:
            warnings.append("VARIABILITY_MAY_EXPLAIN_TRANSIT")

        # Get legacy_mode setting
        legacy_mode = self.config.additional.get("legacy_mode", False)

        # Compute threshold-based flags (for reference, even in metrics-only mode)
        # Primary concern is the period itself
        fails_at_period = period_ratio > threshold
        fails_at_half = half_p_ratio > half_p_thresh
        fails_at_double = double_p_ratio > double_p_thresh

        # Compute harmonic-based failure flag
        half_p_depth = harmonic_analysis.get("variability_induced_depth_at_half_P_ppm", 0.0)
        fails_at_half_with_depth = (
            fails_at_half and half_p_depth > transit_depth_ppm * var_depth_thresh
        )

        # Determine passed value based on mode
        if legacy_mode:
            # Legacy mode: compute passed based on thresholds
            if include_harmonic:
                # Also fail if P/2 variability explains significant fraction of depth
                passed: bool | None = not (fails_at_period or fails_at_half_with_depth)
            else:
                # Legacy behavior: only fail on period itself
                passed = not fails_at_period
        else:
            # Metrics-only mode: return passed=None, let caller make policy decisions
            passed = None

        # Calculate confidence with data quality scaling
        # For metrics-only mode, use True as placeholder for confidence calculation
        confidence_passed = passed if passed is not None else True
        confidence = self._compute_confidence(
            passed=confidence_passed,
            period_ratio=period_ratio,
            threshold=threshold,
            fails_at_half=fails_at_half,
            fails_at_double=fails_at_double,
            warnings=warnings,
            inputs_summary=inputs_summary,
            conf_floor=conf_floor,
        )

        # Build interpretation
        interpretation = self._build_interpretation(
            passed=confidence_passed,
            period_ratio=period_ratio,
            half_p_ratio=half_p_ratio,
            double_p_ratio=double_p_ratio,
            threshold=threshold,
            half_p_thresh=half_p_thresh,
            double_p_thresh=double_p_thresh,
            msg=msg,
            harmonic_analysis=harmonic_analysis,
            include_harmonic=include_harmonic,
        )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=round(confidence, 3),
            details={
                # Metrics-only mode marker
                "_metrics_only": not legacy_mode,
                # Legacy keys (preserved for backward compatibility)
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
                # New keys
                "warnings": warnings,
                "inputs_summary": inputs_summary,
                "harmonic_analysis": harmonic_analysis,
                "aliasing_flags": aliasing_flags,
            },
        )

    def _compute_inputs_summary(
        self,
        lightcurve: LightCurveData,
        candidate: TransitCandidate,
    ) -> dict[str, Any]:
        """Compute input data quality summary for SWEET analysis.

        Args:
            lightcurve: Light curve data
            candidate: Transit candidate parameters

        Returns:
            Dictionary with input summary fields
        """
        mask = lightcurve.valid_mask
        time = lightcurve.time[mask]

        if len(time) < 2:
            return {
                "n_points": len(time),
                "n_transits": 0,
                "n_cycles_observed": 0.0,
                "baseline_days": 0.0,
                "cadence_minutes": 0.0,
                "snr": candidate.snr if candidate.snr > 0 else None,
                "can_detect_2p": False,
            }

        baseline_days = float(time.max() - time.min())
        n_cycles = baseline_days / candidate.period if candidate.period > 0 else 0.0
        n_transits = int(n_cycles) if n_cycles > 0 else 0

        # Compute median cadence
        time_diff = np.diff(time)
        cadence_min = float(np.median(time_diff)) * 24.0 * 60.0 if len(time_diff) > 0 else 0.0

        return {
            "n_points": int(np.sum(mask)),
            "n_transits": n_transits,
            "n_cycles_observed": round(n_cycles, 2),
            "baseline_days": round(baseline_days, 2),
            "cadence_minutes": round(cadence_min, 2),
            "snr": candidate.snr if candidate.snr > 0 else None,
            "can_detect_2p": n_cycles >= 4.0,
        }

    def _analyze_harmonics(
        self,
        amp_results: dict[str, dict[str, float]],
        transit_depth_ppm: float,
        period: float,
    ) -> dict[str, Any]:
        """Compute whether variability at harmonics could explain the transit.

        For P/2 variability: amplitude A creates transit-like depth of ~2*A
        when phase-folded at period P.

        Args:
            amp_results: Amplitude results from exovetter
            transit_depth_ppm: Transit depth in ppm
            period: Orbital period in days

        Returns:
            Dictionary with harmonic analysis results
        """
        # Variability at P/2 with amplitude A creates transit-like depth of ~2*A
        amp_at_half_p = amp_results.get("half_period", {}).get("amplitude", 0.0) * 1e6
        variability_depth_at_half_p = 2.0 * amp_at_half_p

        # Variability at P directly maps to depth
        amp_at_p = amp_results.get("period", {}).get("amplitude", 0.0) * 1e6
        variability_depth_at_p = 2.0 * amp_at_p

        # Fraction of transit depth explainable by variability
        max_var_depth = max(variability_depth_at_p, variability_depth_at_half_p)
        var_explains_fraction = max_var_depth / transit_depth_ppm if transit_depth_ppm > 0 else 0.0

        # Identify dominant variability period
        half_p_ratio = amp_results.get("half_period", {}).get("ratio", 0.0)
        period_ratio = amp_results.get("period", {}).get("ratio", 0.0)
        double_p_ratio = amp_results.get("double_period", {}).get("ratio", 0.0)

        if period_ratio > half_p_ratio and period_ratio > double_p_ratio:
            dominant = "P"
        elif half_p_ratio > period_ratio and half_p_ratio > double_p_ratio:
            dominant = "P/2"
        elif double_p_ratio > period_ratio and double_p_ratio > half_p_ratio:
            dominant = "2P"
        else:
            dominant = "none"

        return {
            "variability_induced_depth_at_P_ppm": round(variability_depth_at_p, 1),
            "variability_induced_depth_at_half_P_ppm": round(variability_depth_at_half_p, 1),
            "variability_explains_depth_fraction": round(min(var_explains_fraction, 1.0), 3),
            "dominant_variability_period": dominant,
        }

    def _compute_aliasing_flags(
        self,
        amp_results: dict[str, dict[str, float]],
        transit_depth_ppm: float,
        half_p_thresh: float,
        var_depth_thresh: float,
    ) -> dict[str, Any]:
        """Compute aliasing risk flags.

        Args:
            amp_results: Amplitude results from exovetter
            transit_depth_ppm: Transit depth in ppm
            half_p_thresh: Threshold for P/2 significance
            var_depth_thresh: Threshold for depth explanation fraction

        Returns:
            Dictionary with aliasing flag information
        """
        half_p_ratio = amp_results.get("half_period", {}).get("ratio", 0.0)
        double_p_ratio = amp_results.get("double_period", {}).get("ratio", 0.0)

        amp_at_half_p = amp_results.get("half_period", {}).get("amplitude", 0.0) * 1e6
        var_depth_half_p = 2.0 * amp_at_half_p

        # P/2 alias risk: variability at P/2 could mimic transit
        half_p_alias_risk = (
            (half_p_ratio > half_p_thresh * 0.5 and var_depth_half_p > transit_depth_ppm * 0.3)
            if transit_depth_ppm > 0
            else False
        )

        # 2P alias risk: variability at 2P could affect odd/even
        double_p_alias_risk = double_p_ratio > 3.0

        # Dominant alias
        if half_p_alias_risk and not double_p_alias_risk:
            dominant_alias = "P/2"
        elif double_p_alias_risk and not half_p_alias_risk:
            dominant_alias = "2P"
        elif half_p_alias_risk and double_p_alias_risk:
            dominant_alias = "P/2"  # P/2 is more concerning
        else:
            dominant_alias = None

        return {
            "half_period_alias_risk": half_p_alias_risk,
            "double_period_alias_risk": double_p_alias_risk,
            "dominant_alias": dominant_alias,
        }

    def _compute_confidence(
        self,
        passed: bool,
        period_ratio: float,
        threshold: float,
        fails_at_half: bool,
        fails_at_double: bool,
        warnings: list[str],
        inputs_summary: dict[str, Any],
        conf_floor: float,
    ) -> float:
        """Compute confidence with data-quality scaling.

        Args:
            passed: Whether the check passed
            period_ratio: Amplitude ratio at transit period
            threshold: Significance threshold
            fails_at_half: Whether P/2 fails threshold
            fails_at_double: Whether 2P fails threshold
            warnings: List of warning strings
            inputs_summary: Input data quality summary
            conf_floor: Minimum confidence floor

        Returns:
            Confidence value
        """
        # Base confidence
        if not passed:
            # Failed - confidence in variability detection
            if period_ratio > threshold * 2:
                base = 0.95
            elif period_ratio > threshold:
                base = 0.85
            else:
                base = 0.80
        else:
            # Passed
            if fails_at_half or fails_at_double:
                base = 0.75
            elif period_ratio < threshold / 2:
                base = 0.95
            elif period_ratio < threshold:
                base = 0.85
            else:
                base = 0.80

        # Apply penalties for data quality issues
        penalties = 0.0

        n_cycles = inputs_summary.get("n_cycles_observed", 0)
        if n_cycles < 5:
            penalties += 0.15

        n_transits = inputs_summary.get("n_transits", 10)
        if n_transits < 5:
            penalties += 0.10

        snr = inputs_summary.get("snr")
        if snr is not None and snr < 10:
            penalties += 0.10

        if "HARMONIC_VARIABILITY_DETECTED" in warnings:
            penalties += 0.10

        return max(base - penalties, conf_floor)

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
        harmonic_analysis: dict[str, Any],
        include_harmonic: bool,
    ) -> str:
        """Build human-readable interpretation of SWEET results."""
        parts = []

        if not passed:
            parts.append(
                f"Significant sinusoidal signal detected at transit period "
                f"(amp/sigma = {period_ratio:.2f}, threshold = {threshold}). "
                "This signal may be stellar variability rather than a planet transit."
            )
            if include_harmonic:
                var_frac = harmonic_analysis.get("variability_explains_depth_fraction", 0.0)
                if var_frac > 0.3:
                    parts.append(
                        f"Variability can explain {var_frac * 100:.0f}% of the transit depth."
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
            if include_harmonic:
                depth_half = harmonic_analysis.get("variability_induced_depth_at_half_P_ppm", 0.0)
                if depth_half > 0:
                    parts.append(
                        f"P/2 variability could induce {depth_half:.0f} ppm transit-like depth."
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
