"""Host-side policy layer for interpreting metrics-only vetting outputs.

`bittr_tess_vetter` returns check results in "metrics-only" mode by default:
`CheckResult.passed` is typically None and callers should interpret the metrics.

This module provides a lightweight, explicit policy layer that turns those
metrics into PASS/WARN/FAIL/UNKNOWN dispositions without changing the underlying
metrics contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bittr_tess_vetter.api.types import CheckResult, VettingBundleResult


class Disposition(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds for default policy interpretation.

    These defaults are conservative and intended to support automated triage,
    not planet validation.
    """

    # V01 odd/even depth
    odd_even_min_transits_per_parity: int = 3
    odd_even_rel_diff_threshold: float = 0.15
    odd_even_warn_sigma: float = 2.5
    odd_even_fail_sigma: float = 4.0

    # V02 secondary eclipse
    secondary_warn_sigma: float = 3.0
    secondary_fail_sigma: float = 5.0

    # V03 duration consistency (only when density-corrected)
    duration_warn_ratio_low: float = 0.5
    duration_warn_ratio_high: float = 2.0

    # V04 depth stability
    depth_stability_warn_chi2: float = 5.0
    depth_stability_fail_chi2: float = 10.0

    # V05 transit shape (V-shape)
    vshape_warn_tflat_ttotal_upper: float = 0.2
    vshape_fail_tflat_ttotal_upper: float = 0.1

    # V08 centroid shift
    centroid_warn_sigma: float = 3.0
    centroid_fail_sigma: float = 5.0

    # V09 pixel-level localization proxy
    pixel_warn_distance_px: float = 1.0
    pixel_fail_distance_px: float = 2.0

    # V10 aperture dependence
    aperture_warn_stability_metric: float = 0.35


@dataclass(frozen=True)
class PolicyCheckResult:
    id: str
    name: str
    disposition: Disposition
    confidence: float
    reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyBundleResult:
    """Policy-interpreted version of VettingBundleResult."""

    results: list[PolicyCheckResult]
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config: PolicyConfig = field(default_factory=PolicyConfig)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.disposition is Disposition.PASS)

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.results if r.disposition is Disposition.WARN)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.disposition is Disposition.FAIL)

    @property
    def n_unknown(self) -> int:
        return sum(1 for r in self.results if r.disposition is Disposition.UNKNOWN)


def _as_float(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_int(d: dict[str, Any], key: str) -> int | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _has_errors_or_missing_dependency(details: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    warnings = details.get("warnings")
    if not isinstance(warnings, list):
        return False, ()
    warning_set = {str(w) for w in warnings}
    blocking = {"EXOVETTER_IMPORT_ERROR", "MODSHIFT_EXECUTION_ERROR", "SWEET_EXECUTION_ERROR"}
    hit = sorted(warning_set & blocking)
    if hit:
        return True, tuple(hit)
    return False, ()


def apply_policy_to_check(check: CheckResult, *, config: PolicyConfig | None = None) -> PolicyCheckResult:
    """Convert one metrics-only CheckResult into a policy disposition.

    This does not modify the original check output.
    """
    cfg = config or PolicyConfig()
    details = dict(check.details or {})

    # If a check explicitly provides pass/fail, honor it (rare; intended for host-side policy).
    if check.passed is True:
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=("passed_boolean",),
            metrics=details,
        )
    if check.passed is False:
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.FAIL,
            confidence=float(check.confidence),
            reasons=("failed_boolean",),
            metrics=details,
        )

    blocked, blocking_reasons = _has_errors_or_missing_dependency(details)
    if blocked:
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.UNKNOWN,
            confidence=float(check.confidence),
            reasons=blocking_reasons,
            metrics=details,
        )

    # Metrics-only policy by check ID.
    if check.id == "V01":
        delta_sigma = _as_float(details, "delta_sigma") or _as_float(details, "depth_diff_sigma")
        rel_diff = _as_float(details, "rel_diff")
        n_odd = _as_int(details, "n_odd_transits")
        n_even = _as_int(details, "n_even_transits")
        if (
            delta_sigma is not None
            and rel_diff is not None
            and n_odd is not None
            and n_even is not None
            and min(n_odd, n_even) >= cfg.odd_even_min_transits_per_parity
            and rel_diff >= cfg.odd_even_rel_diff_threshold
        ):
            if delta_sigma >= cfg.odd_even_fail_sigma:
                return PolicyCheckResult(
                    id=check.id,
                    name=check.name,
                    disposition=Disposition.FAIL,
                    confidence=float(check.confidence),
                    reasons=("odd_even_depth_strong_mismatch",),
                    metrics=details,
                )
            if delta_sigma >= cfg.odd_even_warn_sigma:
                return PolicyCheckResult(
                    id=check.id,
                    name=check.name,
                    disposition=Disposition.WARN,
                    confidence=float(check.confidence),
                    reasons=("odd_even_depth_mismatch",),
                    metrics=details,
                )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.UNKNOWN,
            confidence=float(check.confidence),
            reasons=("odd_even_insufficient_power_or_data",),
            metrics=details,
        )

    if check.id == "V02":
        sigma = _as_float(details, "secondary_depth_sigma")
        if sigma is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("secondary_metrics_missing",),
                metrics=details,
            )
        if sigma >= cfg.secondary_fail_sigma:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.FAIL,
                confidence=float(check.confidence),
                reasons=("significant_secondary_eclipse",),
                metrics=details,
            )
        if sigma >= cfg.secondary_warn_sigma:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("possible_secondary_eclipse",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=(),
            metrics=details,
        )

    if check.id == "V03":
        ratio = _as_float(details, "duration_ratio")
        density_corrected = bool(details.get("density_corrected"))
        if ratio is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("duration_ratio_missing",),
                metrics=details,
            )
        if density_corrected and (ratio <= cfg.duration_warn_ratio_low or ratio >= cfg.duration_warn_ratio_high):
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("duration_inconsistent_with_stellar_density",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.UNKNOWN if not density_corrected else Disposition.PASS,
            confidence=float(check.confidence),
            reasons=("duration_not_density_corrected",) if not density_corrected else (),
            metrics=details,
        )

    if check.id == "V04":
        chi2 = _as_float(details, "chi2_reduced")
        if chi2 is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("depth_stability_metrics_missing",),
                metrics=details,
            )
        if chi2 >= cfg.depth_stability_fail_chi2:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.FAIL,
                confidence=float(check.confidence),
                reasons=("depths_highly_inconsistent",),
                metrics=details,
            )
        if chi2 >= cfg.depth_stability_warn_chi2:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("depths_inconsistent",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=(),
            metrics=details,
        )

    if check.id == "V05":
        ratio = _as_float(details, "tflat_ttotal_ratio")
        ratio_err = _as_float(details, "tflat_ttotal_ratio_err") or _as_float(details, "shape_metric_uncertainty")
        status = str(details.get("status") or "")
        if status != "ok" or ratio is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("vshape_insufficient_data",),
                metrics=details,
            )
        upper = ratio + (ratio_err or 0.0)
        if upper <= cfg.vshape_fail_tflat_ttotal_upper:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.FAIL,
                confidence=float(check.confidence),
                reasons=("strong_vshape_signature",),
                metrics=details,
            )
        if upper <= cfg.vshape_warn_tflat_ttotal_upper:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("possible_vshape_signature",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=(),
            metrics=details,
        )

    if check.id == "V08":
        sigma = _as_float(details, "significance_sigma")
        if sigma is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("centroid_metrics_missing",),
                metrics=details,
            )
        if sigma >= cfg.centroid_fail_sigma:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.FAIL,
                confidence=float(check.confidence),
                reasons=("significant_centroid_shift",),
                metrics=details,
            )
        if sigma >= cfg.centroid_warn_sigma:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("possible_centroid_shift",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=(),
            metrics=details,
        )

    if check.id == "V09":
        dist = _as_float(details, "distance_to_target_pixels")
        if dist is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("pixel_depth_map_missing",),
                metrics=details,
            )
        if dist >= cfg.pixel_fail_distance_px:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.FAIL,
                confidence=float(check.confidence),
                reasons=("pixel_depth_off_target",),
                metrics=details,
            )
        if dist >= cfg.pixel_warn_distance_px:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("pixel_depth_ambiguous_target",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=(),
            metrics=details,
        )

    if check.id == "V10":
        stability = _as_float(details, "stability_metric")
        flags = details.get("flags")
        if stability is None:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("aperture_metrics_missing",),
                metrics=details,
            )
        if isinstance(flags, list) and len(flags) > 0:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("aperture_dependence_flags",),
                metrics=details,
            )
        if stability <= cfg.aperture_warn_stability_metric:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.WARN,
                confidence=float(check.confidence),
                reasons=("aperture_depth_unstable",),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.PASS,
            confidence=float(check.confidence),
            reasons=(),
            metrics=details,
        )

    # Exovetter checks: default to UNKNOWN unless they provide a clear OK/FAIL hint.
    if check.id in {"V11", "V12"}:
        status = str(details.get("status") or "")
        if status in {"error", "invalid"}:
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.UNKNOWN,
                confidence=float(check.confidence),
                reasons=("exovetter_unavailable_or_failed",),
                metrics=details,
            )
        msg = None
        raw = details.get("raw_metrics")
        if isinstance(raw, dict):
            msg = raw.get("msg")
        if isinstance(msg, str) and msg.strip().upper().startswith("OK"):
            return PolicyCheckResult(
                id=check.id,
                name=check.name,
                disposition=Disposition.PASS,
                confidence=float(check.confidence),
                reasons=(),
                metrics=details,
            )
        return PolicyCheckResult(
            id=check.id,
            name=check.name,
            disposition=Disposition.UNKNOWN,
            confidence=float(check.confidence),
            reasons=("exovetter_metrics_uninterpreted",),
            metrics=details,
        )

    return PolicyCheckResult(
        id=check.id,
        name=check.name,
        disposition=Disposition.UNKNOWN,
        confidence=float(check.confidence),
        reasons=("no_policy_rule_for_check",),
        metrics=details,
    )


def apply_policy(bundle: VettingBundleResult, *, config: PolicyConfig | None = None) -> PolicyBundleResult:
    """Apply a conservative default policy to a full vetting bundle."""
    cfg = config or PolicyConfig()
    interpreted = [apply_policy_to_check(r, config=cfg) for r in bundle.results]
    provenance = dict(bundle.provenance or {})
    provenance.setdefault("policy", {})
    provenance["policy"] = {
        "name": "default",
        "config": cfg.__dict__,
    }
    return PolicyBundleResult(
        results=interpreted,
        provenance=provenance,
        warnings=list(bundle.warnings),
        config=cfg,
    )


__all__ = [
    "Disposition",
    "PolicyConfig",
    "PolicyCheckResult",
    "PolicyBundleResult",
    "apply_policy",
    "apply_policy_to_check",
]

