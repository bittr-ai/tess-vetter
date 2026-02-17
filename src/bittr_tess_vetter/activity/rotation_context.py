"""Rotation-context primitives for follow-up feasibility diagnostics."""

from __future__ import annotations

from typing import Any


def estimate_v_eq_kms(
    *,
    stellar_radius_rsun: float | None,
    rotation_period_days: float | None,
) -> float | None:
    """Estimate equatorial velocity (km/s) from radius and rotation period."""
    if stellar_radius_rsun is None or rotation_period_days is None:
        return None
    if stellar_radius_rsun <= 0.0 or rotation_period_days <= 0.0:
        return None
    return 50.6 * float(stellar_radius_rsun) / float(rotation_period_days)


def build_rotation_context(
    *,
    rotation_period_days: float | None,
    stellar_radius_rsun: float | None,
    rotation_period_source: str | None = None,
    stellar_radius_source: str | None = None,
) -> dict[str, Any]:
    """Build threshold-free rotation context payload.

    This is intentionally policy-free: it surfaces raw physically meaningful
    context and quality flags without embedding hard science thresholds.
    """
    quality_flags: list[str] = []
    if rotation_period_days is None:
        quality_flags.append("MISSING_ROTATION_PERIOD")
    elif float(rotation_period_days) <= 0.0:
        quality_flags.append("INVALID_ROTATION_PERIOD")

    if stellar_radius_rsun is None:
        quality_flags.append("MISSING_STELLAR_RADIUS")
    elif float(stellar_radius_rsun) <= 0.0:
        quality_flags.append("INVALID_STELLAR_RADIUS")

    v_eq_est_kms = estimate_v_eq_kms(
        stellar_radius_rsun=stellar_radius_rsun,
        rotation_period_days=rotation_period_days,
    )
    status = "READY" if v_eq_est_kms is not None else "INCOMPLETE_INPUTS"

    return {
        "status": status,
        "rotation_period_days": float(rotation_period_days) if rotation_period_days is not None else None,
        "stellar_radius_rsun": float(stellar_radius_rsun) if stellar_radius_rsun is not None else None,
        "v_eq_est_kms": round(float(v_eq_est_kms), 2) if v_eq_est_kms is not None else None,
        "quality_flags": quality_flags,
        "provenance": {
            "rotation_period_source": rotation_period_source,
            "stellar_radius_source": stellar_radius_source,
        },
    }


__all__ = ["estimate_v_eq_kms", "build_rotation_context"]

