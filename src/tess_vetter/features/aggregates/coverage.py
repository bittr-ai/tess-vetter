"""Feature family coverage computation."""

from .contracts import CheckPresenceFlags, CoverageSummary

# Canonical feature family names
FAMILY_MODSHIFT = "MODSHIFT"
FAMILY_TPF_LOCALIZATION = "TPF_LOCALIZATION"
FAMILY_PIXEL_TIMESERIES = "PIXEL_TIMESERIES"
FAMILY_GHOST_RELIABILITY = "GHOST_RELIABILITY"
FAMILY_HOST_PLAUSIBILITY = "HOST_PLAUSIBILITY"


def compute_missing_families(flags: CheckPresenceFlags) -> CoverageSummary:
    """Compute which feature families are missing based on presence flags."""
    missing: list[str] = []

    # TPF-dependent families
    if not flags.get("has_tpf"):
        missing.append(FAMILY_TPF_LOCALIZATION)
        missing.append(FAMILY_PIXEL_TIMESERIES)
        missing.append(FAMILY_GHOST_RELIABILITY)
    else:
        if not flags.get("has_localization") and not flags.get("has_diff_image"):
            missing.append(FAMILY_TPF_LOCALIZATION)
        if not flags.get("has_pixel_timeseries"):
            missing.append(FAMILY_PIXEL_TIMESERIES)
        if not flags.get("has_ghost_summary"):
            missing.append(FAMILY_GHOST_RELIABILITY)

    if not flags.get("has_host_plausibility"):
        missing.append(FAMILY_HOST_PLAUSIBILITY)

    # TPF coverage OK: has TPF AND (localization OR diff_image) AND aperture
    tpf_ok = (
        flags.get("has_tpf", False)
        and (flags.get("has_localization", False) or flags.get("has_diff_image", False))
        and flags.get("has_aperture_family", False)
    )

    return CoverageSummary(
        tpf_coverage_ok=tpf_ok,
        missing_feature_families=missing,
    )
