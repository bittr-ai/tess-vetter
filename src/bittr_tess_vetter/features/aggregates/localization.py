"""Localization verdict and metrics aggregation."""

from .contracts import LocalizationInput, LocalizationSummary, V09Metrics
from .verdicts import normalize_verdict


def build_localization_summary(
    localization: LocalizationInput | None,
    v09: V09Metrics | None = None,
) -> LocalizationSummary:
    """Build normalized localization summary."""
    if not localization:
        return LocalizationSummary()

    result: LocalizationSummary = {}

    verdict = normalize_verdict(localization.get("verdict"))
    if verdict is not None:
        result["localization_verdict"] = verdict

    target_distance = localization.get("target_distance_arcsec")
    if target_distance is not None:
        result["localization_target_distance_arcsec"] = target_distance

    uncertainty = localization.get("uncertainty_semimajor_arcsec")
    if uncertainty is not None:
        result["localization_uncertainty_semimajor_arcsec"] = uncertainty

    # Low confidence if large uncertainty or warnings present
    warnings = localization.get("warnings") or []
    low_confidence = len(warnings) > 0 or (uncertainty is not None and uncertainty > 10.0)
    result["localization_low_confidence"] = low_confidence

    host_ambiguous = localization.get("host_ambiguous_within_1pix")
    if host_ambiguous is not None:
        result["host_ambiguous_within_1pix"] = host_ambiguous

    # V09 reliability:
    # Prefer the explicit boolean from the V09 check when available.
    # Otherwise infer from pixel-space distance + lack of warnings.
    if v09:
        v09_warnings = v09.get("warnings") or []
        explicit = v09.get("localization_reliable")
        if explicit is not None:
            v09_reliable = bool(explicit)
        else:
            v09_distance_px = v09.get("distance_to_target_pixels")
            v09_reliable = (
                len(v09_warnings) == 0
                and v09_distance_px is not None
                and v09_distance_px < 1.0
            )
        result["v09_localization_reliable"] = v09_reliable

    return result
