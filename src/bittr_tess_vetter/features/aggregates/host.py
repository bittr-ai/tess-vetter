"""Host plausibility summary aggregation."""

from .contracts import HostPlausibilityInput, HostPlausibilitySummary


def build_host_plausibility_summary(
    host: HostPlausibilityInput | None,
) -> HostPlausibilitySummary:
    """Summarize host plausibility findings."""
    if not host:
        return HostPlausibilitySummary()

    result: HostPlausibilitySummary = {}

    requires_followup = host.get("requires_resolved_followup")
    if requires_followup is not None:
        result["host_requires_resolved_followup"] = requires_followup

    impossible_ids = host.get("physically_impossible_source_ids") or []
    result["host_physically_impossible_count"] = len(impossible_ids)
    if impossible_ids:
        result["host_physically_impossible_source_ids"] = impossible_ids

    rationale = host.get("rationale")
    if rationale is not None:
        result["host_plausibility_rationale"] = rationale

    # Find best feasible host (lowest depth_correction_factor among non-impossible).
    #
    # Prefer explicit per-scenario physically_impossible, but also respect the
    # top-level impossible ID list if present.
    scenarios = host.get("scenarios") or []
    impossible_set = set(host.get("physically_impossible_source_ids") or [])
    best_feasible = None
    best_dcf: float = float("inf")
    for scenario in scenarios:
        sid = scenario.get("source_id")
        if scenario.get("physically_impossible") or (sid is not None and sid in impossible_set):
            continue
        dcf = scenario.get("depth_correction_factor")
        if dcf is None:
            continue
        if dcf < best_dcf:
            best_feasible = scenario
            best_dcf = dcf

    if best_feasible is not None:
        source_id = best_feasible.get("source_id")
        if source_id is not None:
            result["host_feasible_best_source_id"] = source_id
        flux_fraction = best_feasible.get("flux_fraction")
        if flux_fraction is not None:
            result["host_feasible_best_flux_fraction"] = flux_fraction
        true_depth_ppm = best_feasible.get("true_depth_ppm")
        if true_depth_ppm is not None:
            result["host_feasible_best_true_depth_ppm"] = true_depth_ppm

    return result
