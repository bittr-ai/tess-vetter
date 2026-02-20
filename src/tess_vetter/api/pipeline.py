"""Vetting pipeline for orchestrating check execution.

This module provides the VettingPipeline class which runs multiple
vetting checks and aggregates their results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    VettingCheck,
    get_default_registry,
)
from tess_vetter.validation.result_schema import (
    CheckResult,
    VettingBundleResult,
    skipped_result,
)

if TYPE_CHECKING:
    from tess_vetter.domain.detection import TransitCandidate
    from tess_vetter.domain.lightcurve import LightCurveData


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        timeout_seconds: Default timeout for each check.
        random_seed: Seed for reproducibility.
        emit_warnings: Whether to emit warnings via warnings module.
        fail_fast: Stop on first error.
    """

    timeout_seconds: float | None = None
    random_seed: int | None = None
    emit_warnings: bool = False
    fail_fast: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)


class VettingPipeline:
    """Pipeline for running multiple vetting checks.

    The pipeline handles:
    - Check selection and ordering
    - Input validation and requirement checking
    - Skipping checks when requirements aren't met
    - Aggregating results into VettingBundleResult

    Example:
        >>> pipeline = VettingPipeline()
        >>> result = pipeline.run(lc, candidate, network=False)
        >>> for check_result in result.results:
        ...     print(f"{check_result.id}: {check_result.status}")
    """

    def __init__(
        self,
        checks: list[str] | None = None,
        *,
        registry: CheckRegistry | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            checks: List of check IDs to run. If None, runs all registered checks.
            registry: Check registry to use. Defaults to global registry.
            config: Pipeline configuration.
        """
        self._registry = get_default_registry() if registry is None else registry
        self._config = config or PipelineConfig()
        self._check_ids = checks  # None means "all"

    def _get_checks_to_run(self) -> list[VettingCheck]:
        """Get the list of checks to execute."""
        if self._check_ids is None:
            return self._registry.list()
        return [self._registry.get(id) for id in self._check_ids]

    def _check_requirements_met(
        self,
        requirements: CheckRequirements,
        inputs: CheckInputs,
    ) -> tuple[bool, str | None]:
        """Check if requirements are satisfied.

        Returns:
            Tuple of (met: bool, reason: str | None).
        """
        if requirements.needs_tpf and inputs.tpf is None:
            return False, "NO_TPF"
        if requirements.needs_network and not inputs.network:
            return False, "NETWORK_DISABLED"
        if requirements.needs_ra_dec and (inputs.ra_deg is None or inputs.dec_deg is None):
            return False, "NO_COORDINATES"
        if requirements.needs_tic_id and inputs.tic_id is None:
            return False, "NO_TIC_ID"
        if requirements.optional_deps:
            for dep in requirements.optional_deps:
                if not _optional_dep_available(dep):
                    return False, f"EXTRA_MISSING:{dep}"
        # needs_stellar is soft - we don't skip, just note it
        return True, None

    def run(
        self,
        lc: LightCurveData,
        candidate: TransitCandidate,
        *,
        stellar: Any | None = None,
        tpf: Any | None = None,
        network: bool = False,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        tic_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> VettingBundleResult:
        """Run the vetting pipeline.

        Args:
            lc: Light curve data.
            candidate: Candidate with ephemeris.
            stellar: Optional stellar parameters.
            tpf: Optional TPF data.
            network: Whether network access is allowed.
            ra_deg: Right ascension in degrees.
            dec_deg: Declination in degrees.
            tic_id: TIC identifier.
            context: Additional context for checks.

        Returns:
            VettingBundleResult with all check results.
        """
        start_time = time.time()

        # Build inputs container
        inputs = CheckInputs(
            lc=lc,
            candidate=candidate,
            stellar=stellar,
            tpf=tpf,
            network=network,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
            context=context or {},
        )

        # Build check config
        check_config = CheckConfig(
            timeout_seconds=self._config.timeout_seconds,
            random_seed=self._config.random_seed,
            extra_params=self._config.extra_params,
        )

        results: list[CheckResult] = []
        warnings: list[str] = []

        checks = self._get_checks_to_run()

        for check in checks:
            # Check requirements
            met, reason = self._check_requirements_met(check.requirements, inputs)

            if not met and reason:
                # Skip with structured reason
                result = skipped_result(
                    check.id,
                    check.name,
                    reason_flag=reason,
                    notes=[_format_skip_note(reason)],
                )
                results.append(result)
                warnings.append(f"{check.id} ({check.name}) skipped: {reason}")
                continue

            # Run the check
            try:
                result = check.run(inputs, check_config)
                results.append(result)
            except Exception as e:
                # Convert exception to error result
                from tess_vetter.validation.result_schema import error_result

                result = error_result(
                    check.id,
                    check.name,
                    error=type(e).__name__,
                    notes=[str(e)],
                )
                results.append(result)
                warnings.append(f"{check.id} ({check.name}) error: {e}")

                if self._config.fail_fast:
                    break

        duration_ms = (time.time() - start_time) * 1000

        # Build inputs summary
        inputs_summary = {
            "has_stellar": stellar is not None,
            "has_tpf": tpf is not None,
            "network": network,
            "has_coordinates": ra_deg is not None and dec_deg is not None,
            "has_tic_id": tic_id is not None,
        }

        return VettingBundleResult(
            results=results,
            warnings=warnings,
            provenance={
                "pipeline_version": "0.1.0",
                "duration_ms": round(duration_ms, 2),
                "checks_requested": self._check_ids,
                "checks_run": len(results),
            },
            inputs_summary=inputs_summary,
        )

    def run_many(
        self,
        lc: LightCurveData,
        candidates: list[TransitCandidate],
        *,
        stellar: Any | None = None,
        tpf: Any | None = None,
        network: bool = False,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        tic_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[list[VettingBundleResult], list[dict[str, Any]]]:
        """Run the vetting pipeline for many candidates against one light curve.

        This is the common researcher workflow: evaluate multiple candidate ephemerides
        (e.g., multi-planet systems or alternate periods) using the same underlying
        light curve and metadata.

        Returns:
            (bundles, summary_rows)
        """
        bundles: list[VettingBundleResult] = []
        summary: list[dict[str, Any]] = []

        for i, candidate in enumerate(candidates):
            bundle = self.run(
                lc,
                candidate,
                stellar=stellar,
                tpf=tpf,
                network=network,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                tic_id=tic_id,
                context=context,
            )
            bundles.append(bundle)
            summary.append(
                _bundle_summary_row(candidate_index=i, candidate=candidate, bundle=bundle)
            )

        return bundles, summary

    def describe(
        self,
        *,
        tpf: Any | None = None,
        network: bool = False,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        tic_id: int | None = None,
    ) -> dict[str, Any]:
        """Describe what the pipeline will do given inputs.

        Returns a dict describing which checks will run vs skip.
        """
        # Mock inputs for requirement checking
        inputs = CheckInputs(
            lc=None,  # type: ignore[arg-type]
            candidate=None,  # type: ignore[arg-type]
            tpf=tpf,
            network=network,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
        )

        checks = self._get_checks_to_run()
        will_run = []
        will_skip = []

        for check in checks:
            met, reason = self._check_requirements_met(check.requirements, inputs)
            info = {
                "id": check.id,
                "name": check.name,
                "tier": check.tier.value,
            }
            if met:
                will_run.append(info)
            else:
                will_skip.append({**info, "reason": reason})

        return {
            "will_run": will_run,
            "will_skip": will_skip,
            "total_checks": len(checks),
        }


def _bundle_summary_row(
    *,
    candidate_index: int,
    candidate: TransitCandidate,
    bundle: VettingBundleResult,
) -> dict[str, Any]:
    flags_top: list[str] = []
    for r in bundle.results:
        for f in r.flags:
            if f not in flags_top:
                flags_top.append(f)
        if len(flags_top) >= 10:
            break

    return {
        "candidate_index": int(candidate_index),
        "period_days": float(candidate.period),
        "t0_btjd": float(candidate.t0),
        "duration_hours": float(candidate.duration_hours),
        "depth_ppm": float(candidate.depth) * 1e6,
        "n_ok": sum(1 for r in bundle.results if r.status == "ok"),
        "n_skipped": sum(1 for r in bundle.results if r.status == "skipped"),
        "n_error": sum(1 for r in bundle.results if r.status == "error"),
        "flags_top": flags_top,
        "runtime_ms": bundle.provenance.get("duration_ms"),
    }


_EXTRA_TO_MODULES: dict[str, tuple[str, ...]] = {
    "tls": ("transitleastsquares", "numba"),
    "fit": ("emcee", "arviz"),
    "wotan": ("wotan",),
    "ldtk": ("ldtk",),
    "triceratops": ("lightkurve", "pytransit"),
    "mlx": ("mlx",),
    "exovetter": ("exovetter",),
}


def _optional_dep_available(extra: str) -> bool:
    modules = _EXTRA_TO_MODULES.get(extra, (extra,))
    return all(find_spec(m) is not None for m in modules)


def _format_skip_note(reason: str) -> str:
    if reason.startswith("EXTRA_MISSING:"):
        extra = reason.split(":", 1)[1]
        return (
            f"Check skipped: missing optional dependency '{extra}'. "
            f"Install with: pip install 'tess-vetter[{extra}]'"
        )
    return f"Check skipped: {reason}"


def list_checks(registry: CheckRegistry | None = None) -> list[dict[str, Any]]:
    """List all available checks.

    Args:
        registry: Registry to query. Defaults to global registry.

    Returns:
        List of check info dicts.
    """
    reg = get_default_registry() if registry is None else registry
    return [
        {
            "id": c.id,
            "name": c.name,
            "tier": c.tier.value,
            "requirements": {
                "needs_tpf": c.requirements.needs_tpf,
                "needs_network": c.requirements.needs_network,
                "needs_ra_dec": c.requirements.needs_ra_dec,
                "needs_tic_id": c.requirements.needs_tic_id,
                "needs_stellar": c.requirements.needs_stellar,
                "optional_deps": list(c.requirements.optional_deps),
            },
            "citations": c.citations,
        }
        for c in reg.list()
    ]


def describe_checks(registry: CheckRegistry | None = None) -> str:
    """Get a human-readable description of available checks.

    Args:
        registry: Registry to query. Defaults to global registry.

    Returns:
        Formatted string describing all checks.
    """
    checks = list_checks(registry)
    if not checks:
        return "No checks registered."

    lines = ["Available vetting checks:", ""]
    for c in checks:
        lines.append(f"  {c['id']}: {c['name']}")
        lines.append(f"       Tier: {c['tier']}")
        if c["citations"]:
            lines.append(f"       Citations: {', '.join(c['citations'])}")
        lines.append("")

    return "\n".join(lines)
