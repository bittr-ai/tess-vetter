"""Vetting pipeline for orchestrating check execution.

This module provides the VettingPipeline class which runs multiple
vetting checks and aggregates their results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    VettingCheck,
    get_default_registry,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    VettingBundleResult,
    skipped_result,
)

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData


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
        self._registry = registry or get_default_registry()
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
                    notes=[f"Check skipped: {reason}"],
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
                from bittr_tess_vetter.validation.result_schema import error_result
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


def list_checks(registry: CheckRegistry | None = None) -> list[dict[str, Any]]:
    """List all available checks.

    Args:
        registry: Registry to query. Defaults to global registry.

    Returns:
        List of check info dicts.
    """
    reg = registry or get_default_registry()
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
        if c['citations']:
            lines.append(f"       Citations: {', '.join(c['citations'])}")
        lines.append("")

    return "\n".join(lines)
