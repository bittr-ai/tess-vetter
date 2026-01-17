"""Single-check execution helpers (researcher-facing, policy-free).

The core pipeline APIs (`vet_candidate`, `VettingPipeline`) can already run a
subset of checks, but tutorials and interactive analysis often want:

- Run one check at a time (to inspect metrics)
- Reuse converted inputs (LC/candidate) across many calls
- Keep outputs structured and policy-free

This module provides a small session object plus convenience functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bittr_tess_vetter.api.pipeline import PipelineConfig, VettingPipeline
from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    LightCurve,
    StellarParams,
    TPFStamp,
)
from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.validation.register_defaults import (
    register_all_defaults,
    register_extended_defaults,
)
from bittr_tess_vetter.validation.registry import CheckRegistry


def _build_registry(*, preset: str) -> CheckRegistry:
    registry = CheckRegistry()
    if str(preset).lower() == "extended":
        register_extended_defaults(registry)
    else:
        register_all_defaults(registry)
    return registry


def _to_internal_candidate(candidate: Candidate) -> TransitCandidate:
    depth = candidate.depth if candidate.depth is not None else 0.001
    return TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=depth,
        snr=0.0,
    )


@dataclass(frozen=True)
class VettingSession:
    """Reusable, single-target vetting context.

    A session converts inputs once and can run one or many check IDs, returning
    canonical `CheckResult` objects (status=ok/skipped/error).
    """

    lc: Any
    candidate: TransitCandidate
    stellar: StellarParams | None
    tpf: TPFStamp | None
    network: bool
    ra_deg: float | None
    dec_deg: float | None
    tic_id: int | None
    context: dict[str, Any]
    registry: CheckRegistry
    pipeline_config: PipelineConfig

    @classmethod
    def from_api(
        cls,
        *,
        lc: LightCurve,
        candidate: Candidate,
        stellar: StellarParams | None = None,
        tpf: TPFStamp | None = None,
        network: bool = False,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        tic_id: int | None = None,
        context: dict[str, Any] | None = None,
        preset: str = "default",
        registry: CheckRegistry | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> VettingSession:
        if registry is None:
            registry = _build_registry(preset=preset)
        cfg = pipeline_config or PipelineConfig()
        internal_lc = lc.to_internal(tic_id=tic_id or 0)
        internal_candidate = _to_internal_candidate(candidate)
        return cls(
            lc=internal_lc,
            candidate=internal_candidate,
            stellar=stellar,
            tpf=tpf,
            network=bool(network),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
            context=context or {},
            registry=registry,
            pipeline_config=cfg,
        )

    def run(self, check_id: str) -> CheckResult:
        """Run exactly one check ID and return its result."""
        return self.run_many([check_id])[0]

    def run_many(self, check_ids: list[str]) -> list[CheckResult]:
        """Run a specific list of check IDs (in order) and return results."""
        try:
            pipeline = VettingPipeline(checks=list(check_ids), registry=self.registry, config=self.pipeline_config)
        except KeyError as e:
            raise ValueError(
                f"Unknown check id {str(e)}. If this is an extended check, pass preset='extended' "
                "when constructing the session, or provide an explicit registry."
            ) from e

        bundle = pipeline.run(
            self.lc,
            self.candidate,
            stellar=self.stellar,
            tpf=self.tpf,
            network=self.network,
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            tic_id=self.tic_id,
            context=self.context,
        )
        return list(bundle.results)


def run_check(
    *,
    lc: LightCurve,
    candidate: Candidate,
    check_id: str,
    stellar: StellarParams | None = None,
    tpf: TPFStamp | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    context: dict[str, Any] | None = None,
    preset: str = "default",
    registry: CheckRegistry | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> CheckResult:
    """Convenience wrapper to run one check and return its `CheckResult`."""
    session = VettingSession.from_api(
        lc=lc,
        candidate=candidate,
        stellar=stellar,
        tpf=tpf,
        network=network,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tic_id=tic_id,
        context=context,
        preset=preset,
        registry=registry,
        pipeline_config=pipeline_config,
    )
    return session.run(check_id)


def run_checks(
    *,
    lc: LightCurve,
    candidate: Candidate,
    check_ids: list[str],
    stellar: StellarParams | None = None,
    tpf: TPFStamp | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    context: dict[str, Any] | None = None,
    preset: str = "default",
    registry: CheckRegistry | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> list[CheckResult]:
    """Convenience wrapper to run a list of checks and return results in order."""
    session = VettingSession.from_api(
        lc=lc,
        candidate=candidate,
        stellar=stellar,
        tpf=tpf,
        network=network,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tic_id=tic_id,
        context=context,
        preset=preset,
        registry=registry,
        pipeline_config=pipeline_config,
    )
    return session.run_many(list(check_ids))


__all__ = ["VettingSession", "run_check", "run_checks"]
