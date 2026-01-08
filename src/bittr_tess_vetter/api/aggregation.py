"""Aggregation helpers for the public API.

Host applications often operate in metrics-only mode where `CheckResult.passed` is `None`.
These helpers provide a stable, supported rollup API that treats unknowns as unknowns
(not failures) and can optionally ignore them entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bittr_tess_vetter.api.types import CheckResult
from bittr_tess_vetter.domain.detection import Disposition, Verdict, VetterCheckResult
from bittr_tess_vetter.validation import base as validation_base


class UnknownPolicy(str, Enum):
    """How `passed=None` should affect the aggregate verdict."""

    WARN = "WARN"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class AggregateResult:
    verdict: str  # "PASS" | "WARN" | "REJECT"
    disposition: str  # "PLANET" | "UNCERTAIN" | "FALSE_POSITIVE"
    n_passed: int
    n_failed: int
    n_unknown: int
    failed_ids: list[str]
    unknown_ids: list[str]
    summary: str


def _to_internal(check: CheckResult) -> VetterCheckResult:
    return VetterCheckResult(
        id=check.id,
        name=check.name,
        passed=check.passed,
        confidence=check.confidence,
        details=dict(check.details),
    )


def aggregate_checks(
    checks: list[CheckResult],
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.WARN,
) -> AggregateResult:
    """Aggregate a list of `CheckResult` into a simple rollup.

    Notes:
    - Unknowns (`passed=None`) are never treated as failures.
    - For verdict computation, checks are tiered by ID using `validation.base` tier sets.
      Non-LC checks (pixel/exovetter/other) are treated as "catalog-tier" for aggregation.
    """
    failed_ids = [c.id for c in checks if c.passed is False]
    unknown_ids = [c.id for c in checks if c.passed is None]
    n_passed = sum(1 for c in checks if c.passed is True)
    n_failed = len(failed_ids)
    n_unknown = len(unknown_ids)

    internal = [_to_internal(c) for c in checks]
    if unknown_policy is UnknownPolicy.IGNORE:
        internal = [c for c in internal if c.passed is not None]

    lc_checks: list[VetterCheckResult] = []
    catalog_checks: list[VetterCheckResult] = []
    for c in internal:
        if c.id in validation_base.LC_ONLY_CHECKS:
            lc_checks.append(c)
        else:
            catalog_checks.append(c)

    verdict: Verdict = validation_base.compute_verdict(lc_checks=lc_checks, catalog_checks=catalog_checks)
    all_checks = lc_checks + catalog_checks
    disposition: Disposition = validation_base.compute_disposition(verdict, all_checks)
    summary = validation_base.generate_summary(disposition, verdict, all_checks)

    return AggregateResult(
        verdict=verdict.value,
        disposition=disposition.value,
        n_passed=n_passed,
        n_failed=n_failed,
        n_unknown=n_unknown,
        failed_ids=failed_ids,
        unknown_ids=unknown_ids,
        summary=summary,
    )

