"""Canonical verdict helpers for report summary assembly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tess_vetter.validation.result_schema import CheckResult, VettingBundleResult

_RED_NOISE_CAVEAT = "RED_NOISE_CAVEAT"


def _has_red_noise_phrase(value: str) -> bool:
    lowered = value.lower()
    return "red noise" in lowered or "red_noise" in lowered


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_strings(nested)


def _check_has_red_noise_caveat(check: CheckResult) -> bool:
    inflation = check.metrics.get("red_noise_inflation")
    if isinstance(inflation, (float, int)) and float(inflation) > 1.0:
        return True

    for item in check.flags:
        if _has_red_noise_phrase(item):
            return True
    for item in check.notes:
        if _has_red_noise_phrase(item):
            return True
    if isinstance(check.raw, dict):
        for item in _iter_strings(check.raw):
            if _has_red_noise_phrase(item):
                return True
    return False


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _failed_ids(bundle: VettingBundleResult | None, checks: dict[str, CheckResult]) -> list[str]:
    if bundle is not None:
        return _sorted_unique(bundle.failed_check_ids)
    return _sorted_unique(check.id for check in checks.values() if check.status == "error")


def _flagged_or_skipped_ids(
    bundle: VettingBundleResult | None,
    checks: dict[str, CheckResult],
) -> list[str]:
    flagged = {
        check.id
        for check in checks.values()
        if check.status == "skipped" or (check.status == "ok" and len(check.flags) > 0)
    }
    if bundle is not None:
        flagged.update(bundle.unknown_check_ids)
    return _sorted_unique(flagged)


def _all_ok(bundle: VettingBundleResult | None, checks: dict[str, CheckResult]) -> bool:
    if bundle is not None:
        return bool(bundle.all_passed)
    return all(check.status == "ok" for check in checks.values())


def _build_caveats(
    bundle: VettingBundleResult | None,
    checks: dict[str, CheckResult],
    noise_summary: dict[str, Any] | None,
) -> list[str]:
    caveats: list[str] = []
    noise_flags = noise_summary.get("flags") if isinstance(noise_summary, dict) else None
    if isinstance(noise_flags, list) and "RED_NOISE_ELEVATED" in noise_flags:
        caveats.append(_RED_NOISE_CAVEAT)

    if any(_check_has_red_noise_caveat(check) for check in checks.values()):
        caveats.append(_RED_NOISE_CAVEAT)

    if bundle is not None and any(_has_red_noise_phrase(w) for w in bundle.warnings):
        caveats.append(_RED_NOISE_CAVEAT)

    return _sorted_unique(caveats)


def build_summary_verdict(
    *,
    bundle: VettingBundleResult | None,
    checks: dict[str, CheckResult],
    noise_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build canonical verdict fields for report summary."""
    failed_ids = _failed_ids(bundle, checks)
    flagged_or_skipped_ids = _flagged_or_skipped_ids(bundle, checks)
    all_ok = _all_ok(bundle, checks)

    if failed_ids:
        verdict = f"CHECK_FAILED:{','.join(failed_ids)}"
    elif all_ok and not flagged_or_skipped_ids:
        verdict = "ALL_CHECKS_PASSED"
    else:
        suffix = ",".join(flagged_or_skipped_ids) if flagged_or_skipped_ids else "0"
        verdict = f"CHECKS_FLAGGED:{suffix}"

    caveats = _build_caveats(bundle, checks, noise_summary)
    if caveats:
        verdict = f"{verdict}_WITH_CAVEATS"

    verdict_source = "$.summary.bundle_summary" if bundle is not None else "$.summary.checks"
    return {
        "verdict": verdict,
        "verdict_source": verdict_source,
        "caveats": caveats,
    }
