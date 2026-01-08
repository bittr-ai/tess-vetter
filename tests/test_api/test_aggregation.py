from __future__ import annotations

from bittr_tess_vetter.api import (
    AggregateResult,
    CheckResult,
    UnknownPolicy,
    aggregate_checks,
    checks_to_evidence_items,
)


def test_aggregate_checks_warns_on_unknown_by_default() -> None:
    checks = [
        CheckResult(id="V01", name="odd_even_depth", passed=None, confidence=0.9, details={}),
        CheckResult(id="V03", name="duration_consistency", passed=None, confidence=0.9, details={}),
    ]
    agg = aggregate_checks(checks)
    assert isinstance(agg, AggregateResult)
    assert agg.verdict == "WARN"
    assert agg.n_unknown == 2


def test_aggregate_checks_ignore_unknown_can_pass() -> None:
    checks = [
        CheckResult(id="V01", name="odd_even_depth", passed=None, confidence=0.9, details={}),
        CheckResult(id="V03", name="duration_consistency", passed=True, confidence=0.9, details={}),
    ]
    agg = aggregate_checks(checks, unknown_policy=UnknownPolicy.IGNORE)
    assert agg.verdict == "PASS"
    assert agg.n_unknown == 1
    assert agg.n_passed == 1


def test_checks_to_evidence_items_is_jsonable() -> None:
    checks = [
        CheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=None,
            confidence=0.5,
            details={"_metrics_only": True, "x": 1},
        )
    ]
    items = checks_to_evidence_items(checks)
    assert isinstance(items, list)
    assert items[0]["id"] == "V02"
    assert items[0]["metrics_only"] is True

