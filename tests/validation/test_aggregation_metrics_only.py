from __future__ import annotations

from bittr_tess_vetter.domain.detection import Disposition, ValidationResult, Verdict, VetterCheckResult
from bittr_tess_vetter.validation.base import compute_verdict, generate_summary


def _res(check_id: str, *, passed: bool | None, confidence: float = 0.9) -> VetterCheckResult:
    return VetterCheckResult(
        id=check_id,
        name=f"check_{check_id}",
        passed=passed,
        confidence=confidence,
        details={},
    )


def test_compute_verdict_unknown_is_not_failure() -> None:
    lc = [_res("V01", passed=True), _res("V02", passed=None)]
    verdict = compute_verdict(lc_checks=lc, catalog_checks=[])
    assert verdict == Verdict.WARN


def test_compute_verdict_reject_requires_explicit_failures() -> None:
    lc = [_res("V01", passed=None), _res("V02", passed=None), _res("V03", passed=None)]
    verdict = compute_verdict(lc_checks=lc, catalog_checks=[])
    assert verdict == Verdict.WARN


def test_compute_verdict_reject_still_triggers_on_two_lc_failures() -> None:
    lc = [_res("V01", passed=False), _res("V02", passed=False), _res("V03", passed=None)]
    verdict = compute_verdict(lc_checks=lc, catalog_checks=[])
    assert verdict == Verdict.REJECT


def test_generate_summary_mentions_metrics_only_when_warned() -> None:
    checks = [_res("V01", passed=True), _res("V02", passed=None)]
    summary = generate_summary(Disposition.UNCERTAIN, Verdict.WARN, checks)
    assert "metrics-only" in summary
    assert "deferred" in summary


def test_validation_result_counts_do_not_treat_unknown_as_failed() -> None:
    checks = [_res("V01", passed=True), _res("V02", passed=None), _res("V03", passed=False)]
    vr = ValidationResult(
        disposition=Disposition.UNCERTAIN,
        verdict=Verdict.WARN,
        checks=checks,
        summary="x",
    )
    assert vr.n_passed == 1
    assert vr.n_failed == 1
    assert vr.n_unknown == 1
    assert vr.failed_checks == ["V03"]
    assert vr.unknown_checks == ["V02"]

