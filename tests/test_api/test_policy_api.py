from __future__ import annotations

from bittr_tess_vetter.api.policy import (
    Disposition,
    PolicyConfig,
    apply_policy,
    apply_policy_to_check,
)
from bittr_tess_vetter.api.types import CheckResult, VettingBundleResult


def test_apply_policy_to_check_v01_warn_and_fail_thresholds() -> None:
    base = CheckResult(
        id="V01",
        name="odd_even_depth",
        passed=None,
        confidence=0.9,
        details={
            "n_odd_transits": 3,
            "n_even_transits": 3,
            "delta_sigma": 3.0,
            "rel_diff": 0.2,
            "_metrics_only": True,
        },
    )
    warn = apply_policy_to_check(base)
    assert warn.disposition is Disposition.WARN

    fail = apply_policy_to_check(
        CheckResult(
            id="V01",
            name="odd_even_depth",
            passed=None,
            confidence=0.9,
            details={
                "n_odd_transits": 3,
                "n_even_transits": 3,
                "delta_sigma": 4.1,
                "rel_diff": 0.2,
                "_metrics_only": True,
            },
        )
    )
    assert fail.disposition is Disposition.FAIL


def test_apply_policy_to_check_v02_secondary_eclipse() -> None:
    ok = apply_policy_to_check(
        CheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=None,
            confidence=0.8,
            details={"secondary_depth_sigma": 2.0, "_metrics_only": True},
        )
    )
    assert ok.disposition is Disposition.PASS

    warn = apply_policy_to_check(
        CheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=None,
            confidence=0.8,
            details={"secondary_depth_sigma": 3.5, "_metrics_only": True},
        )
    )
    assert warn.disposition is Disposition.WARN

    fail = apply_policy_to_check(
        CheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=None,
            confidence=0.8,
            details={"secondary_depth_sigma": 5.2, "_metrics_only": True},
        )
    )
    assert fail.disposition is Disposition.FAIL


def test_apply_policy_marks_missing_dependency_unknown() -> None:
    result = apply_policy_to_check(
        CheckResult(
            id="V11",
            name="modshift",
            passed=None,
            confidence=0.0,
            details={"warnings": ["EXOVETTER_IMPORT_ERROR"], "_metrics_only": True},
        )
    )
    assert result.disposition is Disposition.UNKNOWN


def test_apply_policy_bundle_includes_policy_provenance() -> None:
    bundle = VettingBundleResult(
        results=[
            CheckResult(
                id="V02",
                name="secondary_eclipse",
                passed=None,
                confidence=0.8,
                details={"secondary_depth_sigma": 2.0, "_metrics_only": True},
            )
        ],
        provenance={"package_version": "test"},
        warnings=[],
    )
    out = apply_policy(bundle, config=PolicyConfig(secondary_warn_sigma=10.0))
    assert out.provenance["policy"]["name"] == "default"
    assert out.results[0].disposition is Disposition.PASS


def test_vetting_bundle_apply_policy_helper() -> None:
    bundle = VettingBundleResult(
        results=[
            CheckResult(
                id="V02",
                name="secondary_eclipse",
                passed=None,
                confidence=0.8,
                details={"secondary_depth_sigma": 2.0, "_metrics_only": True},
            )
        ],
        provenance={"package_version": "test"},
        warnings=[],
    )
    out = bundle.apply_policy(config=PolicyConfig(secondary_warn_sigma=10.0))
    assert out.results[0].disposition is Disposition.PASS
