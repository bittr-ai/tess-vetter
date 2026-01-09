from __future__ import annotations

import pytest
from pydantic import ValidationError

from bittr_tess_vetter.api.detection import (
    Detection,
    Disposition,
    PeriodogramPeak,
    PeriodogramResult,
    TransitCandidate,
    ValidationResult,
    Verdict,
    VetterCheckResult,
)


@pytest.fixture
def transit_candidate() -> TransitCandidate:
    return TransitCandidate(
        period=3.14159,
        t0=1325.504,
        duration_hours=2.5,
        depth=0.0007,
        snr=15.5,
    )


@pytest.fixture
def periodogram_peak() -> PeriodogramPeak:
    return PeriodogramPeak(
        period=3.14159,
        power=125.5,
        t0=1325.504,
        duration_hours=2.5,
        snr=15.5,
        fap=0.001,
    )


@pytest.fixture
def vetter_check_passed() -> VetterCheckResult:
    return VetterCheckResult(
        id="V01",
        name="Transit Shape",
        passed=True,
        confidence=0.95,
        details={"depth_ratio": 0.98, "shape_score": 0.92},
    )


@pytest.fixture
def vetter_check_failed() -> VetterCheckResult:
    return VetterCheckResult(
        id="V02",
        name="Odd-Even Depth",
        passed=False,
        confidence=0.88,
        details={"odd_depth": 0.001, "even_depth": 0.0015, "ratio": 0.67},
    )


class TestVerdictEnum:
    def test_verdict_values(self) -> None:
        assert Verdict.PASS.value == "PASS"
        assert Verdict.WARN.value == "WARN"
        assert Verdict.REJECT.value == "REJECT"

    def test_verdict_str_comparison(self) -> None:
        assert Verdict.PASS == "PASS"
        assert Verdict.WARN == "WARN"
        assert Verdict.REJECT == "REJECT"


class TestDispositionEnum:
    def test_disposition_values(self) -> None:
        assert Disposition.PLANET.value == "PLANET"
        assert Disposition.FALSE_POSITIVE.value == "FALSE_POSITIVE"
        assert Disposition.UNCERTAIN.value == "UNCERTAIN"


class TestPeriodogramPeak:
    def test_basic_creation(self, periodogram_peak: PeriodogramPeak) -> None:
        assert periodogram_peak.period == 3.14159
        assert periodogram_peak.power == 125.5
        assert periodogram_peak.t0 == 1325.504
        assert periodogram_peak.duration_hours == 2.5
        assert periodogram_peak.snr == 15.5
        assert periodogram_peak.fap == 0.001

    def test_optional_fields(self) -> None:
        peak = PeriodogramPeak(period=2.5, power=50.0, t0=1000.0)
        assert peak.duration_hours is None
        assert peak.snr is None
        assert peak.fap is None

    def test_is_significant_by_fap(self) -> None:
        significant = PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=0.001)
        not_significant = PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=0.05)

        assert significant.is_significant is True
        assert not_significant.is_significant is False

    def test_is_significant_by_snr(self) -> None:
        significant = PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, snr=10.0)
        not_significant = PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, snr=5.0)

        assert significant.is_significant is True
        assert not_significant.is_significant is False

    def test_is_significant_fallback_to_power(self) -> None:
        peak = PeriodogramPeak(period=1.0, power=10.0, t0=1000.0)
        assert peak.is_significant is True

    def test_period_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PeriodogramPeak(period=0.0, power=10.0, t0=1000.0)

        with pytest.raises(ValidationError):
            PeriodogramPeak(period=-1.0, power=10.0, t0=1000.0)

    def test_power_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            PeriodogramPeak(period=1.0, power=-1.0, t0=1000.0)

    def test_fap_range(self) -> None:
        PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=0.0)
        PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=1.0)
        PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=0.5)

        with pytest.raises(ValidationError):
            PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=-0.1)

        with pytest.raises(ValidationError):
            PeriodogramPeak(period=1.0, power=10.0, t0=1000.0, fap=1.5)


class TestTransitCandidate:
    def test_basic_creation(self, transit_candidate: TransitCandidate) -> None:
        assert transit_candidate.period == 3.14159
        assert transit_candidate.t0 == 1325.504
        assert transit_candidate.duration_hours == 2.5
        assert transit_candidate.depth == 0.0007
        assert transit_candidate.snr == 15.5

    def test_duration_days_property(self, transit_candidate: TransitCandidate) -> None:
        assert transit_candidate.duration_days == pytest.approx(2.5 / 24.0)

    def test_depth_validation(self) -> None:
        TransitCandidate(period=1.0, t0=1000.0, duration_hours=2.0, depth=0.001, snr=10.0)
        TransitCandidate(period=1.0, t0=1000.0, duration_hours=2.0, depth=1.0, snr=10.0)

        with pytest.raises(ValidationError):
            TransitCandidate(period=1.0, t0=1000.0, duration_hours=2.0, depth=0.0, snr=10.0)

        with pytest.raises(ValidationError):
            TransitCandidate(period=1.0, t0=1000.0, duration_hours=2.0, depth=1.5, snr=10.0)

        with pytest.raises(ValidationError):
            TransitCandidate(
                period=1.0,
                t0=1000.0,
                duration_hours=2.0,
                depth=-0.001,
                snr=10.0,
            )


class TestVetterCheckResult:
    def test_passed_check(self, vetter_check_passed: VetterCheckResult) -> None:
        assert vetter_check_passed.id == "V01"
        assert vetter_check_passed.passed is True
        assert vetter_check_passed.confidence == 0.95

    def test_failed_check(self, vetter_check_failed: VetterCheckResult) -> None:
        assert vetter_check_failed.id == "V02"
        assert vetter_check_failed.passed is False
        assert vetter_check_failed.confidence == 0.88

    def test_metrics_only(self) -> None:
        check = VetterCheckResult(id="V03", name="metrics-only", passed=None, confidence=0.5)
        assert check.passed is None

    def test_is_high_confidence(self) -> None:
        high_conf = VetterCheckResult(id="V01", name="Test", passed=True, confidence=0.9)
        low_conf = VetterCheckResult(id="V02", name="Test", passed=True, confidence=0.7)

        assert high_conf.is_high_confidence is True
        assert low_conf.is_high_confidence is False

    def test_id_pattern_validation(self) -> None:
        VetterCheckResult(id="V01", name="Test", passed=True, confidence=0.9)
        VetterCheckResult(id="PF01", name="Test", passed=True, confidence=0.9)
        VetterCheckResult(id="V99", name="Test", passed=True, confidence=0.9)

        with pytest.raises(ValidationError):
            VetterCheckResult(id="V1", name="Test", passed=True, confidence=0.9)

        with pytest.raises(ValidationError):
            VetterCheckResult(id="X01", name="Test", passed=True, confidence=0.9)

        with pytest.raises(ValidationError):
            VetterCheckResult(id="V100", name="Test", passed=True, confidence=0.9)

    def test_confidence_range(self) -> None:
        VetterCheckResult(id="V01", name="Test", passed=True, confidence=0.0)
        VetterCheckResult(id="V01", name="Test", passed=True, confidence=1.0)

        with pytest.raises(ValidationError):
            VetterCheckResult(id="V01", name="Test", passed=True, confidence=-0.1)

        with pytest.raises(ValidationError):
            VetterCheckResult(id="V01", name="Test", passed=True, confidence=1.5)

    def test_details_default(self) -> None:
        check = VetterCheckResult(id="V01", name="Test", passed=True, confidence=0.9)
        assert check.details == {}


class TestValidationResult:
    def test_basic_creation(
        self, vetter_check_passed: VetterCheckResult, vetter_check_failed: VetterCheckResult
    ) -> None:
        result = ValidationResult(
            disposition=Disposition.UNCERTAIN,
            verdict=Verdict.WARN,
            checks=[vetter_check_passed, vetter_check_failed],
            summary="One check passed, one failed",
        )
        assert result.disposition == Disposition.UNCERTAIN
        assert result.verdict == Verdict.WARN
        assert len(result.checks) == 2

    def test_n_passed(
        self, vetter_check_passed: VetterCheckResult, vetter_check_failed: VetterCheckResult
    ) -> None:
        result = ValidationResult(
            disposition=Disposition.UNCERTAIN,
            verdict=Verdict.WARN,
            checks=[vetter_check_passed, vetter_check_failed],
            summary="Test",
        )
        assert result.n_passed == 1

    def test_n_failed(
        self, vetter_check_passed: VetterCheckResult, vetter_check_failed: VetterCheckResult
    ) -> None:
        result = ValidationResult(
            disposition=Disposition.UNCERTAIN,
            verdict=Verdict.WARN,
            checks=[vetter_check_passed, vetter_check_failed],
            summary="Test",
        )
        assert result.n_failed == 1

    def test_metrics_only_counts(self) -> None:
        result = ValidationResult(
            disposition=Disposition.UNCERTAIN,
            verdict=Verdict.WARN,
            checks=[
                VetterCheckResult(id="V01", name="m1", passed=None, confidence=0.5),
                VetterCheckResult(id="V02", name="m2", passed=None, confidence=0.5),
                VetterCheckResult(id="V03", name="p", passed=True, confidence=0.9),
            ],
            summary="Test",
        )
        assert result.n_passed == 1
        assert result.n_failed == 0
        assert result.n_unknown == 2
        assert result.unknown_checks == ["V01", "V02"]

    def test_failed_checks(
        self, vetter_check_passed: VetterCheckResult, vetter_check_failed: VetterCheckResult
    ) -> None:
        result = ValidationResult(
            disposition=Disposition.FALSE_POSITIVE,
            verdict=Verdict.REJECT,
            checks=[vetter_check_passed, vetter_check_failed],
            summary="Test",
        )
        assert result.failed_checks == ["V02"]


class TestDetection:
    def test_basic_creation(self, transit_candidate: TransitCandidate) -> None:
        detection = Detection(
            candidate=transit_candidate,
            data_ref="lc:141914082:1:pdcsap",
            method="bls",
            rank=1,
        )
        assert detection.candidate == transit_candidate
        assert detection.data_ref == "lc:141914082:1:pdcsap"
        assert detection.method == "bls"
        assert detection.rank == 1
        assert detection.validation is None

    def test_is_validated_without_validation(self, transit_candidate: TransitCandidate) -> None:
        detection = Detection(
            candidate=transit_candidate,
            data_ref="lc:1:1:pdcsap",
            method="bls",
            rank=1,
        )
        assert detection.is_validated is False

    def test_is_validated_with_validation(
        self,
        transit_candidate: TransitCandidate,
        vetter_check_passed: VetterCheckResult,
    ) -> None:
        validation = ValidationResult(
            disposition=Disposition.PLANET,
            verdict=Verdict.PASS,
            checks=[vetter_check_passed],
            summary="Test",
        )
        detection = Detection(
            candidate=transit_candidate,
            data_ref="lc:1:1:pdcsap",
            method="bls",
            rank=1,
            validation=validation,
        )
        assert detection.is_validated is True

    def test_is_planet_candidate_without_validation(self, transit_candidate: TransitCandidate) -> None:
        detection = Detection(
            candidate=transit_candidate,
            data_ref="lc:1:1:pdcsap",
            method="bls",
            rank=1,
        )
        assert detection.is_planet_candidate is False

    def test_is_planet_candidate_planet(
        self,
        transit_candidate: TransitCandidate,
        vetter_check_passed: VetterCheckResult,
    ) -> None:
        validation = ValidationResult(
            disposition=Disposition.PLANET,
            verdict=Verdict.PASS,
            checks=[vetter_check_passed],
            summary="Planet candidate",
        )
        detection = Detection(
            candidate=transit_candidate,
            data_ref="lc:1:1:pdcsap",
            method="bls",
            rank=1,
            validation=validation,
        )
        assert detection.is_planet_candidate is True

    def test_is_planet_candidate_false_positive(
        self,
        transit_candidate: TransitCandidate,
        vetter_check_failed: VetterCheckResult,
    ) -> None:
        validation = ValidationResult(
            disposition=Disposition.FALSE_POSITIVE,
            verdict=Verdict.REJECT,
            checks=[vetter_check_failed],
            summary="False positive",
        )
        detection = Detection(
            candidate=transit_candidate,
            data_ref="lc:1:1:pdcsap",
            method="bls",
            rank=1,
            validation=validation,
        )
        assert detection.is_planet_candidate is False

    def test_method_validation(self, transit_candidate: TransitCandidate) -> None:
        Detection(candidate=transit_candidate, data_ref="lc:1:1:pdcsap", method="bls", rank=1)
        Detection(candidate=transit_candidate, data_ref="lc:1:1:pdcsap", method="ls", rank=1)
        Detection(candidate=transit_candidate, data_ref="lc:1:1:pdcsap", method="auto", rank=1)

        with pytest.raises(ValidationError):
            Detection(
                candidate=transit_candidate,
                data_ref="lc:1:1:pdcsap",
                method="invalid",  # type: ignore[arg-type]
                rank=1,
            )

    def test_rank_validation(self, transit_candidate: TransitCandidate) -> None:
        Detection(candidate=transit_candidate, data_ref="lc:1:1:pdcsap", method="bls", rank=1)
        Detection(candidate=transit_candidate, data_ref="lc:1:1:pdcsap", method="bls", rank=10)

        with pytest.raises(ValidationError):
            Detection(
                candidate=transit_candidate,
                data_ref="lc:1:1:pdcsap",
                method="bls",
                rank=0,
            )


class TestPeriodogramResult:
    def test_basic_creation(self, periodogram_peak: PeriodogramPeak) -> None:
        result = PeriodogramResult(
            data_ref="lc:1:1:pdcsap",
            method="bls",
            peaks=[periodogram_peak],
            best_period=3.14159,
            best_t0=1325.504,
            best_duration_hours=2.5,
            snr=15.5,
            fap=0.001,
            n_periods_searched=10000,
            period_range=(0.5, 15.0),
        )

        assert result.data_ref == "lc:1:1:pdcsap"
        assert result.method == "bls"
        assert result.best_period == 3.14159
        assert len(result.peaks) == 1

    def test_top_peak_with_peaks(self, periodogram_peak: PeriodogramPeak) -> None:
        result = PeriodogramResult(
            data_ref="lc:1:1:pdcsap",
            method="bls",
            peaks=[periodogram_peak],
            best_period=3.14159,
            best_t0=1325.504,
            n_periods_searched=10000,
            period_range=(0.5, 15.0),
        )
        assert result.top_peak == periodogram_peak

    def test_top_peak_no_peaks(self) -> None:
        result = PeriodogramResult(
            data_ref="lc:1:1:pdcsap",
            method="ls",
            peaks=[],
            best_period=1.0,
            best_t0=1000.0,
            n_periods_searched=10000,
            period_range=(0.5, 15.0),
        )
        assert result.top_peak is None

    def test_n_periods_searched_validation(self) -> None:
        with pytest.raises(ValidationError):
            PeriodogramResult(
                data_ref="lc:1:1:pdcsap",
                method="bls",
                peaks=[],
                best_period=1.0,
                best_t0=1000.0,
                n_periods_searched=0,
                period_range=(0.5, 15.0),
            )

        PeriodogramResult(
            data_ref="lc:1:1:pdcsap",
            method="tls",
            peaks=[],
            best_period=1.0,
            best_t0=1000.0,
            n_periods_searched=0,
            period_range=(0.5, 15.0),
        )

