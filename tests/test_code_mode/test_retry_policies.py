from __future__ import annotations

import pytest

from tess_vetter.code_mode.retry import RetryPolicy, TransientExhaustionError, retry_transient
from tess_vetter.code_mode.retry.wrappers import wrap_with_transient_retry


def test_backoff_schedule_without_jitter_is_deterministic() -> None:
    policy = RetryPolicy(attempts=5, backoff_seconds=0.5, jitter=0.25, cap_seconds=2.0)
    schedule = [policy.backoff_delay(i, use_jitter=False) for i in range(1, policy.attempts)]
    assert schedule == [0.5, 1.0, 2.0, 2.0]


def test_retry_transient_uses_backoff_and_eventually_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def _op() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("try again")
        return "ok"

    policy = RetryPolicy(attempts=4, backoff_seconds=0.2, jitter=0.5, cap_seconds=1.0)
    result = retry_transient(_op, policy=policy, sleep=sleeps.append, use_jitter=False)

    assert result == "ok"
    assert calls["count"] == 3
    assert sleeps == [0.2, 0.4]


def test_retry_transient_raises_deterministic_transient_exhaustion_payload() -> None:
    sleeps: list[float] = []

    def _op() -> str:
        raise TimeoutError("temporary outage")

    policy = RetryPolicy(attempts=3, backoff_seconds=0.1, jitter=0.4, cap_seconds=0.2)
    with pytest.raises(TransientExhaustionError) as exc_info:
        retry_transient(_op, policy=policy, sleep=sleeps.append, use_jitter=False)

    payload = exc_info.value.payload
    assert payload["code"] == "TRANSIENT_EXHAUSTION"
    assert payload["message"] == "Transient retry attempts exhausted."
    assert payload["retryable"] is False
    assert payload["details"] == {
        "attempts": 3,
        "backoff_seconds": 0.1,
        "jitter": 0.4,
        "cap_seconds": 0.2,
        "last_exception_type": "TimeoutError",
        "last_exception_text": "temporary outage",
    }
    assert sleeps == [0.1, 0.2]


def test_retry_transient_repeated_transient_failures_eventually_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def _op() -> str:
        calls["count"] += 1
        if calls["count"] < 6:
            raise ConnectionError("upstream busy")
        return "recovered"

    policy = RetryPolicy(attempts=6, backoff_seconds=0.1, jitter=0.4, cap_seconds=0.4)
    result = retry_transient(_op, policy=policy, sleep=sleeps.append, use_jitter=False)

    assert result == "recovered"
    assert calls["count"] == 6
    assert sleeps == [0.1, 0.2, 0.4, 0.4, 0.4]


def test_retry_transient_non_transient_exceptions_are_passed_through_without_retry() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def _op() -> str:
        calls["count"] += 1
        raise ValueError("bad payload")

    policy = RetryPolicy(attempts=5, backoff_seconds=0.1, jitter=0.4, cap_seconds=0.2)
    with pytest.raises(ValueError, match="bad payload"):
        retry_transient(_op, policy=policy, sleep=sleeps.append, use_jitter=False)

    assert calls["count"] == 1
    assert sleeps == []


def test_wrap_with_transient_retry_surfaces_deterministic_transient_exhaustion() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def _op() -> str:
        calls["count"] += 1
        raise ConnectionError("upstream unavailable")

    wrapped = wrap_with_transient_retry(
        _op,
        policy=RetryPolicy(attempts=2, backoff_seconds=0.1, jitter=0.4, cap_seconds=0.2),
        sleep=sleeps.append,
        use_jitter=False,
    )

    with pytest.raises(TransientExhaustionError) as exc_info:
        wrapped()

    payload = exc_info.value.payload
    assert payload["code"] == "TRANSIENT_EXHAUSTION"
    assert payload["message"] == "Transient retry attempts exhausted."
    assert payload["retryable"] is False
    assert payload["details"] == {
        "attempts": 2,
        "backoff_seconds": 0.1,
        "jitter": 0.4,
        "cap_seconds": 0.2,
        "last_exception_type": "ConnectionError",
        "last_exception_text": "upstream unavailable",
    }
    assert calls["count"] == 2
    assert sleeps == [0.1]
