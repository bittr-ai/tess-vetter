from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from tess_vetter.code_mode.retry.policies import RetryPolicy

T = TypeVar("T")

DEFAULT_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
)


class TransientExhaustionError(RuntimeError):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(str(payload["message"]))
        self.payload = payload


def make_transient_exhaustion_payload(
    policy: RetryPolicy,
    *,
    last_exception: BaseException,
) -> dict[str, object]:
    return {
        "code": "TRANSIENT_EXHAUSTION",
        "message": "Transient retry attempts exhausted.",
        "retryable": False,
        "details": {
            "attempts": policy.attempts,
            "backoff_seconds": policy.backoff_seconds,
            "jitter": policy.jitter,
            "cap_seconds": policy.cap_seconds,
            "last_exception_type": type(last_exception).__name__,
            "last_exception_text": str(last_exception),
        },
    }


def retry_transient(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    transient_exceptions: tuple[type[BaseException], ...] = DEFAULT_TRANSIENT_EXCEPTIONS,
    sleep: Callable[[float], None] = time.sleep,
    use_jitter: bool = True,
) -> T:
    """Execute operation with exponential backoff for transient exceptions."""
    for attempt in range(1, policy.attempts + 1):
        try:
            return operation()
        except transient_exceptions as exc:
            if attempt >= policy.attempts:
                payload = make_transient_exhaustion_payload(policy, last_exception=exc)
                raise TransientExhaustionError(payload) from exc
            delay = policy.backoff_delay(attempt, use_jitter=use_jitter)
            sleep(delay)


__all__ = [
    "DEFAULT_TRANSIENT_EXCEPTIONS",
    "TransientExhaustionError",
    "make_transient_exhaustion_payload",
    "retry_transient",
]
