from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 3
    backoff_seconds: float = 0.25
    jitter: float = 0.10
    cap_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        if self.jitter < 0:
            raise ValueError("jitter must be >= 0")
        if self.cap_seconds < 0:
            raise ValueError("cap_seconds must be >= 0")

    def backoff_delay(
        self,
        retry_index: int,
        *,
        use_jitter: bool = True,
        random_value: float | None = None,
    ) -> float:
        """Return delay (seconds) before the next retry.

        retry_index is 1-based: first retry after first failure uses 1.
        """
        if retry_index < 1:
            raise ValueError("retry_index must be >= 1")

        delay = self.backoff_seconds * (2 ** (retry_index - 1))
        delay = min(delay, self.cap_seconds)

        if not use_jitter or self.jitter == 0:
            return delay

        sample = random.random() if random_value is None else random_value
        if sample < 0 or sample > 1:
            raise ValueError("random_value must be between 0 and 1")

        factor = 1 + (2 * sample - 1) * self.jitter
        jittered = delay * factor
        return max(0.0, min(jittered, self.cap_seconds))


__all__ = ["RetryPolicy"]
