from __future__ import annotations

import math

BJD_TO_BTJD_OFFSET = 2_457_000.0
_ABSOLUTE_BJD_THRESHOLD = 2_400_000.0


def normalize_epoch_to_btjd(epoch: float | None) -> float | None:
    """Normalize a transit epoch to BTJD.

    Expects either:
    - absolute BJD values (~2.4M+), or
    - already-offset BTJD values (~0-10k).
    """
    if epoch is None:
        return None
    value = float(epoch)
    if not math.isfinite(value):
        raise ValueError(f"Epoch must be finite, got {value!r}")
    if value >= _ABSOLUTE_BJD_THRESHOLD:
        return value - BJD_TO_BTJD_OFFSET
    return value


def looks_like_absolute_bjd(epoch: float) -> bool:
    """Best-effort guard for mislabeled absolute-BJD values."""
    return math.isfinite(epoch) and float(epoch) >= _ABSOLUTE_BJD_THRESHOLD
