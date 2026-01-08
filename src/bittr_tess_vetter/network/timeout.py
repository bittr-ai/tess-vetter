"""Network timeout utilities for external API calls.

This module provides timeout handling for network operations that don't
have built-in timeout support. Uses signal-based timeout on Unix systems.
"""

from __future__ import annotations

import logging
import signal
import sys
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default timeouts for various operations (in seconds)
MAST_QUERY_TIMEOUT = 60.0  # TIC queries, Catalogs.query_criteria, etc.
TRICERATOPS_INIT_TIMEOUT = 90.0  # Gaia DR3 TAP queries can be slow
TRICERATOPS_CALC_TIMEOUT = 120.0  # MCMC probability calculation


class NetworkTimeoutError(Exception):
    """Raised when a network operation exceeds its timeout."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"{operation} timed out after {timeout_seconds:.1f}s. "
            "The external service may be slow or unavailable."
        )


def _is_timeout_supported() -> bool:
    """Return True if SIGALRM timeouts are supported on this platform."""
    return hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


@contextmanager
def network_timeout(
    seconds: float,
    operation: str = "Network operation",
) -> Generator[None, None, None]:
    """Context manager that raises NetworkTimeoutError after timeout."""
    if not _is_timeout_supported():
        logger.debug(
            f"Signal-based timeout not supported on {sys.platform}, "
            f"skipping timeout for: {operation}"
        )
        yield
        return

    if seconds <= 0:
        raise ValueError(f"Timeout must be positive, got {seconds}")

    def handler(signum: int, frame: object) -> None:
        raise NetworkTimeoutError(operation, seconds)

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

