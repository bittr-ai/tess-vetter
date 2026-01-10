"""Response caps enforcement utilities.

This module provides utilities for enforcing maximum sizes on response data
to prevent oversized payloads in API responses. Each capping function:
- Truncates lists to a specified maximum size
- Logs a warning when truncation occurs
- Returns the truncated list

These caps are essential for:
- Preventing memory exhaustion in clients
- Maintaining reasonable response times
- Ensuring predictable API behavior
- Graceful degradation under high-cardinality results

Example:
    >>> from bittr_tess_vetter.utils.caps import cap_top_k, cap_neighbors
    >>> results = list(range(100))
    >>> capped = cap_top_k(results, max_items=10)
    >>> len(capped)
    10
    >>> # Warning logged: "Truncated response from 100 to 10 items"
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

# Module logger
logger = logging.getLogger(__name__)

# Type variable for generic list capping
T = TypeVar("T")

# Default caps for different response types
DEFAULT_TOP_K_CAP = 50
DEFAULT_VARIANT_SUMMARIES_CAP = 50
DEFAULT_NEIGHBORS_CAP = 10
DEFAULT_PLOTS_CAP = 6


def _cap_list(
    items: list[T],
    max_items: int,
    context: str,
    cap_name: str,
) -> list[T]:
    """Internal helper to cap a list with logging.

    Args:
        items: List to potentially truncate.
        max_items: Maximum number of items to return.
        context: Additional context for the log message (e.g., target ID).
        cap_name: Name of the cap being applied (for logging).

    Returns:
        The original list if length <= max_items, otherwise truncated list.
    """
    if len(items) <= max_items:
        return items

    # Log warning about truncation
    context_str = f" ({context})" if context else ""
    logger.warning(
        "Truncated %s response from %d to %d items%s",
        cap_name,
        len(items),
        max_items,
        context_str,
    )

    return items[:max_items]


def cap_top_k(
    items: list[T],
    max_items: int = DEFAULT_TOP_K_CAP,
    context: str = "",
) -> list[T]:
    """Truncate list to max_items, log warning if truncated.

    Used for capping general top-k results like search results,
    candidate lists, and ranked matches.

    Args:
        items: List of items to potentially truncate.
        max_items: Maximum number of items to return. Default: 50.
        context: Additional context for the log message
            (e.g., "TIC 123456789" or "period search").

    Returns:
        The original list if length <= max_items, otherwise a new list
        containing only the first max_items elements.

    Example:
        >>> candidates = list(range(100))
        >>> capped = cap_top_k(candidates, max_items=10, context="BLS candidates")
        >>> len(capped)
        10
    """
    return _cap_list(items, max_items, context, "top_k")


def cap_variant_summaries(
    items: list[T],
    max_items: int = DEFAULT_VARIANT_SUMMARIES_CAP,
    context: str = "",
) -> list[T]:
    """Truncate variant summaries, log warning if truncated.

    Used for capping lists of variant/parameter combinations,
    detrending variants, or model variations.

    Args:
        items: List of variant summaries to potentially truncate.
        max_items: Maximum number of items to return. Default: 50.
        context: Additional context for the log message
            (e.g., "detrend variants" or "TIC 123456789").

    Returns:
        The original list if length <= max_items, otherwise a new list
        containing only the first max_items elements.

    Example:
        >>> variants = [{"window": w} for w in range(100)]
        >>> capped = cap_variant_summaries(variants, context="detrend")
        >>> len(capped)
        50
    """
    return _cap_list(items, max_items, context, "variant_summaries")


def cap_neighbors(
    items: list[T],
    max_items: int = DEFAULT_NEIGHBORS_CAP,
    context: str = "",
) -> list[T]:
    """Truncate neighbor list, log warning if truncated.

    Used for capping lists of nearby objects, spatial neighbors,
    or similar items from similarity search.

    Args:
        items: List of neighbors to potentially truncate.
        max_items: Maximum number of items to return. Default: 10.
        context: Additional context for the log message
            (e.g., "TIC 123456789" or "RA=180.0 Dec=45.0").

    Returns:
        The original list if length <= max_items, otherwise a new list
        containing only the first max_items elements.

    Example:
        >>> nearby = [{"tic_id": i, "dist": i * 0.1} for i in range(50)]
        >>> capped = cap_neighbors(nearby, context="TIC 123456789")
        >>> len(capped)
        10
    """
    return _cap_list(items, max_items, context, "neighbors")


def cap_plots(
    items: list[T],
    max_items: int = DEFAULT_PLOTS_CAP,
    context: str = "",
) -> list[T]:
    """Truncate plot list, log warning if truncated.

    Used for capping lists of plot data, images, or visualizations
    to prevent response bloat.

    Args:
        items: List of plots to potentially truncate.
        max_items: Maximum number of items to return. Default: 6.
        context: Additional context for the log message
            (e.g., "phase fold plots" or "TIC 123456789").

    Returns:
        The original list if length <= max_items, otherwise a new list
        containing only the first max_items elements.

    Example:
        >>> plots = [{"sector": s, "data": "..."} for s in range(20)]
        >>> capped = cap_plots(plots, context="multi-sector plots")
        >>> len(capped)
        6
    """
    return _cap_list(items, max_items, context, "plots")


# =============================================================================
# Tests (run with: python -m pytest src/astro_arc/utils/caps.py -v)
# =============================================================================

if __name__ == "__main__":
    import sys

    def run_tests() -> int:
        """Run all tests and return exit code."""
        failed = 0
        passed = 0
        log_messages: list[str] = []

        class TestHandler(logging.Handler):
            """Capture log messages for testing."""

            def emit(self, record: logging.LogRecord) -> None:
                log_messages.append(self.format(record))

        # Set up test logging
        handler = TestHandler()
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        def test(name: str, condition: bool, msg: str = "") -> None:
            nonlocal failed, passed
            if condition:
                print(f"  OK: {name}")
                passed += 1
            else:
                print(f"  ERR: {name} - {msg}")
                failed += 1

        def clear_logs() -> None:
            log_messages.clear()

        print("\n=== cap_top_k tests ===")

        # Test no truncation needed
        clear_logs()
        items = [1, 2, 3]
        result = cap_top_k(items, max_items=10)
        test("no truncation - same list", result == items)
        test("no truncation - no log", len(log_messages) == 0)

        # Test exact limit
        clear_logs()
        items = list(range(50))
        result = cap_top_k(items, max_items=50)
        test("exact limit - same list", result == items)
        test("exact limit - no log", len(log_messages) == 0)

        # Test truncation
        clear_logs()
        items = list(range(100))
        result = cap_top_k(items, max_items=10)
        test("truncation - correct length", len(result) == 10)
        test("truncation - first items", result == list(range(10)))
        test("truncation - logged warning", len(log_messages) == 1)
        test("truncation - log content", "100 to 10" in log_messages[0])

        # Test with context
        clear_logs()
        items = list(range(100))
        result = cap_top_k(items, max_items=5, context="BLS search")
        test("with context - correct length", len(result) == 5)
        test("with context - log has context", "BLS search" in log_messages[0])

        # Test default max_items
        clear_logs()
        items = list(range(60))
        result = cap_top_k(items)  # default is 50
        test("default cap - correct length", len(result) == 50)

        # Test empty list
        clear_logs()
        result = cap_top_k([], max_items=10)
        test("empty list", result == [])
        test("empty list - no log", len(log_messages) == 0)

        print("\n=== cap_variant_summaries tests ===")

        # Test basic truncation
        clear_logs()
        variant_items: list[Any] = [{"variant": i} for i in range(100)]
        result = cap_variant_summaries(variant_items, max_items=20)
        test("variant truncation", len(result) == 20)
        test("variant log", "variant_summaries" in log_messages[0])

        # Test default cap
        clear_logs()
        items = list(range(60))
        result = cap_variant_summaries(items)
        test("variant default cap", len(result) == 50)

        print("\n=== cap_neighbors tests ===")

        # Test basic truncation
        clear_logs()
        neighbor_items: list[Any] = [{"neighbor": i} for i in range(50)]
        result = cap_neighbors(neighbor_items, max_items=5)
        test("neighbor truncation", len(result) == 5)
        test("neighbor log", "neighbors" in log_messages[0])

        # Test default cap (10)
        clear_logs()
        items = list(range(20))
        result = cap_neighbors(items)
        test("neighbor default cap", len(result) == 10)

        print("\n=== cap_plots tests ===")

        # Test basic truncation
        clear_logs()
        plot_items: list[Any] = [{"plot": i} for i in range(20)]
        result = cap_plots(plot_items, max_items=4)
        test("plot truncation", len(result) == 4)
        test("plot log", "plots" in log_messages[0])

        # Test default cap (6)
        clear_logs()
        items = list(range(10))
        result = cap_plots(items)
        test("plot default cap", len(result) == 6)

        print("\n=== Edge cases ===")

        # Test preserves original list
        clear_logs()
        original = [1, 2, 3, 4, 5]
        result = cap_top_k(original, max_items=3)
        test("original unchanged", original == [1, 2, 3, 4, 5])
        test("result is new list", result is not original[:3])

        # Test various types in list
        clear_logs()
        mixed_items: list[Any] = ["a", "b", "c", 1, 2, 3, None, True]
        result = cap_top_k(mixed_items, max_items=4)
        test("mixed types", result == ["a", "b", "c", 1])

        # Test nested structures
        clear_logs()
        nested_items: list[Any] = [
            {"data": [1, 2, 3]},
            {"data": [4, 5, 6]},
            {"data": [7, 8, 9]},
        ]
        nested_result: list[Any] = cap_top_k(nested_items, max_items=2)
        test(
            "nested structures",
            nested_result == [{"data": [1, 2, 3]}, {"data": [4, 5, 6]}],
        )

        # Test max_items = 1
        clear_logs()
        items = list(range(10))
        result = cap_top_k(items, max_items=1)
        test("max_items 1", result == [0])

        # Test max_items = 0
        clear_logs()
        items = list(range(10))
        result = cap_top_k(items, max_items=0)
        test("max_items 0", result == [])

        # Test logging level is WARNING
        test("log level is warning", logger.level == logging.WARNING)

        # Cleanup
        logger.removeHandler(handler)

        # Summary
        print(f"\n=== Summary: {passed} passed, {failed} failed ===")
        return 0 if failed == 0 else 1

    sys.exit(run_tests())
