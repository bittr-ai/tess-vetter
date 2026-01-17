"""Default check registration for the vetting pipeline.

This module provides functions to register the default vetting checks
with a CheckRegistry, enabling the VettingPipeline to discover and
execute them.

Usage:
    >>> from bittr_tess_vetter.validation import get_default_registry
    >>> from bittr_tess_vetter.validation.register_defaults import register_all_defaults
    >>> registry = get_default_registry()
    >>> register_all_defaults(registry)
    >>> registry.list_ids()
    ['V01', 'V02', 'V03', 'V04', 'V05', 'V13', 'V15', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12']
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bittr_tess_vetter.validation.registry import CheckRegistry


def register_lc_only_checks(registry: CheckRegistry) -> None:
    """Register LC-only checks V01-V05 plus false-alarm checks V13 and V15.

    These checks require only light curve data and ephemeris.

    Args:
        registry: CheckRegistry to register checks with.
    """
    from bittr_tess_vetter.validation.checks_lc_wrapped import register_lc_checks

    register_lc_checks(registry)


def register_catalog_checks(registry: CheckRegistry) -> None:
    """Register catalog checks V06-V07.

    These checks require network access for catalog queries.

    Args:
        registry: CheckRegistry to register checks with.
    """
    from bittr_tess_vetter.validation.checks_catalog_wrapped import (
        register_catalog_checks as _register,
    )

    _register(registry)


def register_pixel_checks(registry: CheckRegistry) -> None:
    """Register pixel checks V08-V10.

    These checks require TPF (pixel) data.

    Args:
        registry: CheckRegistry to register checks with.
    """
    from bittr_tess_vetter.validation.checks_pixel_wrapped import (
        register_pixel_checks as _register,
    )

    _register(registry)


def register_exovetter_checks(registry: CheckRegistry) -> None:
    """Register exovetter checks V11-V12.

    These checks use external vetting algorithms.

    Args:
        registry: CheckRegistry to register checks with.
    """
    from bittr_tess_vetter.validation.checks_exovetter_wrapped import (
        register_exovetter_checks as _register,
    )

    _register(registry)


def register_modshift_uniqueness_check(registry: CheckRegistry) -> None:
    """Register ModShift uniqueness check V11b.

    This is an independent implementation of ModShift with properly-scaled
    Fred and MS1-MS6 metrics. Runs alongside V11 for A/B comparison.

    Args:
        registry: CheckRegistry to register checks with.
    """
    from bittr_tess_vetter.validation.checks_modshift_uniqueness_wrapped import (
        register_modshift_uniqueness_check as _register,
    )

    _register(registry)


def register_all_defaults(registry: CheckRegistry) -> None:
    """Register all default vetting checks V01-V12 plus V11b.

    This registers:
    - V01-V05: LC-only checks (always available)
    - V06-V07: Catalog checks (need network + metadata)
    - V08-V10: Pixel checks (need TPF data)
    - V11-V12: Exovetter checks (external algorithms)
    - V11b: ModShift uniqueness (independent impl, A/B comparison with V11)

    Args:
        registry: CheckRegistry to register checks with.

    Example:
        >>> from bittr_tess_vetter.validation import get_default_registry
        >>> from bittr_tess_vetter.validation.register_defaults import register_all_defaults
        >>> registry = get_default_registry()
        >>> register_all_defaults(registry)
        >>> len(registry)
        13
    """
    register_lc_only_checks(registry)
    register_catalog_checks(registry)
    register_pixel_checks(registry)
    register_exovetter_checks(registry)
    register_modshift_uniqueness_check(registry)


def register_extended_defaults(registry: CheckRegistry) -> None:
    """Register the default checks plus extended metrics-only diagnostics.

    This preserves the existing semantics of `register_all_defaults` while
    providing an opt-in, richer check set for host applications that want
    additional guardrail metrics without embedding policy thresholds.
    """
    register_all_defaults(registry)

    from bittr_tess_vetter.validation.checks_extended_wrapped import register_extended_checks

    register_extended_checks(registry)


__all__ = [
    "register_lc_only_checks",
    "register_catalog_checks",
    "register_pixel_checks",
    "register_exovetter_checks",
    "register_modshift_uniqueness_check",
    "register_all_defaults",
    "register_extended_defaults",
]
