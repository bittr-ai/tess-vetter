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
    ['V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07']
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bittr_tess_vetter.validation.registry import CheckRegistry


def register_lc_only_checks(registry: CheckRegistry) -> None:
    """Register LC-only checks V01-V05.

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


def register_all_defaults(registry: CheckRegistry) -> None:
    """Register all default vetting checks V01-V07.

    This registers:
    - V01-V05: LC-only checks (always available)
    - V06-V07: Catalog checks (need network + metadata)

    Args:
        registry: CheckRegistry to register checks with.

    Example:
        >>> from bittr_tess_vetter.validation import get_default_registry
        >>> from bittr_tess_vetter.validation.register_defaults import register_all_defaults
        >>> registry = get_default_registry()
        >>> register_all_defaults(registry)
        >>> len(registry)
        7
    """
    register_lc_only_checks(registry)
    register_catalog_checks(registry)


__all__ = [
    "register_lc_only_checks",
    "register_catalog_checks",
    "register_all_defaults",
]
