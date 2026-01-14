"""Check registry for extensible vetting pipeline.

This module provides the infrastructure for registering, discovering,
and running vetting checks in a structured way.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.validation.result_schema import CheckResult


class CheckTier(Enum):
    """Classification of check types by their data requirements."""

    LC_ONLY = "lc_only"  # Only needs light curve + ephemeris
    CATALOG = "catalog"  # Needs network access for catalog queries
    PIXEL = "pixel"  # Needs TPF/pixel data
    EXOVETTER = "exovetter"  # External vetter integration
    AUX = "aux"  # Auxiliary/optional checks


@dataclass(frozen=True)
class CheckRequirements:
    """Data requirements for a vetting check.

    Attributes:
        needs_tpf: Requires TPF (pixel) data.
        needs_network: Requires network access for catalogs.
        needs_ra_dec: Requires sky coordinates.
        needs_tic_id: Requires TIC identifier.
        needs_stellar: Requires stellar parameters (soft requirement).
        optional_deps: List of pip extras required (e.g., ["tls"], ["triceratops"]).
    """

    needs_tpf: bool = False
    needs_network: bool = False
    needs_ra_dec: bool = False
    needs_tic_id: bool = False
    needs_stellar: bool = False
    optional_deps: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CheckInputs:
    """Container for all possible check inputs.

    Checks receive this container and extract what they need.
    """

    lc: LightCurveData
    candidate: TransitCandidate
    stellar: Any | None = None
    tpf: Any | None = None
    network: bool = False
    ra_deg: float | None = None
    dec_deg: float | None = None
    tic_id: int | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckConfig:
    """Configuration for check execution.

    Attributes:
        timeout_seconds: Maximum execution time.
        random_seed: Seed for reproducibility.
        extra_params: Check-specific parameters.
    """

    timeout_seconds: float | None = None
    random_seed: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VettingCheck(Protocol):
    """Protocol defining a vetting check.

    All vetting checks must implement this interface to be registered
    and executed by the VettingPipeline.
    """

    @property
    def id(self) -> str:
        """Unique check identifier (e.g., 'V01')."""
        ...

    @property
    def name(self) -> str:
        """Human-readable check name."""
        ...

    @property
    def tier(self) -> CheckTier:
        """Check classification tier."""
        ...

    @property
    def requirements(self) -> CheckRequirements:
        """Data requirements for this check."""
        ...

    @property
    def citations(self) -> list[str]:
        """Academic citations for this check's methodology."""
        ...

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the check and return a structured result."""
        ...


class CheckRegistry:
    """Registry for vetting checks.

    Provides registration, lookup, and enumeration of available checks.

    Example:
        >>> registry = CheckRegistry()
        >>> registry.register(MyCheck())
        >>> check = registry.get("V01")
        >>> all_checks = registry.list()
    """

    def __init__(self) -> None:
        self._checks: dict[str, VettingCheck] = {}

    def register(self, check: VettingCheck) -> None:
        """Register a check with the registry.

        Args:
            check: A VettingCheck implementation.

        Raises:
            ValueError: If a check with the same ID is already registered.
        """
        if check.id in self._checks:
            raise ValueError(f"Check '{check.id}' is already registered")
        self._checks[check.id] = check

    def get(self, id: str) -> VettingCheck:
        """Get a check by ID.

        Args:
            id: The check identifier.

        Returns:
            The registered check.

        Raises:
            KeyError: If no check with that ID is registered.
        """
        if id not in self._checks:
            raise KeyError(f"No check registered with ID '{id}'")
        return self._checks[id]

    def list(self) -> builtins.list[VettingCheck]:
        """List all registered checks.

        Returns:
            List of all registered checks, sorted by ID.
        """
        return sorted(self._checks.values(), key=lambda c: c.id)

    def list_by_tier(self, tier: CheckTier) -> builtins.list[VettingCheck]:
        """List checks filtered by tier.

        Args:
            tier: The tier to filter by.

        Returns:
            List of checks in the specified tier, sorted by ID.
        """
        return sorted(
            [c for c in self._checks.values() if c.tier == tier],
            key=lambda c: c.id,
        )

    def list_ids(self) -> builtins.list[str]:
        """List all registered check IDs.

        Returns:
            Sorted list of check IDs.
        """
        return sorted(self._checks.keys())

    def __contains__(self, id: str) -> bool:
        """Check if an ID is registered."""
        return id in self._checks

    def __len__(self) -> int:
        """Return number of registered checks."""
        return len(self._checks)


# Global default registry
DEFAULT_REGISTRY = CheckRegistry()


def get_default_registry() -> CheckRegistry:
    """Get the default global check registry.

    Returns:
        The default CheckRegistry instance.
    """
    # Lazy-load default checks so API helpers like `list_checks()` work
    # without requiring callers to remember to register defaults.
    if len(DEFAULT_REGISTRY) == 0:
        from bittr_tess_vetter.validation.register_defaults import register_all_defaults

        register_all_defaults(DEFAULT_REGISTRY)
    return DEFAULT_REGISTRY
