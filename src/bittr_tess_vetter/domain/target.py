"""Target and stellar parameter models.

This module provides:
- StellarParameters: Stellar physical parameters from TIC
- Target: Complete target information including position and cross-matches
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
from pydantic import Field


class StellarParameters(FrozenModel):
    """Stellar parameters from TIC catalog.

    All fields are optional since TIC coverage varies.
    Used for transit duration consistency checks and stellar density calculations.

    Validation:
    - Physical parameters (teff, radius, mass, luminosity) must be non-negative
    - logg can be negative for giant stars
    - contamination ratio is non-negative (TIC can occasionally report values > 1)
    """

    teff: float | None = Field(default=None, ge=0, description="Effective temperature (K)")
    logg: float | None = Field(default=None, description="Surface gravity (log10 cm/s^2)")
    radius: float | None = Field(default=None, ge=0, description="Stellar radius (solar radii)")
    mass: float | None = Field(default=None, ge=0, description="Stellar mass (solar masses)")
    tmag: float | None = Field(default=None, description="TESS magnitude")
    contamination: float | None = Field(
        default=None,
        ge=0,
        description="Contamination ratio (nominally 0-1; TIC can report >1 in edge cases)",
    )
    luminosity: float | None = Field(
        default=None, ge=0, description="Luminosity (solar luminosities)"
    )
    metallicity: float | None = Field(default=None, description="[Fe/H] metallicity")

    def has_minimum_params(self) -> bool:
        """Check if minimum parameters for transit analysis are available."""
        return self.radius is not None and self.mass is not None

    def stellar_density_solar(self) -> float | None:
        """Compute stellar density in solar units.

        Returns:
            Density in solar densities, or None if mass/radius unavailable or invalid.
        """
        # Note: With Field(ge=0) validation, radius can't be negative at construction.
        # However, we still check for <= 0 for safety (handles 0 and any edge cases).
        if self.mass is None or self.radius is None or self.radius <= 0:
            return None
        return self.mass / (self.radius**3)


class Target(FrozenModel):
    """Complete target information for a TESS observation.

    Combines TIC ID with stellar parameters, astrometric data,
    and cross-match identifiers.
    """

    tic_id: int
    stellar: StellarParameters = Field(default_factory=lambda: StellarParameters())

    # Astrometric position
    ra: float | None = None  # Right ascension (degrees)
    dec: float | None = None  # Declination (degrees)
    pmra: float | None = None  # Proper motion in RA (mas/yr)
    pmdec: float | None = None  # Proper motion in Dec (mas/yr)
    distance_pc: float | None = None  # Distance in parsecs

    # Cross-match identifiers
    gaia_dr3_id: int | None = None
    twomass_id: str | None = None

    # Known planet disposition (if any)
    known_planet_count: int = 0
    toi_id: str | None = None  # TESS Object of Interest ID

    def has_position(self) -> bool:
        """Check if sky coordinates are available."""
        return self.ra is not None and self.dec is not None

    @classmethod
    def from_tic_response(cls, tic_id: int, tic_data: dict[str, Any]) -> Target:
        """Create Target from TIC catalog query response.

        Args:
            tic_id: TIC identifier
            tic_data: Dictionary from TIC query

        Returns:
            Target instance with available parameters
        """
        stellar = StellarParameters(
            teff=tic_data.get("Teff"),
            logg=tic_data.get("logg"),
            radius=tic_data.get("rad"),
            mass=tic_data.get("mass"),
            tmag=tic_data.get("Tmag"),
            contamination=tic_data.get("contratio"),
            luminosity=tic_data.get("lum"),
            metallicity=tic_data.get("MH"),
        )

        return cls(
            tic_id=tic_id,
            stellar=stellar,
            ra=tic_data.get("ra"),
            dec=tic_data.get("dec"),
            pmra=tic_data.get("pmRA"),
            pmdec=tic_data.get("pmDEC"),
            distance_pc=tic_data.get("d"),
            gaia_dr3_id=tic_data.get("GAIA"),
            twomass_id=tic_data.get("TWOMASS"),
        )
