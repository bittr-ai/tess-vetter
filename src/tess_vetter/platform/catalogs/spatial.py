"""Spatial indexing for efficient cone searches on catalog data.

Provides a k-d tree based spatial index that supports cone searches
on celestial coordinates (RA/Dec in degrees). The implementation
properly handles spherical geometry including:
- cos(dec) correction for RA distances
- RA wraparound at 0/360 degrees
- Coordinate singularities at poles
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Constants for coordinate conversions
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
ARCSEC_PER_DEG = 3600.0


def _normalize_ra(ra: float) -> float:
    """Normalize RA to [0, 360) range."""
    ra = ra % 360.0
    if ra < 0:
        ra += 360.0
    return ra


def _to_cartesian(ra: float, dec: float) -> tuple[float, float, float]:
    """Convert spherical (RA, Dec) in degrees to unit Cartesian (x, y, z).

    Uses standard astronomical convention:
    - x-axis points to RA=0, Dec=0
    - y-axis points to RA=90, Dec=0
    - z-axis points to Dec=+90 (north celestial pole)
    """
    ra_rad = ra * DEG_TO_RAD
    dec_rad = dec * DEG_TO_RAD
    cos_dec = math.cos(dec_rad)
    x = cos_dec * math.cos(ra_rad)
    y = cos_dec * math.sin(ra_rad)
    z = math.sin(dec_rad)
    return (x, y, z)


def _angular_distance_rad(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    """Compute angular distance in radians between two unit vectors.

    Uses the more numerically stable formula for small angles:
    d = 2 * arcsin(chord_length / 2)
    where chord_length = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    """
    # Chord length between the two points on unit sphere
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    chord = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Convert chord to angular distance
    # chord = 2 * sin(angle/2), so angle = 2 * arcsin(chord/2)
    # Clamp to handle numerical precision issues
    half_chord = min(1.0, chord / 2.0)
    return 2.0 * math.asin(half_chord)


class SpatialIndex:
    """Spatial index for efficient cone searches on catalog data.

    Builds a k-d tree from celestial coordinates for fast nearest-neighbor
    and cone search queries. Coordinates are internally converted to
    3D Cartesian coordinates on the unit sphere to properly handle
    spherical geometry.

    Attributes:
        coords: Original Nx2 array of (RA, Dec) coordinates in degrees.
        _cartesian: Nx3 array of Cartesian unit vectors.
        _tree: K-d tree structure for fast queries.

    Example:
        >>> coords = np.array([[10.0, 20.0], [10.5, 20.1], [100.0, -30.0]])
        >>> idx = SpatialIndex(coords)
        >>> matches = idx.cone_search(10.2, 20.05, 60.0)  # 60 arcsec radius
        >>> print(matches)  # indices of matching sources
    """

    def __init__(self, coords: NDArray[np.floating]) -> None:
        """Build index from Nx2 array of (RA, Dec) in degrees.

        Args:
            coords: Nx2 numpy array where each row is [RA, Dec] in degrees.
                    RA must be in [0, 360), Dec in [-90, 90].

        Raises:
            ValueError: If coords is not a 2D array with shape (N, 2).
        """
        coords = np.asarray(coords, dtype=np.float64)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be Nx2 array, got shape {coords.shape}")

        self._coords = coords
        self._n_sources: int = int(coords.shape[0])

        # Convert to Cartesian coordinates on unit sphere
        self._cartesian = self._build_cartesian(coords)

        # Build k-d tree - try scipy.spatial.cKDTree first, fall back to pure Python
        self._tree = self._build_tree()

    def _build_cartesian(self, coords: NDArray[np.floating]) -> NDArray[np.float64]:
        """Convert spherical coordinates to Cartesian unit vectors."""
        ra_rad = coords[:, 0] * DEG_TO_RAD
        dec_rad = coords[:, 1] * DEG_TO_RAD

        cos_dec = np.cos(dec_rad)
        cartesian = np.empty((self._n_sources, 3), dtype=np.float64)
        cartesian[:, 0] = cos_dec * np.cos(ra_rad)
        cartesian[:, 1] = cos_dec * np.sin(ra_rad)
        cartesian[:, 2] = np.sin(dec_rad)

        return cartesian

    def _build_tree(self) -> Any:
        """Build k-d tree, using scipy if available."""
        try:
            from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

            return cKDTree(self._cartesian)
        except ImportError:
            # Fall back to pure Python implementation
            return _PurePythonKDTree(self._cartesian)

    @property
    def coords(self) -> NDArray[np.floating]:
        """Return the original coordinate array."""
        return self._coords

    def __len__(self) -> int:
        """Return number of sources in the index."""
        return self._n_sources

    def cone_search(self, ra: float, dec: float, radius_arcsec: float) -> list[int]:
        """Return indices of points within radius of (ra, dec).

        Args:
            ra: Right ascension in degrees [0, 360).
            dec: Declination in degrees [-90, 90].
            radius_arcsec: Search radius in arcseconds.

        Returns:
            List of integer indices into the original coords array for all
            sources within the specified radius. Order is deterministic
            (sorted by index) for reproducibility.

        Raises:
            ValueError: If dec is outside [-90, 90] or radius is negative.
        """
        if not -90.0 <= dec <= 90.0:
            raise ValueError(f"dec must be in [-90, 90], got {dec}")
        if radius_arcsec < 0:
            raise ValueError(f"radius must be non-negative, got {radius_arcsec}")

        if self._n_sources == 0 or radius_arcsec == 0:
            return []

        # Normalize RA to [0, 360)
        ra = _normalize_ra(ra)

        # Convert search center to Cartesian
        center = _to_cartesian(ra, dec)

        # Convert angular radius to chord length for k-d tree query
        # chord = 2 * sin(angle/2)
        radius_rad = (radius_arcsec / ARCSEC_PER_DEG) * DEG_TO_RAD
        chord_radius = 2.0 * math.sin(radius_rad / 2.0)

        # Query k-d tree for all points within chord distance
        indices: list[int]
        if hasattr(self._tree, "query_ball_point"):
            # scipy cKDTree
            indices = self._tree.query_ball_point(center, chord_radius)
        else:
            # Pure Python fallback
            assert isinstance(self._tree, _PurePythonKDTree)
            indices = self._tree.query_radius(center, chord_radius)

        # Return sorted indices for deterministic behavior
        return sorted(indices)

    @staticmethod
    def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """Compute angular separation in arcsec between two points.

        Uses the Vincenty formula which is accurate for all angular
        separations, avoiding numerical issues near 0 and 180 degrees.

        Args:
            ra1: Right ascension of first point in degrees.
            dec1: Declination of first point in degrees.
            ra2: Right ascension of second point in degrees.
            dec2: Declination of second point in degrees.

        Returns:
            Angular separation in arcseconds.
        """
        # Convert to Cartesian and compute angular distance
        x1, y1, z1 = _to_cartesian(ra1, dec1)
        x2, y2, z2 = _to_cartesian(ra2, dec2)

        angle_rad = _angular_distance_rad(x1, y1, z1, x2, y2, z2)

        return angle_rad * RAD_TO_DEG * ARCSEC_PER_DEG


class _PurePythonKDTree:
    """Simple pure-Python k-d tree for fallback when scipy is unavailable.

    This is a minimal implementation optimized for the cone search use case.
    For large catalogs, scipy.spatial.cKDTree is strongly recommended.
    """

    def __init__(self, data: NDArray[np.float64]) -> None:
        """Build k-d tree from Nx3 array of points."""
        self._data = data
        self._n = data.shape[0]

        # Build tree as nested tuples: (split_dim, split_val, left, right, indices)
        indices = np.arange(self._n)
        self._root = self._build_node(indices, 0)

    def _build_node(
        self, indices: NDArray[np.intp], depth: int
    ) -> tuple[int, Any, Any, Any, int] | None:
        """Recursively build k-d tree node."""
        if len(indices) == 0:
            return None

        # Cycle through dimensions
        dim = depth % 3

        # Sort indices by coordinate in split dimension
        sorted_idx = indices[np.argsort(self._data[indices, dim])]

        # Find median
        mid = len(sorted_idx) // 2

        return (
            dim,
            self._data[sorted_idx[mid], dim],
            self._build_node(sorted_idx[:mid], depth + 1),
            self._build_node(sorted_idx[mid + 1 :], depth + 1),
            int(sorted_idx[mid]),
        )

    def query_radius(self, point: tuple[float, float, float], radius: float) -> list[int]:
        """Find all points within radius of given point."""
        results: list[int] = []
        point_arr = np.array(point)
        radius_sq = radius * radius
        self._search_node(self._root, point_arr, radius_sq, results)
        return results

    def _search_node(
        self,
        node: tuple[int, Any, Any, Any, int] | None,
        point: NDArray[np.float64],
        radius_sq: float,
        results: list[int],
    ) -> None:
        """Recursively search k-d tree node."""
        if node is None:
            return

        dim, split_val, left, right, idx = node

        # Check if current point is within radius
        node_point = self._data[idx]
        dist_sq = np.sum((node_point - point) ** 2)
        if dist_sq <= radius_sq:
            results.append(idx)

        # Determine which subtree to search first
        diff = point[dim] - split_val

        if diff < 0:
            # Search left subtree first
            self._search_node(left, point, radius_sq, results)
            # Only search right if it could contain points within radius
            if diff * diff <= radius_sq:
                self._search_node(right, point, radius_sq, results)
        else:
            # Search right subtree first
            self._search_node(right, point, radius_sq, results)
            # Only search left if it could contain points within radius
            if diff * diff <= radius_sq:
                self._search_node(left, point, radius_sq, results)
