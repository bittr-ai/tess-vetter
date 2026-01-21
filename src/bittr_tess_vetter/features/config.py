"""Feature extraction configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for feature extraction pipeline.

    This dataclass is frozen (immutable) to ensure config consistency
    throughout a pipeline run. All defaults are designed for batch
    processing of existing data.

    Attributes
    ----------
    allow_20s : bool
        Allow 20-second cadence data (default: False for 2-min only).
    network_ok : bool
        Allow network requests during processing (default: False).
    bulk_mode : bool
        Optimize for bulk processing (default: True).
    no_download : bool
        Skip downloading any new data (default: False).
    local_data_path : str | None
        Path to local data directory containing tic{tic_id}/ subdirectories
        with sector*_pdcsap.csv files. If set, light curves are loaded from
        local files instead of MAST queries (default: None).
    enable_t0_refine : bool
        Enable T0 refinement optimization (default: False).
    t0_refine_max_minutes : float
        Maximum time budget for T0 refinement in minutes.
    t0_refine_min_delta_score : float
        Minimum score improvement to accept refined T0.
    enable_pixel_timeseries : bool
        Enable pixel-level timeseries analysis (default: True).
    enable_host_plausibility : bool
        Enable host star plausibility checks (default: True).
    enable_ghost_reliability : bool
        Enable ghost/scattered light reliability features (default: True).
    enable_sector_quality : bool
        Enable sector quality metrics (default: True).
    enable_diff_image_score : bool
        Enable difference image scoring (default: False).
    enable_ephemeris_match : bool
        Enable ephemeris matching features (default: False).
    enable_sector_consistency : bool
        Enable cross-sector consistency checks (default: False).
    enable_transit_uniformity : bool
        Enable transit uniformity analysis (default: False).
    enable_transit_shape_categorical : bool
        Enable categorical transit shape classification (default: False).
    """

    # Input handling
    allow_20s: bool = False
    network_ok: bool = False
    bulk_mode: bool = True
    no_download: bool = False
    local_data_path: str | None = None

    # Optional operations
    enable_t0_refine: bool = False
    t0_refine_max_minutes: float = 60.0
    t0_refine_min_delta_score: float = 2.0

    # Feature families
    enable_pixel_timeseries: bool = True
    enable_host_plausibility: bool = True
    enable_ghost_reliability: bool = True
    enable_sector_quality: bool = True
    enable_diff_image_score: bool = False
    enable_ephemeris_match: bool = False
    enable_sector_consistency: bool = False
    enable_transit_uniformity: bool = False
    enable_transit_shape_categorical: bool = False
