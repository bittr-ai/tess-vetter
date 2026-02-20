"""
Feature extraction module for tess-vetter.

This module provides the schema and builder for converting raw vetting
pipeline outputs into ML-ready feature rows.

Key Components
--------------
FeatureConfig : dataclass
    Configuration for feature extraction (frozen/immutable).
EnrichedRow : TypedDict
    Schema for ML feature rows with all feature families.
RawEvidencePacket : TypedDict
    Intermediate representation of vetting pipeline outputs.
build_features : function
    Deterministic feature extraction from raw evidence.

Version Tracking
----------------
FEATURE_SCHEMA_VERSION tracks the schema version. Any semantic change
to feature definitions requires a version bump.

Example
-------
>>> from tess_vetter.features import (
...     FeatureConfig,
...     EnrichedRow,
...     RawEvidencePacket,
...     build_features,
...     FEATURE_SCHEMA_VERSION,
... )
>>> config = FeatureConfig(enable_pixel_timeseries=True)
>>> FEATURE_SCHEMA_VERSION
'6.0.0'
"""

from .builder import build_features
from .config import FeatureConfig
from .evidence import RawEvidencePacket
from .schema import FEATURE_SCHEMA_VERSION, EnrichedRow

__all__ = [
    "FeatureConfig",
    "EnrichedRow",
    "FEATURE_SCHEMA_VERSION",
    "RawEvidencePacket",
    "build_features",
]
