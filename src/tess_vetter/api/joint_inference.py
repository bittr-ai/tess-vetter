"""Joint inference API surface (host-facing).

This module provides a stable facade for multi-sector joint inference schema
types and helpers so host applications don't need to import from internal
`compute.*` modules.
"""

from __future__ import annotations

from tess_vetter.compute.joint_inference_schemas import (  # noqa: F401
    InferenceMode,
    JointInferenceResult,
    SectorEvidence,
    create_joint_result_from_sectors,
    joint_result_to_dict,
)

__all__ = [
    "InferenceMode",
    "SectorEvidence",
    "JointInferenceResult",
    "create_joint_result_from_sectors",
    "joint_result_to_dict",
]
