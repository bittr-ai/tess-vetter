# Module Review: `api/joint_inference.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this matters

This file defines the host-facing import surface for joint-inference schema/types. The numerical/physics logic lives in `compute/joint_inference_schemas.py` and `compute/joint_likelihood.py`; this wrapper is primarily about API stability.

## File: `api/joint_inference.py`

- Pure re-export wrapper over:
  - `compute.joint_inference_schemas.InferenceMode`
  - `compute.joint_inference_schemas.SectorEvidence`
  - `compute.joint_inference_schemas.JointInferenceResult`
  - `compute.joint_inference_schemas.create_joint_result_from_sectors`
  - `compute.joint_inference_schemas.joint_result_to_dict`
- No unit conversions or additional logic.

## Cross-references

- `working_docs/physics_audit/modules/completed/api_pixel_prf.md` (joint inference schemas already used/audited in the pixel PRF stack)

## Fixes / follow-ups

No physics correctness issues identified in this wrapper.

