# Module Review: `api/bls_like_search.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this matters

This file defines the host-facing import surface for the BLS-like search functions used by discovery tooling. The primary physics risk here is accidental API drift (renaming, re-export omissions) rather than numerical mistakes (those live in `compute/bls_like_search.py`).

## File: `api/bls_like_search.py`

- Pure re-export wrapper over:
  - `compute.bls_like_search.BlsLikeCandidate`
  - `compute.bls_like_search.BlsLikeSearchResult`
  - `compute.bls_like_search.bls_like_search_numpy`
  - `compute.bls_like_search.bls_like_search_numpy_top_k`
- No unit conversions or additional logic.

## Cross-references

- Compute implementation audit: `working_docs/physics_audit/modules/completed/compute_bls_like_search.md`

## Fixes / follow-ups

No physics correctness issues identified in this wrapper.

