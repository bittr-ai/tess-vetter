# Physics Audit (Function-by-Function)

This folder is the working log for a **function-by-function astrophysics audit** of `bittr-tess-vetter`.

## Goals

- Verify the **physics correctness**, **unit conventions**, and **numerical stability** of each early-agent function first.
- Make it easy to answer: “Is this calculation trustworthy? Under what assumptions?”
- Add targeted **regression tests** for the highest-risk calculations.

## How to use this folder

- Start at `working_docs/physics_audit/INDEX.md` for the ordered review queue.
- Each module gets a review file under `modules/in_progress/` while it's being worked.
- Once a module has **zero unchecked boxes**, move it to `modules/completed/`.
- Cross-cutting conventions live in `CONVENTIONS.md` and `REVIEW_TEMPLATE.md`.

## Rules

- **Do not change algorithms “for elegance”** during the audit; only fix correctness issues or document limitations.
- Every change should be accompanied by either:
  - a new test, or
  - a documented rationale for why testing is infeasible (rare).

## Helper script

- Run `python working_docs/physics_audit/scripts/check_module_completion.py` to see which module notes still have unchecked boxes.
