from __future__ import annotations

from tess_vetter.code_mode.migration import OpsLibraryIdDiff, compare_legacy_seed_ids
from tess_vetter.code_mode.migration.ops_library_migration import (
    summarize_legacy_seed_coverage_delta,
)


def test_compare_legacy_seed_ids_reports_deterministic_missing_added_sets() -> None:
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=[
            "code_mode.primitive.fold",
            "code_mode.golden.vet_candidate",
            "code_mode.primitive.fold",
        ],
        discovered_modular_ids=[
            "code_mode.primitive.fold",
            "code_mode.golden.run_periodogram",
            "code_mode.golden.run_periodogram",
        ],
    )

    assert diff == OpsLibraryIdDiff(
        missing_ids=("code_mode.golden.vet_candidate",),
        added_ids=("code_mode.golden.run_periodogram",),
        renamed_ids=(),
    )


def test_compare_legacy_seed_ids_honors_explicit_rename_hints() -> None:
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=[
            "code_mode.golden.vet_candidate",
            "code_mode.primitive.fold",
        ],
        discovered_modular_ids=[
            "code_mode.golden.vet_target",
            "code_mode.primitive.fold",
        ],
        renamed_id_hints={
            "code_mode.golden.vet_candidate": "code_mode.golden.vet_target",
        },
    )

    assert diff.missing_ids == ()
    assert diff.added_ids == ()
    assert diff.renamed_ids == (
        ("code_mode.golden.vet_candidate", "code_mode.golden.vet_target"),
    )


def test_compare_legacy_seed_ids_inferrs_unique_leaf_rename() -> None:
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=[
            "code_mode.golden.vet_candidate",
            "code_mode.primitive.fold",
        ],
        discovered_modular_ids=[
            "code_mode.adapters.vet_candidate",
            "code_mode.primitive.fold",
        ],
    )

    assert diff.missing_ids == ()
    assert diff.added_ids == ()
    assert diff.renamed_ids == (
        ("code_mode.golden.vet_candidate", "code_mode.adapters.vet_candidate"),
    )


def test_compare_legacy_seed_ids_does_not_infer_ambiguous_leaf_rename() -> None:
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=[
            "code_mode.golden.run",
            "code_mode.primitive.run",
        ],
        discovered_modular_ids=[
            "code_mode.adapters.run",
        ],
    )

    assert diff.missing_ids == (
        "code_mode.golden.run",
        "code_mode.primitive.run",
    )
    assert diff.added_ids == ("code_mode.adapters.run",)
    assert diff.renamed_ids == ()


def test_compare_legacy_seed_ids_report_payload_is_deterministic() -> None:
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=[
            "code_mode.zeta.keep",
            "code_mode.alpha.rename_me",
            "code_mode.beta.missing",
        ],
        discovered_modular_ids=[
            "code_mode.adapters.rename_me",
            "code_mode.theta.added",
            "code_mode.zeta.keep",
        ],
    )

    assert diff.as_dict() == {
        "missing_ids": ("code_mode.beta.missing",),
        "added_ids": ("code_mode.theta.added",),
        "renamed_ids": (("code_mode.alpha.rename_me", "code_mode.adapters.rename_me"),),
    }


def test_ops_library_id_diff_tier_prefix_breakdown_is_deterministic() -> None:
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=[
            "code_mode.golden.beta_missing",
            "code_mode.primitive.alpha_missing",
            "code_mode.alpha.keep",
        ],
        discovered_modular_ids=[
            "code_mode.delta.beta_added",
            "code_mode.gamma.alpha_added",
            "code_mode.alpha.keep",
        ],
    )

    assert diff.missing_by_tier_prefix() == (
        ("code_mode.golden", ("code_mode.golden.beta_missing",)),
        ("code_mode.primitive", ("code_mode.primitive.alpha_missing",)),
    )
    assert diff.added_by_tier_prefix() == (
        ("code_mode.delta", ("code_mode.delta.beta_added",)),
        ("code_mode.gamma", ("code_mode.gamma.alpha_added",)),
    )


def test_summarize_legacy_seed_coverage_delta_reports_counts_and_breakdown() -> None:
    summary = summarize_legacy_seed_coverage_delta(
        legacy_seed_ids=[
            "code_mode.alpha.keep",
            "code_mode.golden.old_one",
            "code_mode.primitive.old_two",
            "code_mode.golden.old_one",
        ],
        discovered_modular_ids=[
            "code_mode.alpha.keep",
            "code_mode.adapters.new_one",
            "code_mode.adapters.new_two",
            "code_mode.adapters.new_two",
        ],
        unavailable_operation_ids=[
            "code_mode.adapters.new_two",
            "code_mode.adapters.not_present",
            "code_mode.adapters.new_two",
        ],
    )

    assert summary["coverage_delta_counts"] == {
        "legacy_seed_total": 3,
        "expanded_discovery_total": 3,
        "missing_total": 2,
        "added_total": 2,
        "renamed_total": 0,
        "unavailable_total": 1,
        "unavailable_added_total": 1,
        "net_new_total": 0,
    }
    assert summary["missing_by_tier_prefix"] == (
        ("code_mode.golden", ("code_mode.golden.old_one",)),
        ("code_mode.primitive", ("code_mode.primitive.old_two",)),
    )
    assert summary["added_by_tier_prefix"] == (
        (
            "code_mode.adapters",
            ("code_mode.adapters.new_one", "code_mode.adapters.new_two"),
        ),
    )
    assert summary["unavailable_ids"] == ("code_mode.adapters.new_two",)
