from __future__ import annotations

from tess_vetter.code_mode.migration import OpsLibraryIdDiff, compare_legacy_seed_ids


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
