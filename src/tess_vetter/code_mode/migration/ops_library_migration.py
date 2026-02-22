"""Helpers for migrating legacy ops-library ids to modular registry ids."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class OpsLibraryIdDiff:
    """Deterministic diff between legacy seed ids and discovered modular ids."""

    missing_ids: tuple[str, ...]
    added_ids: tuple[str, ...]
    renamed_ids: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, tuple[str, ...] | tuple[tuple[str, str], ...]]:
        """Return a serialization-friendly deterministic report payload."""
        return {
            "missing_ids": self.missing_ids,
            "added_ids": self.added_ids,
            "renamed_ids": self.renamed_ids,
        }


def _normalize_ids(ids: Iterable[str]) -> tuple[str, ...]:
    """Normalize ids into sorted unique tuples for deterministic processing."""
    return tuple(sorted({str(value) for value in ids}))


def _leaf_token(operation_id: str) -> str:
    """Return the suffix token used for conservative rename inference."""
    return operation_id.rsplit(".", 1)[-1]


def _infer_renames(
    missing_ids: Sequence[str],
    added_ids: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Infer probable renames when missing/added ids share a unique leaf token."""
    missing_by_leaf: dict[str, list[str]] = defaultdict(list)
    added_by_leaf: dict[str, list[str]] = defaultdict(list)
    for legacy_id in missing_ids:
        missing_by_leaf[_leaf_token(legacy_id)].append(legacy_id)
    for modular_id in added_ids:
        added_by_leaf[_leaf_token(modular_id)].append(modular_id)

    inferred: list[tuple[str, str]] = []
    for leaf in sorted(set(missing_by_leaf) & set(added_by_leaf)):
        missing_for_leaf = sorted(missing_by_leaf[leaf])
        added_for_leaf = sorted(added_by_leaf[leaf])
        if len(missing_for_leaf) == 1 and len(added_for_leaf) == 1:
            inferred.append((missing_for_leaf[0], added_for_leaf[0]))
    return tuple(sorted(inferred))


def compare_legacy_seed_ids(
    legacy_seed_ids: Iterable[str],
    discovered_modular_ids: Iterable[str],
    *,
    renamed_id_hints: Mapping[str, str] | None = None,
) -> OpsLibraryIdDiff:
    """Compare legacy seed ids against discovered modular ids.

    The result is deterministic: outputs are sorted tuples, independent of input
    ordering and duplicate values.

    Args:
        legacy_seed_ids: Legacy ids from the monolithic seed registry.
        discovered_modular_ids: Ids discovered from modularized adapters.
        renamed_id_hints: Optional explicit mapping of legacy id -> modular id.
            Hints that are absent from either side are ignored.

    Returns:
        OpsLibraryIdDiff with missing, added, and renamed id sets.
    """

    legacy_set = set(_normalize_ids(legacy_seed_ids))
    modular_set = set(_normalize_ids(discovered_modular_ids))

    hinted_pairs: list[tuple[str, str]] = []
    if renamed_id_hints:
        for legacy_id, modular_id in renamed_id_hints.items():
            legacy_id_str = str(legacy_id)
            modular_id_str = str(modular_id)
            if legacy_id_str in legacy_set and modular_id_str in modular_set:
                hinted_pairs.append((legacy_id_str, modular_id_str))

    hinted_pairs_tuple = tuple(sorted(set(hinted_pairs)))
    hinted_legacy = {legacy_id for legacy_id, _ in hinted_pairs_tuple}
    hinted_modular = {modular_id for _, modular_id in hinted_pairs_tuple}

    unresolved_missing = sorted((legacy_set - modular_set) - hinted_legacy)
    unresolved_added = sorted((modular_set - legacy_set) - hinted_modular)

    inferred_pairs = _infer_renames(unresolved_missing, unresolved_added)
    inferred_legacy = {legacy_id for legacy_id, _ in inferred_pairs}
    inferred_modular = {modular_id for _, modular_id in inferred_pairs}

    missing_ids = tuple(sorted(set(unresolved_missing) - inferred_legacy))
    added_ids = tuple(sorted(set(unresolved_added) - inferred_modular))
    renamed_ids = tuple(sorted(set(hinted_pairs_tuple) | set(inferred_pairs)))

    return OpsLibraryIdDiff(
        missing_ids=missing_ids,
        added_ids=added_ids,
        renamed_ids=renamed_ids,
    )


__all__ = [
    "OpsLibraryIdDiff",
    "compare_legacy_seed_ids",
]
