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

    def missing_by_tier_prefix(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Group missing ids by tier prefix for machine-readable reporting."""
        return _group_ids_by_tier_prefix(self.missing_ids)

    def added_by_tier_prefix(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Group added ids by tier prefix for machine-readable reporting."""
        return _group_ids_by_tier_prefix(self.added_ids)


def _normalize_ids(ids: Iterable[str]) -> tuple[str, ...]:
    """Normalize ids into sorted unique tuples for deterministic processing."""
    return tuple(sorted({str(value) for value in ids}))


def _leaf_token(operation_id: str) -> str:
    """Return the suffix token used for conservative rename inference."""
    return operation_id.rsplit(".", 1)[-1]


def _tier_prefix(operation_id: str) -> str:
    """Return the first two path segments used as a tier prefix."""
    segments = operation_id.split(".")
    if len(segments) >= 2:
        return ".".join(segments[:2])
    return segments[0]


def _group_ids_by_tier_prefix(ids: Iterable[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Group ids by tier prefix with deterministic ordering."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for operation_id in _normalize_ids(ids):
        grouped[_tier_prefix(operation_id)].append(operation_id)
    return tuple(
        (tier_prefix, tuple(sorted(values)))
        for tier_prefix, values in sorted(grouped.items())
    )


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


def summarize_legacy_seed_coverage_delta(
    legacy_seed_ids: Iterable[str],
    discovered_modular_ids: Iterable[str],
    *,
    renamed_id_hints: Mapping[str, str] | None = None,
    unavailable_operation_ids: Iterable[str] = (),
) -> dict[str, object]:
    """Build a deterministic coverage summary for migration reporting."""

    normalized_legacy = _normalize_ids(legacy_seed_ids)
    normalized_discovered = _normalize_ids(discovered_modular_ids)
    normalized_unavailable = _normalize_ids(unavailable_operation_ids)
    discovered_set = set(normalized_discovered)
    unavailable_in_discovered = tuple(
        operation_id for operation_id in normalized_unavailable if operation_id in discovered_set
    )
    diff = compare_legacy_seed_ids(
        legacy_seed_ids=normalized_legacy,
        discovered_modular_ids=normalized_discovered,
        renamed_id_hints=renamed_id_hints,
    )
    return {
        "coverage_delta_counts": {
            "legacy_seed_total": len(normalized_legacy),
            "expanded_discovery_total": len(normalized_discovered),
            "missing_total": len(diff.missing_ids),
            "added_total": len(diff.added_ids),
            "renamed_total": len(diff.renamed_ids),
            "unavailable_total": len(unavailable_in_discovered),
            "unavailable_added_total": len(set(diff.added_ids) & set(unavailable_in_discovered)),
            "net_new_total": len(normalized_discovered) - len(normalized_legacy),
        },
        "missing_by_tier_prefix": diff.missing_by_tier_prefix(),
        "added_by_tier_prefix": diff.added_by_tier_prefix(),
        "unavailable_ids": unavailable_in_discovered,
    }


__all__ = [
    "OpsLibraryIdDiff",
    "compare_legacy_seed_ids",
    "summarize_legacy_seed_coverage_delta",
]
