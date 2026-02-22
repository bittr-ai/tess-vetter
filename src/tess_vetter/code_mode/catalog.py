"""Deterministic code-mode catalog construction and version hashing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

CatalogTier = Literal["golden_path", "primitive", "internal"]

TIER_ORDER: tuple[CatalogTier, ...] = ("golden_path", "primitive", "internal")
_TIER_RANK: dict[str, int] = {tier: idx for idx, tier in enumerate(TIER_ORDER)}
_TIER_ALIASES: dict[str, CatalogTier] = {
    "golden": "golden_path",
    "golden_path": "golden_path",
    "golden-path": "golden_path",
    "golden path": "golden_path",
    "api": "primitive",
    "primitive": "primitive",
    "primitives": "primitive",
    "internal": "internal",
}
_SCHEMA_VOLATILE_KEYS: frozenset[str] = frozenset(
    {"$defs", "$id", "$schema", "default", "description", "examples", "title"}
)
DEFAULT_REQUIRED_PATHS_CAP = 64


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Single deterministic catalog entry."""

    id: str
    tier: str
    title: str
    description: str
    tags: tuple[str, ...]
    schema: Any
    schema_fingerprint: str
    availability: str = "available"
    status: str = "active"
    deprecated: bool = False
    replacement: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogBuildResult:
    """Deterministic catalog build result."""

    entries: tuple[CatalogEntry, ...]
    canonical_lines: tuple[str, ...]
    catalog_version_hash: str


def canonicalize_value(value: Any) -> Any:
    """Canonicalize nested JSON-like values and reject non-finite numbers."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numeric value is not supported in schema")
        return value
    if isinstance(value, dict):
        return {
            str(key): canonicalize_value(value[key])
            for key in sorted(value, key=lambda k: str(k))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_value(item) for item in value]
    raise TypeError(f"Unsupported schema value type: {type(value)!r}")


def schema_fingerprint(schema: Any) -> str:
    """Return SHA-256 fingerprint of canonicalized schema."""
    canonical_schema = canonicalize_value(schema)
    payload = json.dumps(
        canonical_schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if raw_tags is None:
        return ()
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    elif isinstance(raw_tags, (list, tuple, set)):
        tags = [str(item) for item in raw_tags]
    else:
        raise TypeError("tags must be a string or sequence of strings")

    return tuple(sorted({tag.strip().lower() for tag in tags if tag and tag.strip()}))


def _tier_sort_key(tier: str) -> tuple[int, str]:
    return (_TIER_RANK.get(tier, len(TIER_ORDER)), tier)


def normalize_tier_label(raw_tier: Any) -> str:
    """Normalize tier labels for compatibility across legacy and expanded names."""
    tier = str(raw_tier).strip().lower()
    return _TIER_ALIASES.get(tier, tier)


def _canonical_line(entry: CatalogEntry) -> str:
    tags = ",".join(entry.tags)
    replacement = entry.replacement or ""
    parts = [
        entry.tier,
        entry.id,
        entry.title,
        entry.description,
        tags,
        entry.availability,
        entry.status,
        "1" if entry.deprecated else "0",
        replacement,
        entry.schema_fingerprint,
    ]
    return "|".join(parts)


def _resolve_local_json_ref(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        return {"$ref": ref}
    node: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return {"$ref": ref}
        node = node[part]
    return node


def _static_schema_snippet(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        resolved = value
        ref_value = value.get("$ref")
        if isinstance(ref_value, str):
            target = _resolve_local_json_ref(root, ref_value)
            if isinstance(target, dict):
                resolved = {**target, **{k: v for k, v in value.items() if k != "$ref"}}

        cleaned: dict[str, Any] = {}
        for key in sorted(resolved, key=lambda k: str(k)):
            if key in _SCHEMA_VOLATILE_KEYS:
                continue
            if key == "properties" and isinstance(resolved[key], dict):
                cleaned[key] = {
                    str(prop_key): _static_schema_snippet(prop_value, root)
                    for prop_key, prop_value in sorted(
                        resolved[key].items(),
                        key=lambda item: str(item[0]),
                    )
                }
                continue
            cleaned[key] = _static_schema_snippet(resolved[key], root)
        return cleaned
    if isinstance(value, list):
        return [_static_schema_snippet(item, root) for item in value]
    return value


def _pydantic_model_schema_snippet(model: Any, *, mode: str) -> dict[str, Any] | None:
    model_json_schema = getattr(model, "model_json_schema", None)
    if not callable(model_json_schema):
        return None
    raw_schema = model_json_schema(mode=mode)
    if not isinstance(raw_schema, dict):
        return None
    snippet = _static_schema_snippet(raw_schema, raw_schema)
    if not isinstance(snippet, dict):
        return None
    return snippet


def _wrapper_schema_snippets(raw: dict[str, Any]) -> dict[str, Any] | None:
    input_schema = raw.get("input_schema_snippet")
    if input_schema is None:
        input_schema = _pydantic_model_schema_snippet(raw.get("input_model"), mode="validation")
    output_schema = raw.get("output_schema_snippet")
    if output_schema is None:
        output_schema = _pydantic_model_schema_snippet(raw.get("output_model"), mode="serialization")

    snippets: dict[str, Any] = {}
    if input_schema is not None:
        snippets["input"] = canonicalize_value(input_schema)
    if output_schema is not None:
        snippets["output"] = canonicalize_value(output_schema)
    if not snippets:
        return None
    return snippets


def _join_object_path(prefix: str, key: str) -> str:
    return key if not prefix else f"{prefix}.{key}"


def _join_array_path(prefix: str) -> str:
    return "[]" if not prefix else f"{prefix}[]"


def _schema_type_set(schema: dict[str, Any]) -> frozenset[str]:
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return frozenset((raw_type,))
    if isinstance(raw_type, list):
        return frozenset(str(item) for item in raw_type if isinstance(item, str))
    return frozenset()


def _iter_required_property_names(schema: dict[str, Any]) -> tuple[str, ...]:
    raw_required = schema.get("required")
    if not isinstance(raw_required, list):
        return ()
    return tuple(sorted({name for name in raw_required if isinstance(name, str) and name}))


def _collect_required_paths(schema: Any, *, prefix: str = "") -> set[str]:
    if not isinstance(schema, dict):
        return set()

    paths: set[str] = set()

    raw_all_of = schema.get("allOf")
    if isinstance(raw_all_of, list):
        for branch in raw_all_of:
            paths.update(_collect_required_paths(branch, prefix=prefix))

    schema_types = _schema_type_set(schema)
    is_object = "object" in schema_types or "properties" in schema or "required" in schema
    if is_object:
        raw_properties = schema.get("properties")
        properties = raw_properties if isinstance(raw_properties, dict) else {}
        for property_name in _iter_required_property_names(schema):
            path = _join_object_path(prefix, property_name)
            paths.add(path)
            nested = properties.get(property_name)
            if isinstance(nested, dict):
                paths.update(_collect_required_paths(nested, prefix=path))

    is_array = "array" in schema_types or "items" in schema
    if is_array:
        array_prefix = _join_array_path(prefix)
        raw_items = schema.get("items")
        if isinstance(raw_items, dict):
            paths.update(_collect_required_paths(raw_items, prefix=array_prefix))
        elif isinstance(raw_items, list):
            for item_schema in raw_items:
                paths.update(_collect_required_paths(item_schema, prefix=array_prefix))

    for branch_key in ("anyOf", "oneOf"):
        raw_branches = schema.get(branch_key)
        if not isinstance(raw_branches, list):
            continue
        branch_sets = [
            _collect_required_paths(branch, prefix=prefix)
            for branch in raw_branches
            if isinstance(branch, dict)
        ]
        if not branch_sets:
            continue
        common = set.intersection(*branch_sets) if len(branch_sets) > 1 else set(branch_sets[0])
        paths.update(common)

    return paths


def _cap_sorted_paths(paths: set[str], *, max_paths: int) -> tuple[str, ...]:
    if max_paths < 0:
        raise ValueError("max_paths must be >= 0")
    if max_paths == 0:
        return ()
    return tuple(sorted(paths))[:max_paths]


def extract_required_paths(schema: Any, *, max_paths: int = DEFAULT_REQUIRED_PATHS_CAP) -> tuple[str, ...]:
    """Extract deterministic required field paths from a JSON schema."""
    return _cap_sorted_paths(_collect_required_paths(schema), max_paths=max_paths)


def extract_required_input_paths(
    schema: Any,
    *,
    wrapper_schemas: Any | None = None,
    max_paths: int = DEFAULT_REQUIRED_PATHS_CAP,
) -> tuple[str, ...]:
    """Extract required input paths from operation schema and optional wrapper schema."""
    operation_schema = schema
    resolved_wrapper_schemas = wrapper_schemas

    if isinstance(schema, dict):
        raw_input_schema = schema.get("input")
        if isinstance(raw_input_schema, dict):
            operation_schema = raw_input_schema
        if resolved_wrapper_schemas is None:
            embedded_wrapper_schemas = schema.get("wrapper_schemas")
            if isinstance(embedded_wrapper_schemas, dict):
                resolved_wrapper_schemas = embedded_wrapper_schemas

    paths = _collect_required_paths(operation_schema)
    if isinstance(resolved_wrapper_schemas, dict):
        wrapper_input_schema = resolved_wrapper_schemas.get("input")
        if isinstance(wrapper_input_schema, dict):
            paths.update(_collect_required_paths(wrapper_input_schema))

    return _cap_sorted_paths(paths, max_paths=max_paths)


def short_hash(value: str, *, length: int = 12) -> str:
    """Return a compact prefix for a hash string."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return value[:length]


def build_catalog(entries: list[dict[str, Any]]) -> CatalogBuildResult:
    """Build deterministic catalog from input entries."""
    normalized: list[CatalogEntry] = []

    for raw in entries:
        entry_id = str(raw["id"])
        tier = normalize_tier_label(raw["tier"])
        title = str(raw.get("title") or raw.get("name") or entry_id)
        description = str(raw.get("description") or "")
        tags = _normalize_tags(raw.get("tags"))
        availability = str(raw.get("availability") or "available").strip().lower() or "available"
        status = str(raw.get("status") or "active").strip().lower() or "active"
        deprecated = bool(raw.get("deprecated", False))
        replacement_raw = raw.get("replacement")
        replacement = None
        if replacement_raw is not None:
            replacement_value = str(replacement_raw).strip()
            if replacement_value:
                replacement = replacement_value

        raw_schema = raw.get("schema", {})
        canonical_schema = canonicalize_value(raw_schema)
        wrapper_schemas = _wrapper_schema_snippets(raw)
        if wrapper_schemas is not None:
            if isinstance(canonical_schema, dict):
                canonical_schema = {
                    **canonical_schema,
                    "wrapper_schemas": wrapper_schemas,
                }
            else:
                canonical_schema = {
                    "schema": canonical_schema,
                    "wrapper_schemas": wrapper_schemas,
                }
            canonical_schema = canonicalize_value(canonical_schema)
        schema_fp = schema_fingerprint(canonical_schema)

        normalized.append(
            CatalogEntry(
                id=entry_id,
                tier=tier,
                title=title,
                description=description,
                tags=tags,
                availability=availability,
                status=status,
                deprecated=deprecated,
                replacement=replacement,
                schema=canonical_schema,
                schema_fingerprint=schema_fp,
            )
        )

    sorted_entries = tuple(sorted(normalized, key=lambda e: (_tier_sort_key(e.tier), e.id)))
    canonical_lines = tuple(_canonical_line(entry) for entry in sorted_entries)
    lines_payload = "\n".join(canonical_lines)
    version_hash = sha256(lines_payload.encode("utf-8")).hexdigest()

    return CatalogBuildResult(
        entries=sorted_entries,
        canonical_lines=canonical_lines,
        catalog_version_hash=version_hash,
    )


__all__ = [
    "CatalogBuildResult",
    "CatalogEntry",
    "CatalogTier",
    "DEFAULT_REQUIRED_PATHS_CAP",
    "TIER_ORDER",
    "build_catalog",
    "canonicalize_value",
    "extract_required_input_paths",
    "extract_required_paths",
    "normalize_tier_label",
    "schema_fingerprint",
    "short_hash",
]
