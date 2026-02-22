from __future__ import annotations

import inspect
import json

import tess_vetter.api as public_api
from tess_vetter.api.primitives_catalog import list_primitives
from tess_vetter.code_mode.generators.surface_inventory import (
    build_surface_inventory,
    surface_inventory_jsonable,
)
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol

_DOCUMENTED_OPTIONAL_EXPORT_SKIPS: frozenset[str] = frozenset(
    set(public_api._MLX_GUARDED_EXPORTS) | set(public_api._MATPLOTLIB_GUARDED_EXPORTS)
)


def _expected_discovery_aligned_rows() -> tuple[set[tuple[str, str, str, str]], set[str], set[str]]:
    expected: set[tuple[str, str, str, str]] = set()
    unloadable_documented: set[str] = set()
    unloadable_unexpected: set[str] = set()

    for export_name, (module_name, _attr_name) in sorted(public_api._get_export_map().items()):
        symbol = ApiSymbol(module=module_name, name=export_name)
        try:
            value = getattr(public_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            if export_name in _DOCUMENTED_OPTIONAL_EXPORT_SKIPS:
                unloadable_documented.add(export_name)
            else:
                unloadable_unexpected.add(export_name)
            continue

        if inspect.isclass(value):
            continue
        if not (inspect.isroutine(value) or callable(value)):
            continue

        tier = tier_for_api_symbol(symbol)
        expected.add(
            (
                symbol.name,
                symbol.module,
                build_operation_id(tier=tier, name=normalize_operation_name(symbol.name)),
                tier,
            )
        )

    return expected, unloadable_documented, unloadable_unexpected


def test_surface_inventory_is_deterministic_and_json_serializable() -> None:
    rows_a = build_surface_inventory()
    rows_b = build_surface_inventory()

    assert rows_a == rows_b

    payload_a = surface_inventory_jsonable(rows_a)
    payload_b = surface_inventory_jsonable(rows_b)
    assert payload_a == payload_b

    encoded = json.dumps(payload_a, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    decoded = json.loads(encoded)
    assert decoded == payload_a


def test_surface_inventory_has_basic_coverage_counts() -> None:
    rows = build_surface_inventory()
    payload = surface_inventory_jsonable(rows)

    assert payload
    assert all(item["operation_id"] for item in payload)
    assert all(item["tier"] for item in payload)
    assert all(item["symbol"] for item in payload)
    assert all(item["module"] for item in payload)
    assert all(item["status"] for item in payload)

    tier_counts: dict[str, int] = {}
    for item in payload:
        tier_counts[item["tier"]] = tier_counts.get(item["tier"], 0) + 1

    assert tier_counts.get("golden_path", 0) > 0
    assert tier_counts.get("primitive", 0) > 0
    assert tier_counts.get("internal", 0) > 0

    primitive_catalog_size = len(list_primitives(include_unimplemented=True))
    primitive_catalog_rows = [
        item
        for item in payload
        if item["module"] == "tess_vetter.compute" and item["tier"] == "primitive"
    ]
    assert len(primitive_catalog_rows) == primitive_catalog_size
    assert any(item["status"] == "planned" for item in primitive_catalog_rows)

    assert any(item["symbol"] == "vet_candidate" for item in payload)
    assert all(not item["replaced_by"] for item in payload)


def test_surface_inventory_fully_covers_loadable_export_map_routines() -> None:
    payload = surface_inventory_jsonable(build_surface_inventory())
    expected_rows, unloadable_documented, unloadable_unexpected = _expected_discovery_aligned_rows()

    assert not unloadable_unexpected, f"Unexpected unloadable exports: {sorted(unloadable_unexpected)}"
    assert unloadable_documented <= _DOCUMENTED_OPTIONAL_EXPORT_SKIPS

    actual_rows = {
        (item["symbol"], item["module"], item["operation_id"], item["tier"])
        for item in payload
        if item["status"] == "available"
    }
    missing = expected_rows - actual_rows
    assert not missing, f"Missing inventory rows: {sorted(missing)}"

    coverage = len(expected_rows - missing) / max(len(expected_rows), 1)
    assert coverage == 1.0
