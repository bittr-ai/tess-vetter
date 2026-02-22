from __future__ import annotations

import json

from tess_vetter.api.primitives_catalog import list_primitives
from tess_vetter.code_mode.generators.surface_inventory import (
    build_surface_inventory,
    surface_inventory_jsonable,
)


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

    primitive_catalog_size = len(list_primitives(include_unimplemented=True))
    primitive_catalog_rows = [
        item
        for item in payload
        if item["module"] == "tess_vetter.compute" and item["tier"] == "primitive"
    ]
    assert len(primitive_catalog_rows) == primitive_catalog_size
    assert any(item["status"] == "planned" for item in primitive_catalog_rows)

    assert any(item["symbol"] == "vet_candidate" for item in payload)
    assert any(item["symbol"] == "vet" and item["replaced_by"] for item in payload)
