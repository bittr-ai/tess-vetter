from __future__ import annotations

import json

import tess_vetter.api as public_api
import tess_vetter.code_mode.generators.surface_inventory as surface_inventory_module
from tess_vetter.api.primitives_catalog import list_primitives
from tess_vetter.code_mode.generators.surface_inventory import (
    build_surface_inventory,
    surface_inventory_jsonable,
)
from tess_vetter.code_mode.policy import (
    EXPORT_POLICY_ACTIONABLE,
    EXPORT_POLICY_LEGACY_DYNAMIC,
    classify_api_export_policy,
)
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol

_DOCUMENTED_OPTIONAL_EXPORT_SKIPS: frozenset[str] = frozenset(
    set(public_api._MLX_GUARDED_EXPORTS) | set(public_api._MATPLOTLIB_GUARDED_EXPORTS)
)


def _expected_discovery_aligned_rows() -> tuple[set[tuple[str, str, str, str, str, str]], set[str], set[str]]:
    expected: set[tuple[str, str, str, str, str, str]] = set()
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
            tier = tier_for_api_symbol(symbol)
            export_policy = classify_api_export_policy(
                export_name=export_name,
                module_name=module_name,
                value=None,
            )
            expected.add(
                (
                    symbol.name,
                    symbol.module,
                    build_operation_id(tier=tier, name=normalize_operation_name(symbol.name)),
                    tier,
                    "unavailable" if export_policy == EXPORT_POLICY_ACTIONABLE else "legacy_dynamic",
                    export_policy,
                )
            )
            continue

        tier = tier_for_api_symbol(symbol)
        export_policy = classify_api_export_policy(
            export_name=export_name,
            module_name=module_name,
            value=value,
        )
        expected.add(
            (
                symbol.name,
                symbol.module,
                build_operation_id(tier=tier, name=normalize_operation_name(symbol.name)),
                tier,
                "available" if export_policy == EXPORT_POLICY_ACTIONABLE else "legacy_dynamic",
                export_policy,
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
    by_operation_id = {item["operation_id"]: item for item in payload}

    assert payload
    assert all(item["operation_id"] for item in payload)
    assert all(item["tier"] for item in payload)
    assert all(item["symbol"] for item in payload)
    assert all(item["module"] for item in payload)
    assert all(item["status"] for item in payload)
    assert {item["status"] for item in payload} <= {"available", "planned", "unavailable", "legacy_dynamic"}
    assert {item["export_policy"] for item in payload} <= {"actionable", "legacy_dynamic"}

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
    assert by_operation_id["code_mode.golden_path.vet_candidate"]["export_policy"] == EXPORT_POLICY_ACTIONABLE
    assert by_operation_id["code_mode.golden_path.vet_candidate"]["status"] == "available"
    assert by_operation_id["code_mode.golden_path.vet_candidate"]["module"] == "tess_vetter.api.vet"
    assert by_operation_id["code_mode.golden_path.run_periodogram"]["status"] == "available"
    assert by_operation_id["code_mode.golden_path.run_periodogram"]["module"] == "tess_vetter.api.periodogram"
    assert by_operation_id["code_mode.internal.vet_catalog"]["status"] == "available"
    assert by_operation_id["code_mode.internal.vet_catalog"]["module"] == "tess_vetter.api.catalog"
    assert by_operation_id["code_mode.internal.run_check"]["status"] == "available"
    assert by_operation_id["code_mode.internal.run_check"]["module"] == "tess_vetter.api.check_runner"
    assert by_operation_id["code_mode.internal.generate_control"]["status"] == "legacy_dynamic"
    assert by_operation_id["code_mode.internal.generate_control"]["export_policy"] == EXPORT_POLICY_LEGACY_DYNAMIC


def test_surface_inventory_fully_covers_loadable_export_map_routines() -> None:
    payload = surface_inventory_jsonable(build_surface_inventory())
    expected_rows, unloadable_documented, unloadable_unexpected = _expected_discovery_aligned_rows()

    assert not unloadable_unexpected, f"Unexpected unloadable exports: {sorted(unloadable_unexpected)}"
    assert unloadable_documented <= _DOCUMENTED_OPTIONAL_EXPORT_SKIPS

    expected_symbols = {row[0] for row in expected_rows}
    actual_rows = {
        (
            item["symbol"],
            item["module"],
            item["operation_id"],
            item["tier"],
            item["status"],
            item["export_policy"],
        )
        for item in payload
        if item["symbol"] in expected_symbols
    }
    missing = expected_rows - actual_rows
    assert not missing, f"Missing inventory rows: {sorted(missing)}"

    coverage = len(expected_rows - missing) / max(len(expected_rows), 1)
    assert coverage == 1.0


def test_surface_inventory_emits_unavailable_rows_for_unloadable_exports(monkeypatch) -> None:
    export_map = {
        "missing_optional": ("tess_vetter.api.optional", "missing_optional"),
        "loadable_fn": ("tess_vetter.api.loadable", "loadable_fn"),
    }

    class FakeApi:
        def _get_export_map(self) -> dict[str, tuple[str, str]]:
            return export_map

        @staticmethod
        def loadable_fn() -> str:
            return "ok"

        def __getattr__(self, name: str) -> object:
            if name == "missing_optional":
                raise ImportError("optional dependency missing")
            raise AttributeError(name)

    monkeypatch.setattr(surface_inventory_module, "_api", FakeApi())
    monkeypatch.setattr(
        surface_inventory_module,
        "list_primitives",
        lambda include_unimplemented=True: {},
    )

    payload = surface_inventory_jsonable(build_surface_inventory())
    assert [item["symbol"] for item in payload] == ["loadable_fn", "missing_optional"]
    assert [item["operation_id"] for item in payload] == [
        "code_mode.internal.loadable_fn",
        "code_mode.internal.missing_optional",
    ]
    assert [item["tier"] for item in payload] == ["internal", "internal"]
    assert [item["status"] for item in payload] == ["available", "unavailable"]
    assert [item["export_policy"] for item in payload] == ["actionable", "actionable"]
