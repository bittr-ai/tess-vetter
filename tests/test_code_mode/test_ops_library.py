from __future__ import annotations

import inspect

import tess_vetter.api as public_api
import tess_vetter.code_mode.adapters.manual as manual_adapters
from tess_vetter.code_mode.operation_spec import OperationSpec
from tess_vetter.code_mode.ops_library import OperationAdapter, OpsLibrary, make_default_ops_library
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol

_DOCUMENTED_OPTIONAL_EXPORT_SKIPS: frozenset[str] = frozenset(
    set(public_api._MLX_GUARDED_EXPORTS) | set(public_api._MATPLOTLIB_GUARDED_EXPORTS)
)


def _expected_loadable_operation_ids_from_export_map() -> tuple[set[str], set[str], set[str]]:
    expected: set[str] = set()
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

        expected.add(
            build_operation_id(
                tier=tier_for_api_symbol(symbol),
                name=normalize_operation_name(symbol.name),
            )
        )

    return expected, unloadable_documented, unloadable_unexpected


def test_operation_adapter_forwards_call_unchanged() -> None:
    seen: dict[str, object] = {}

    def _fn(*args: object, **kwargs: object) -> dict[str, object]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"ok": True}

    adapter = OperationAdapter(
        spec=OperationSpec(id="code_mode.primitive.local_fn", name="Local Fn"),
        fn=_fn,
    )

    result = adapter(1, 2, key="value")
    assert result == {"ok": True}
    assert seen["args"] == (1, 2)
    assert seen["kwargs"] == {"key": "value"}


def test_ops_library_list_outputs_are_deterministic() -> None:
    library = OpsLibrary()
    library.register(
        OperationAdapter(
            spec=OperationSpec(id="code_mode.zeta.second", name="Second"),
            fn=lambda: None,
        )
    )
    library.register(
        OperationAdapter(
            spec=OperationSpec(id="code_mode.alpha.first", name="First"),
            fn=lambda: None,
        )
    )

    assert library.list_ids() == ["code_mode.alpha.first", "code_mode.zeta.second"]
    assert [op.id for op in library.list()] == ["code_mode.alpha.first", "code_mode.zeta.second"]


def test_default_library_uses_current_api_callables(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def _fake_vet_candidate(*args: object, **kwargs: object) -> str:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(manual_adapters._api, "vet_candidate", _fake_vet_candidate)

    library = make_default_ops_library()
    op = library.get("code_mode.golden.vet_candidate")

    result = op("lc", "candidate", network=False)
    assert result == "ok"
    assert seen["args"] == ("lc", "candidate")
    assert seen["kwargs"] == {"network": False}


def test_default_library_ids_are_stable_sorted_and_unique() -> None:
    library_a = make_default_ops_library()
    library_b = make_default_ops_library()

    ids_a = library_a.list_ids()
    ids_b = library_b.list_ids()

    assert ids_a == ids_b
    assert ids_a == sorted(ids_a)
    assert len(ids_a) == len(set(ids_a))


def test_default_library_includes_seed_and_broad_discovered_exports() -> None:
    library = make_default_ops_library()
    ids = library.list_ids()

    # Original manual seed operations remain present for backward compatibility.
    assert "code_mode.golden.vet_candidate" in ids
    assert "code_mode.golden.run_periodogram" in ids
    assert "code_mode.primitive.fold" in ids
    assert "code_mode.primitive.median_detrend" in ids

    # Auto-discovered callable exports provide much broader surface coverage.
    discovered_ids = [op_id for op_id in ids if op_id.startswith(("code_mode.golden_path.", "code_mode.internal."))]
    assert len(discovered_ids) >= 100
    assert len(ids) >= 120

    # Key golden-path and primitive exports should be auto-registered too.
    assert "code_mode.golden_path.vet_candidate" in ids
    assert "code_mode.golden_path.run_periodogram" in ids
    assert "code_mode.internal.calculate_fpp" in ids
    assert "code_mode.primitive.box_model" in ids


def test_default_library_fully_covers_loadable_operation_like_exports() -> None:
    library_ids = set(make_default_ops_library().list_ids())
    expected_ids, unloadable_documented, unloadable_unexpected = (
        _expected_loadable_operation_ids_from_export_map()
    )

    assert not unloadable_unexpected, f"Unexpected unloadable exports: {sorted(unloadable_unexpected)}"
    assert unloadable_documented <= _DOCUMENTED_OPTIONAL_EXPORT_SKIPS

    missing = expected_ids - library_ids
    assert not missing, f"Missing discovered operation ids: {sorted(missing)}"

    coverage = len(expected_ids - missing) / max(len(expected_ids), 1)
    assert coverage == 1.0
