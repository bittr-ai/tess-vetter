from __future__ import annotations

import tess_vetter.api as public_api
import tess_vetter.code_mode.adapters.manual as manual_adapters
from tess_vetter.code_mode.operation_spec import OperationSpec
from tess_vetter.code_mode.ops_library import OperationAdapter, OpsLibrary, make_default_ops_library
from tess_vetter.code_mode.registries.operation_ids import (
    build_operation_id,
    normalize_operation_name,
)
from tess_vetter.code_mode.registries.tiering import ApiSymbol, tier_for_api_symbol
from tess_vetter.code_mode.retry import RetryPolicy, retry_transient

_DOCUMENTED_OPTIONAL_EXPORT_SKIPS: frozenset[str] = frozenset(
    set(public_api._MLX_GUARDED_EXPORTS) | set(public_api._MATPLOTLIB_GUARDED_EXPORTS)
)


def _expected_operation_ids_from_export_map() -> tuple[set[str], set[str], set[str]]:
    available: set[str] = set()
    unavailable: set[str] = set()
    unloadable_unexpected: set[str] = set()

    for export_name, (module_name, _attr_name) in sorted(public_api._get_export_map().items()):
        symbol = ApiSymbol(module=module_name, name=export_name)
        operation_id = build_operation_id(
            tier=tier_for_api_symbol(symbol),
            name=normalize_operation_name(symbol.name),
        )

        try:
            value = getattr(public_api, export_name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            if export_name in _DOCUMENTED_OPTIONAL_EXPORT_SKIPS:
                unavailable.add(operation_id)
            else:
                unloadable_unexpected.add(operation_id)
            continue

        if isinstance(value, type):
            continue
        if not callable(value):
            continue

        available.add(operation_id)

    return available, unavailable, unloadable_unexpected


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


def test_default_library_applies_retry_wrapper_to_network_seed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = {"count": 0}

    def _flaky_vet_candidate(*_args: object, **_kwargs: object) -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("temporary network timeout")
        return "ok"

    def _deterministic_retry_wrapper(fn):  # type: ignore[no-untyped-def]
        def _wrapped(*args: object, **kwargs: object) -> object:
            return retry_transient(
                lambda: fn(*args, **kwargs),
                policy=RetryPolicy(attempts=3, backoff_seconds=0.0, jitter=0.0, cap_seconds=0.0),
                sleep=lambda _seconds: None,
                use_jitter=False,
            )

        return _wrapped

    monkeypatch.setattr(manual_adapters._api, "vet_candidate", _flaky_vet_candidate)
    monkeypatch.setattr(manual_adapters, "wrap_with_transient_retry", _deterministic_retry_wrapper)

    library = make_default_ops_library()
    op = library.get("code_mode.golden.vet_candidate")
    assert op("lc", "candidate", network=True) == "ok"
    assert calls["count"] == 3


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


def test_default_library_fully_covers_export_map_operation_ids_including_unavailable() -> None:
    library = make_default_ops_library()
    library_ids = set(library.list_ids())
    available_ids, unavailable_ids, unloadable_unexpected = _expected_operation_ids_from_export_map()

    assert not unloadable_unexpected, f"Unexpected unloadable exports: {sorted(unloadable_unexpected)}"
    assert unavailable_ids

    expected_ids = available_ids | unavailable_ids
    missing = expected_ids - library_ids
    assert not missing, f"Missing discovered operation ids: {sorted(missing)}"

    coverage = len(expected_ids - missing) / max(len(expected_ids), 1)
    assert coverage == 1.0

    for operation_id in sorted(unavailable_ids):
        op = library.get(operation_id)
        try:
            op()
        except ImportError:
            pass
        else:
            raise AssertionError(f"Unavailable operation did not raise ImportError: {operation_id}")
