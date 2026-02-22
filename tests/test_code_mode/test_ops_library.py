from __future__ import annotations

import tess_vetter.code_mode.ops_library as ops_library
from tess_vetter.code_mode.operation_spec import OperationSpec
from tess_vetter.code_mode.ops_library import OperationAdapter, OpsLibrary, make_default_ops_library


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

    monkeypatch.setattr(ops_library._api, "vet_candidate", _fake_vet_candidate)

    library = make_default_ops_library()
    op = library.get("code_mode.golden.vet_candidate")

    result = op("lc", "candidate", network=False)
    assert result == "ok"
    assert seen["args"] == ("lc", "candidate")
    assert seen["kwargs"] == {"network": False}


def test_default_library_ids_are_stable_and_sorted() -> None:
    library = make_default_ops_library()

    assert library.list_ids() == sorted(
        [
            "code_mode.golden.run_periodogram",
            "code_mode.golden.vet_candidate",
            "code_mode.primitive.fold",
            "code_mode.primitive.median_detrend",
        ]
    )
