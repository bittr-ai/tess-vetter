from __future__ import annotations

from tess_vetter.code_mode.adapters.check_wrappers import (
    check_wrapper_definitions,
    check_wrapper_functions,
)
from tess_vetter.code_mode.adapters.manual import manual_seed_adapters
from tess_vetter.validation.result_schema import ok_result


def _base_payload() -> dict[str, object]:
    return {
        "lc": {
            "time": [1.0, 2.0, 3.0],
            "flux": [1.0, 0.99, 1.01],
            "flux_err": [0.01, 0.01, 0.01],
        },
        "candidate": {
            "ephemeris": {
                "period_days": 2.5,
                "t0_btjd": 1000.0,
                "duration_hours": 2.0,
            },
            "depth_ppm": 500.0,
        },
    }


def test_check_wrapper_inventory_is_deterministic_v01_v15_set() -> None:
    definitions = check_wrapper_definitions()
    ids = [definition.check_id for definition in definitions]
    operation_ids = [definition.operation_id for definition in definitions]

    assert ids == [
        "V01",
        "V02",
        "V03",
        "V04",
        "V05",
        "V06",
        "V07",
        "V08",
        "V09",
        "V10",
        "V11",
        "V12",
        "V13",
        "V15",
    ]
    assert operation_ids == sorted(operation_ids)
    assert len(operation_ids) == len(set(operation_ids))


def test_check_wrapper_callable_returns_typed_model(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    definition, wrapper = check_wrapper_functions()[0]

    def _fake_run_check(**_kwargs):
        return ok_result(
            id=definition.check_id,
            name="Odd-Even Depth",
            metrics={
                "odd_depth": 0.001,
                "even_depth": 0.0011,
                "odd_even_diff_sigma": 0.3,
                "legacy_metric": 7.0,
            },
            confidence=0.8,
            flags=["INFO"],
            notes=["ok"],
            provenance={"source": "test"},
        )

    monkeypatch.setattr("tess_vetter.code_mode.adapters.check_wrappers._api.run_check", _fake_run_check)

    result = wrapper(**_base_payload())

    assert result.check_id == "V01"
    assert result.status == "ok"
    assert result.metrics.odd_depth == 0.001
    assert result.metrics.even_depth == 0.0011
    assert result.metrics.odd_even_diff_sigma == 0.3
    assert result.metrics.extras == {"legacy_metric": 7.0}


def test_manual_seed_adapters_include_typed_check_wrappers_with_schemas() -> None:
    adapters = manual_seed_adapters()
    wrapper_ids = [
        adapter.id
        for adapter in adapters
        if adapter.id.startswith("code_mode.internal.check_v")
    ]

    assert len(wrapper_ids) == 14
    assert wrapper_ids == sorted(wrapper_ids)

    wrapper = next(adapter for adapter in adapters if adapter.id == "code_mode.internal.check_v01_odd_even_depth")
    assert wrapper.spec.input_json_schema.get("type") == "object"
    assert "properties" in wrapper.spec.input_json_schema
    assert wrapper.spec.output_json_schema.get("type") == "object"
    assert "properties" in wrapper.spec.output_json_schema
