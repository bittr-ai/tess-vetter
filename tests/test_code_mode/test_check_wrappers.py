from __future__ import annotations

import tess_vetter.api.constructor_contracts as constructor_contracts
import tess_vetter.code_mode.adapters.manual as manual_adapters_module
from tess_vetter.code_mode.adapters.check_wrappers import (
    check_wrapper_definitions,
    check_wrapper_functions,
)
from tess_vetter.code_mode.adapters.manual import legacy_manual_seed_ids, manual_seed_adapters
from tess_vetter.code_mode.mcp_adapter import SearchRequest, make_default_mcp_adapter
from tess_vetter.code_mode.ops_library import (
    make_default_ops_library,
    required_input_paths_for_adapter,
)
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


def test_only_v06_v07_check_definitions_require_network() -> None:
    network_flags = {
        definition.check_id: definition.needs_network
        for definition in check_wrapper_definitions()
    }

    assert network_flags["V06"] is True
    assert network_flags["V07"] is True
    assert {
        check_id for check_id, needs_network in network_flags.items() if needs_network
    } == {"V06", "V07"}


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


def test_manual_seed_adapters_include_typed_constructor_composers() -> None:
    adapters = manual_seed_adapters()
    composer_ids = [
        adapter.id
        for adapter in adapters
        if adapter.id.startswith("code_mode.primitive.compose_")
    ]

    assert composer_ids == [
        "code_mode.primitive.compose_candidate",
        "code_mode.primitive.compose_lightcurve",
        "code_mode.primitive.compose_stellar",
        "code_mode.primitive.compose_tpf",
    ]

    for operation_id in composer_ids:
        adapter = next(op for op in adapters if op.id == operation_id)
        assert adapter.spec.input_json_schema.get("type") == "object"
        assert adapter.spec.output_json_schema.get("type") == "object"
        assert adapter.spec.examples


def test_legacy_manual_seed_ids_track_leading_manual_seed_adapters() -> None:
    adapters = manual_seed_adapters()
    expected_ids = legacy_manual_seed_ids()
    assert tuple(adapter.id for adapter in adapters[: len(expected_ids)]) == expected_ids


def test_constructor_contract_bindings_are_owned_by_api_module() -> None:
    assert manual_adapters_module.COMPOSE_CANDIDATE_INPUT_SCHEMA is constructor_contracts.COMPOSE_CANDIDATE_INPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_CANDIDATE_OUTPUT_SCHEMA is constructor_contracts.COMPOSE_CANDIDATE_OUTPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_LIGHTCURVE_INPUT_SCHEMA is constructor_contracts.COMPOSE_LIGHTCURVE_INPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA is constructor_contracts.COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_STELLAR_INPUT_SCHEMA is constructor_contracts.COMPOSE_STELLAR_INPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_STELLAR_OUTPUT_SCHEMA is constructor_contracts.COMPOSE_STELLAR_OUTPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_TPF_INPUT_SCHEMA is constructor_contracts.COMPOSE_TPF_INPUT_SCHEMA
    assert manual_adapters_module.COMPOSE_TPF_OUTPUT_SCHEMA is constructor_contracts.COMPOSE_TPF_OUTPUT_SCHEMA
    assert manual_adapters_module.compose_candidate is constructor_contracts.compose_candidate
    assert manual_adapters_module.compose_lightcurve is constructor_contracts.compose_lightcurve
    assert manual_adapters_module.compose_stellar is constructor_contracts.compose_stellar
    assert manual_adapters_module.compose_tpf is constructor_contracts.compose_tpf


def test_constructor_composer_schemas_match_api_constructor_contracts() -> None:
    adapters = {adapter.id: adapter for adapter in manual_seed_adapters()}
    expected = {
        "code_mode.primitive.compose_candidate": (
            constructor_contracts.compose_candidate,
            constructor_contracts.COMPOSE_CANDIDATE_INPUT_SCHEMA,
            constructor_contracts.COMPOSE_CANDIDATE_OUTPUT_SCHEMA,
        ),
        "code_mode.primitive.compose_lightcurve": (
            constructor_contracts.compose_lightcurve,
            constructor_contracts.COMPOSE_LIGHTCURVE_INPUT_SCHEMA,
            constructor_contracts.COMPOSE_LIGHTCURVE_OUTPUT_SCHEMA,
        ),
        "code_mode.primitive.compose_stellar": (
            constructor_contracts.compose_stellar,
            constructor_contracts.COMPOSE_STELLAR_INPUT_SCHEMA,
            constructor_contracts.COMPOSE_STELLAR_OUTPUT_SCHEMA,
        ),
        "code_mode.primitive.compose_tpf": (
            constructor_contracts.compose_tpf,
            constructor_contracts.COMPOSE_TPF_INPUT_SCHEMA,
            constructor_contracts.COMPOSE_TPF_OUTPUT_SCHEMA,
        ),
    }

    for operation_id, (expected_fn, expected_input_schema, expected_output_schema) in expected.items():
        adapter = adapters[operation_id]
        assert adapter.fn is expected_fn
        assert adapter.spec.input_json_schema == expected_input_schema
        assert adapter.spec.output_json_schema == expected_output_schema


def test_constructor_composers_never_use_signature_schema_inference(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _forbidden_signature_helper(_fn: object) -> dict[str, object]:
        raise AssertionError("constructor composer must not use signature-based schema inference")

    monkeypatch.setattr(
        manual_adapters_module,
        "callable_input_schema_from_signature",
        _forbidden_signature_helper,
        raising=False,
    )

    adapters = manual_adapters_module.manual_seed_adapters()
    composer_ids = [
        adapter.id
        for adapter in adapters
        if adapter.id.startswith("code_mode.primitive.compose_")
    ]
    assert composer_ids == [
        "code_mode.primitive.compose_candidate",
        "code_mode.primitive.compose_lightcurve",
        "code_mode.primitive.compose_stellar",
        "code_mode.primitive.compose_tpf",
    ]


def test_constructor_composer_required_paths_and_callability_examples() -> None:
    library = make_default_ops_library()
    adapter = library.get("code_mode.primitive.compose_candidate")
    required_paths = required_input_paths_for_adapter(adapter)

    assert required_paths == ("ephemeris",)

    mcp_adapter = make_default_mcp_adapter()
    response = mcp_adapter.search(SearchRequest(query="compose candidate", limit=20, tags=[]))
    target = next(result for result in response.results if result.id == "code_mode.primitive.compose_candidate")
    callability = target.metadata["operation_callability"]

    assert callability["required_paths"] == list(required_paths)
    assert "operation_kwargs" in callability["minimal_payload_example"]["context"]


def test_legacy_manual_seed_required_paths_are_non_empty_and_informative() -> None:
    expected_required_paths = {
        "code_mode.golden.vet_candidate": {"candidate", "lc"},
        "code_mode.golden.run_periodogram": {"flux", "time"},
    }

    library = make_default_ops_library()
    for operation_id, expected_paths in expected_required_paths.items():
        adapter = library.get(operation_id)
        required_paths = set(required_input_paths_for_adapter(adapter))
        assert required_paths
        assert expected_paths <= required_paths

    response = make_default_mcp_adapter().search(SearchRequest(query="", limit=1_000, tags=[]))
    assert response.error is None
    by_id = {result.id: result for result in response.results}
    for operation_id, expected_paths in expected_required_paths.items():
        row = by_id.get(operation_id)
        assert row is not None, f"Missing search row for {operation_id}"
        callability = row.metadata.get("operation_callability")
        assert isinstance(callability, dict), f"Missing operation_callability for {operation_id}"
        required_paths = callability.get("required_paths")
        assert isinstance(required_paths, list) and required_paths, f"Missing required_paths for {operation_id}"
        assert expected_paths <= set(required_paths)
