"""Contract tests for periodogram API boundary typing artifacts."""

from __future__ import annotations

from tess_vetter.api.contracts import (
    callable_input_schema_from_signature,
    model_input_schema,
    model_output_schema,
)
from tess_vetter.api.periodogram import (
    REFINE_PERIOD_CALL_SCHEMA,
    REFINE_PERIOD_INPUT_SCHEMA,
    REFINE_PERIOD_OUTPUT_SCHEMA,
    RUN_PERIODOGRAM_CALL_SCHEMA,
    RUN_PERIODOGRAM_INPUT_SCHEMA,
    RUN_PERIODOGRAM_OUTPUT_SCHEMA,
    RefinePeriodRequest,
    RefinePeriodResponse,
    RunPeriodogramRequest,
    RunPeriodogramResponse,
    refine_period,
    run_periodogram,
)


def test_run_periodogram_schema_constants_track_models() -> None:
    assert model_input_schema(RunPeriodogramRequest) == RUN_PERIODOGRAM_INPUT_SCHEMA
    assert model_output_schema(RunPeriodogramResponse) == RUN_PERIODOGRAM_OUTPUT_SCHEMA

    assert RUN_PERIODOGRAM_INPUT_SCHEMA["required"] == ["time", "flux"]
    assert sorted(RUN_PERIODOGRAM_INPUT_SCHEMA["properties"]) == [
        "data_ref",
        "downsample_factor",
        "flux",
        "flux_err",
        "max_period",
        "max_planets",
        "method",
        "min_period",
        "per_sector",
        "preset",
        "stellar_mass_msun",
        "stellar_radius_rsun",
        "tic_id",
        "time",
        "use_threads",
    ]
    assert sorted(RUN_PERIODOGRAM_OUTPUT_SCHEMA["properties"]) == ["result"]


def test_refine_period_schema_constants_track_models() -> None:
    assert model_input_schema(RefinePeriodRequest) == REFINE_PERIOD_INPUT_SCHEMA
    assert model_output_schema(RefinePeriodResponse) == REFINE_PERIOD_OUTPUT_SCHEMA

    assert REFINE_PERIOD_INPUT_SCHEMA["required"] == [
        "time",
        "flux",
        "flux_err",
        "initial_period",
        "initial_duration",
    ]
    assert sorted(REFINE_PERIOD_OUTPUT_SCHEMA["properties"]) == ["period", "power", "t0"]


def test_run_periodogram_call_schema_is_stable() -> None:
    expected = {
        "type": "object",
        "properties": {
            "data_ref": {},
            "downsample_factor": {},
            "flux": {},
            "flux_err": {},
            "max_period": {},
            "max_planets": {},
            "method": {},
            "min_period": {},
            "per_sector": {},
            "preset": {},
            "stellar_mass_msun": {},
            "stellar_radius_rsun": {},
            "tic_id": {},
            "time": {},
            "use_threads": {},
        },
        "additionalProperties": False,
        "required": ["flux", "time"],
    }
    assert expected == RUN_PERIODOGRAM_CALL_SCHEMA
    assert callable_input_schema_from_signature(run_periodogram) == RUN_PERIODOGRAM_CALL_SCHEMA


def test_refine_period_call_schema_is_stable() -> None:
    expected = {
        "type": "object",
        "properties": {
            "flux": {},
            "flux_err": {},
            "initial_duration": {},
            "initial_period": {},
            "n_refine": {},
            "refine_factor": {},
            "stellar_mass_msun": {},
            "stellar_radius_rsun": {},
            "tic_id": {},
            "time": {},
        },
        "additionalProperties": False,
        "required": ["flux", "flux_err", "initial_duration", "initial_period", "time"],
    }
    assert expected == REFINE_PERIOD_CALL_SCHEMA
    assert callable_input_schema_from_signature(refine_period) == REFINE_PERIOD_CALL_SCHEMA
