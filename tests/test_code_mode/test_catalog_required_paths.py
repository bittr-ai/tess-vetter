from __future__ import annotations

import pytest

from tess_vetter.api.contracts import opaque_object_schema
from tess_vetter.code_mode.catalog import extract_required_input_paths, extract_required_paths
from tess_vetter.code_mode.operation_spec import OperationSpec
from tess_vetter.code_mode.ops_library import OperationAdapter, required_input_paths_for_adapter


def test_extract_required_paths_nested_objects_and_arrays() -> None:
    schema = {
        "type": "object",
        "required": ["lc", "candidate"],
        "properties": {
            "candidate": {
                "type": "object",
                "required": ["ephemeris"],
                "properties": {
                    "ephemeris": {
                        "type": "object",
                        "required": ["period_days", "t0_btjd", "duration_hours"],
                        "properties": {
                            "period_days": {"type": "number"},
                            "t0_btjd": {"type": "number"},
                            "duration_hours": {"type": "number"},
                        },
                    }
                },
            },
            "lc": {
                "type": "object",
                "required": ["time", "flux", "apertures"],
                "properties": {
                    "time": {"type": "array", "items": {"type": "number"}},
                    "flux": {"type": "array", "items": {"type": "number"}},
                    "apertures": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "weights"],
                            "properties": {
                                "id": {"type": "integer"},
                                "weights": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["pixel"],
                                        "properties": {
                                            "pixel": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    assert extract_required_paths(schema) == (
        "candidate",
        "candidate.ephemeris",
        "candidate.ephemeris.duration_hours",
        "candidate.ephemeris.period_days",
        "candidate.ephemeris.t0_btjd",
        "lc",
        "lc.apertures",
        "lc.apertures[].id",
        "lc.apertures[].weights",
        "lc.apertures[].weights[].pixel",
        "lc.flux",
        "lc.time",
    )


def test_extract_required_paths_is_deterministic_and_capped() -> None:
    schema_a = {
        "type": "object",
        "required": ["zeta", "alpha", "middle"],
        "properties": {
            "zeta": {"type": "integer"},
            "alpha": {"type": "integer"},
            "middle": {"type": "integer"},
        },
    }
    schema_b = {
        "properties": {
            "middle": {"type": "integer"},
            "alpha": {"type": "integer"},
            "zeta": {"type": "integer"},
        },
        "required": ["middle", "zeta", "alpha"],
        "type": "object",
    }

    assert extract_required_paths(schema_a) == extract_required_paths(schema_b)
    assert extract_required_paths(schema_a, max_paths=2) == ("alpha", "middle")
    assert extract_required_paths(schema_a, max_paths=0) == ()

    with pytest.raises(ValueError, match="max_paths"):
        extract_required_paths(schema_a, max_paths=-1)


def test_extract_required_input_paths_merges_operation_and_wrapper_truth() -> None:
    catalog_schema = {
        "input": {
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                    },
                }
            },
        },
        "wrapper_schemas": {
            "input": {
                "type": "object",
                "required": ["payload"],
                "properties": {
                    "payload": {
                        "type": "object",
                        "required": ["id", "coords"],
                        "properties": {
                            "id": {"type": "string"},
                            "coords": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["ra", "dec"],
                                    "properties": {
                                        "ra": {"type": "number"},
                                        "dec": {"type": "number"},
                                    },
                                },
                            },
                        },
                    }
                },
            }
        },
    }

    assert extract_required_input_paths(catalog_schema) == (
        "payload",
        "payload.coords",
        "payload.coords[].dec",
        "payload.coords[].ra",
        "payload.id",
    )


def test_required_input_paths_for_adapter_uses_operation_schema() -> None:
    adapter = OperationAdapter(
        spec=OperationSpec(
            id="code_mode.primitive.required_paths",
            name="Required Paths",
            input_json_schema={
                "type": "object",
                "required": ["context"],
                "properties": {
                    "context": {
                        "type": "object",
                        "required": ["candidate_id"],
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    }
                },
            },
        ),
        fn=lambda **_kwargs: None,
    )

    assert required_input_paths_for_adapter(adapter) == ("context", "context.candidate_id")


def test_required_paths_from_upstream_opaque_schema_contract() -> None:
    assert extract_required_paths(opaque_object_schema()) == ()
