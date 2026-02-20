from __future__ import annotations

import pytest

from tess_vetter.cli.common_cli import EXIT_INPUT_ERROR, BtvCliError
from tess_vetter.pipeline_composition.ref_resolver import extract_path, resolve_value


def test_extract_path_supports_steps_prefix_dollar_prefix_and_indices() -> None:
    payload = {"outer": {"items": [{"x": "a"}, {"x": "b"}]}}

    assert extract_path(payload, "steps.outer.items[1].x") == "b"
    assert extract_path(payload, "$.outer.items[0].x") == "a"
    assert extract_path(payload, "$outer.items[1].x") == "b"


def test_extract_path_raises_for_missing_key_and_bad_index() -> None:
    payload = {"outer": {"items": [{"x": "a"}]}}

    with pytest.raises(KeyError, match="Missing key 'missing'"):
        extract_path(payload, "outer.missing")

    with pytest.raises(KeyError, match=r"Missing list index \[2\]"):
        extract_path(payload, "outer.items[2].x")


def test_resolve_value_resolves_port_report_from_ref_and_nested_structures() -> None:
    step_outputs = {
        "seed": {
            "_step_output_path": "/tmp/seed.json",
            "result": {"nested": [{"value": 11}]},
        }
    }
    step_ports = {"neighbors": {"reference_sources_file": "/tmp/ref.json"}}

    resolved = resolve_value(
        {
            "path": {"report_from": "seed"},
            "ref": {"$ref": "steps.seed.result.nested[0].value"},
            "port": {"port": "neighbors.reference_sources_file"},
            "list": [{"$ref": "steps.seed.result.nested[0].value"}, "x"],
        },
        step_outputs=step_outputs,
        step_ports=step_ports,
    )

    assert resolved == {
        "path": "/tmp/seed.json",
        "ref": 11,
        "port": "/tmp/ref.json",
        "list": [11, "x"],
    }


def test_resolve_value_rejects_invalid_port_shape_and_unknown_port() -> None:
    with pytest.raises(BtvCliError) as exc:
        resolve_value({"port": "missingdot"}, step_outputs={}, step_ports={})
    assert exc.value.exit_code == EXIT_INPUT_ERROR
    assert "Invalid port reference" in str(exc.value)

    with pytest.raises(BtvCliError) as exc:
        resolve_value({"port": "s.p"}, step_outputs={}, step_ports={})
    assert exc.value.exit_code == EXIT_INPUT_ERROR
    assert "Unknown port reference" in str(exc.value)


def test_resolve_value_rejects_unknown_report_from_step() -> None:
    with pytest.raises(BtvCliError) as exc:
        resolve_value({"report_from": "missing"}, step_outputs={}, step_ports={})

    assert exc.value.exit_code == EXIT_INPUT_ERROR
    assert "report_from references unknown step 'missing'" in str(exc.value)


def test_resolve_value_rejects_invalid_or_unknown_refs() -> None:
    with pytest.raises(BtvCliError, match="Unsupported \\$ref"):
        resolve_value({"$ref": "bad.prefix"}, step_outputs={}, step_ports={})

    with pytest.raises(BtvCliError, match="Invalid \\$ref"):
        resolve_value({"$ref": "steps.only_step"}, step_outputs={}, step_ports={})

    with pytest.raises(BtvCliError, match="references unknown step"):
        resolve_value({"$ref": "steps.missing.value"}, step_outputs={}, step_ports={})


def test_resolve_value_wraps_json_path_key_errors_in_cli_error() -> None:
    step_outputs = {"seed": {"result": {"x": 1}}}

    with pytest.raises(BtvCliError) as exc:
        resolve_value(
            {"$ref": "steps.seed.result.missing"},
            step_outputs=step_outputs,
            step_ports={},
        )

    assert exc.value.exit_code == EXIT_INPUT_ERROR
    assert "$ref 'steps.seed.result.missing' failed" in str(exc.value)
