from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli


def _base_vet_args() -> list[str]:
    return [
        "vet",
        "--tic-id",
        "123",
        "--period-days",
        "10.5",
        "--t0-btjd",
        "2000.2",
        "--duration-hours",
        "2.5",
    ]


def _base_measure_args() -> list[str]:
    return [
        "measure-sectors",
        "--tic-id",
        "123",
        "--period-days",
        "10.5",
        "--t0-btjd",
        "2000.2",
        "--duration-hours",
        "2.5",
    ]


def _patch_measure_sectors_execute(monkeypatch, fake_impl) -> str:
    """Patch whichever measure-sectors execution seam exists in this revision."""
    seam_candidates = [
        ("bittr_tess_vetter.cli.measure_sectors_cli", "_execute_measure_sectors"),
        ("bittr_tess_vetter.cli.measure_sectors_cli", "_execute_measurements"),
        ("bittr_tess_vetter.cli.vet_cli", "_execute_measure_sectors"),
        ("bittr_tess_vetter.cli.vet_cli", "_execute_measurements"),
    ]

    for module_name, attr_name in seam_candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, attr_name):
            monkeypatch.setattr(f"{module_name}.{attr_name}", fake_impl)
            return f"{module_name}.{attr_name}"

    pytest.fail(
        "No measure-sectors execution seam found. "
        "Expected one of: _execute_measure_sectors/_execute_measurements "
        "in cli.measure_sectors_cli or cli.vet_cli."
    )


def test_cli007_measure_sectors_emits_schema_and_provenance(monkeypatch, tmp_path: Path) -> None:
    expected = {
        "schema_version": 1,
        "sector_measurements": [
            {"sector": 82, "depth_ppm": 410.2, "depth_err_ppm": 21.3},
            {"sector": 83, "depth_ppm": 432.1, "depth_err_ppm": 19.7},
        ],
        "provenance": {
            "tic_id": 123,
            "sectors_requested": [82, 83],
            "sectors_used": [82, 83],
            "exclusions": {},
            "generator": "test-double",
        },
    }

    def _fake_measure(**_kwargs: Any) -> dict[str, Any]:
        return expected

    _patch_measure_sectors_execute(monkeypatch, _fake_measure)

    out_path = tmp_path / "sectors.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [*_base_measure_args(), "--sectors", "82", "--sectors", "83", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert isinstance(payload.get("sector_measurements"), list)
    assert len(payload["sector_measurements"]) == 2
    assert set(payload["sector_measurements"][0].keys()) >= {"sector", "depth_ppm", "depth_err_ppm"}

    provenance = payload.get("provenance")
    assert isinstance(provenance, dict)
    assert provenance.get("sectors_requested") == [82, 83]
    assert provenance.get("sectors_used") == [82, 83]


def test_cli007_cli008_vet_with_sector_measurements_injects_context_and_emits_blocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    measurements_path = tmp_path / "sectors.json"
    measurements_payload = {
        "schema_version": 1,
        "sector_measurements": [
            {"sector": 82, "depth_ppm": 410.2, "depth_err_ppm": 21.3},
            {"sector": 83, "depth_ppm": 432.1, "depth_err_ppm": 19.7},
        ],
        "provenance": {
            "sectors_requested": [82, 83],
            "sectors_used": [82, 83],
            "exclusions": {},
        },
    }
    measurements_path.write_text(json.dumps(measurements_payload), encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_execute_vet(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "results": [],
            "warnings": [],
            "inputs_summary": {},
            "provenance": {"pipeline_version": "0.1.0"},
            "sector_measurements": measurements_payload["sector_measurements"],
            "sector_gating": {
                "sectors_requested": [82, 83],
                "sectors_used": [82, 83],
                "exclusions": {},
            },
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            *_base_vet_args(),
            "--preset",
            "extended",
            "--sector-measurements",
            str(measurements_path),
            "--sectors",
            "82",
            "--sectors",
            "83",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output

    # Accept either direct kwarg plumbed to execute seam or nested context handoff.
    if "sector_measurements" in captured:
        injected = captured["sector_measurements"]
    else:
        context = captured.get("context")
        assert isinstance(context, dict), "Expected context dict with sector_measurements"
        injected = context.get("sector_measurements")

    assert isinstance(injected, list)
    assert len(injected) == 2

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("sector_measurements"), list)
    assert isinstance(payload.get("sector_gating"), dict)
    assert payload["sector_gating"].get("n_input_rows") == 2


def test_cli007_malformed_sector_measurements_file_maps_to_exit_1(monkeypatch, tmp_path: Path) -> None:
    broken_path = tmp_path / "broken-sectors.json"
    broken_path.write_text("{not-json", encoding="utf-8")

    called = {"execute_vet": False}

    def _fake_execute_vet(**_kwargs: Any) -> dict[str, Any]:
        called["execute_vet"] = True
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [*_base_vet_args(), "--sector-measurements", str(broken_path)],
    )

    assert result.exit_code == 1
    assert called["execute_vet"] is False


def test_cli008_requested_sectors_unavailable_maps_to_exit_4_or_documented_mapping(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _missing_requested(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Requested sectors unavailable: [82, 83]")

    _patch_measure_sectors_execute(monkeypatch, _missing_requested)

    out_path = tmp_path / "sectors.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [*_base_measure_args(), "--sectors", "82", "--sectors", "83", "--out", str(out_path)],
    )

    # Preferred mapping is EXIT_DATA_UNAVAILABLE=4. Some implementations currently
    # surface this through the generic runtime path (EXIT_RUNTIME_ERROR=2).
    assert result.exit_code in {2, 4}, result.output
