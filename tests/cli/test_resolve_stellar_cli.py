from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
from bittr_tess_vetter.platform.catalogs.toi_resolution import LookupStatus


def test_resolve_stellar_with_tic_id_writes_payload(monkeypatch) -> None:
    def _fake_auto(*, tic_id: int, toi: str | None = None):
        assert tic_id == 123
        assert toi is None
        return (
            {"radius": 1.0, "mass": 0.9, "tmag": 10.5},
            {"selected_source": "tic_mast", "echo_of_tic": False},
        )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_stellar_cli.load_auto_stellar_with_fallback",
        _fake_auto,
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "resolve-stellar",
            "--tic-id",
            "123",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "cli.resolve-stellar.v1"
    assert payload["tic_id"] == 123
    assert payload["stellar"]["radius"] == 1.0


def test_resolve_stellar_with_toi_adds_echo_note(monkeypatch) -> None:
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_stellar_cli.resolve_toi_to_tic_ephemeris_depth",
        lambda toi: SimpleNamespace(
            status=LookupStatus.OK,
            toi_query=toi,
            tic_id=321,
            matched_toi="5639.01",
            message=None,
            missing_fields=[],
            source_record=SimpleNamespace(model_dump=lambda **_: {"name": "exofop_toi_table"}),
        ),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_stellar_cli.load_auto_stellar_with_fallback",
        lambda **_: (
            {"radius": 0.8, "mass": 0.7, "tmag": 12.0},
            {"selected_source": "exofop_toi_table", "echo_of_tic": True},
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "resolve-stellar",
            "--toi",
            "TOI-5639.01",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tic_id"] == 321
    assert "source continuity" in payload["note"]
