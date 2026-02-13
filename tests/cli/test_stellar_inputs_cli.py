from __future__ import annotations

import json
from pathlib import Path

import pytest

from bittr_tess_vetter.cli.common_cli import BtvCliError
from bittr_tess_vetter.cli.stellar_inputs import load_stellar_inputs_file, resolve_stellar_inputs


def test_load_stellar_inputs_file_reads_nested_stellar_block(tmp_path: Path) -> None:
    path = tmp_path / "stellar.json"
    path.write_text(json.dumps({"stellar": {"radius": 1.1, "mass": 0.9, "tmag": 10.5}}), encoding="utf-8")

    values, meta = load_stellar_inputs_file(path)
    assert values == {"radius": 1.1, "mass": 0.9, "tmag": 10.5}
    assert meta["path"] == str(path)


def test_resolve_stellar_inputs_fieldwise_precedence() -> None:
    values, provenance = resolve_stellar_inputs(
        tic_id=123,
        stellar_radius=1.2,
        stellar_mass=None,
        stellar_tmag=None,
        stellar_file=None,
        use_stellar_auto=True,
        require_stellar=False,
        auto_loader=lambda _tic_id: {"radius": 0.8, "mass": 0.7, "tmag": 11.0},
    )
    assert values["radius"] == 1.2
    assert values["mass"] == 0.7
    assert values["tmag"] == 11.0
    assert provenance["sources"] == {"radius": "explicit", "mass": "auto", "tmag": "auto"}


def test_resolve_stellar_inputs_require_stellar_raises_when_missing() -> None:
    with pytest.raises(BtvCliError) as exc:
        resolve_stellar_inputs(
            tic_id=123,
            stellar_radius=None,
            stellar_mass=None,
            stellar_tmag=None,
            stellar_file=None,
            use_stellar_auto=False,
            require_stellar=True,
            auto_loader=None,
        )
    assert exc.value.exit_code == 4
