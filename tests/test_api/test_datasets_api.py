from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tess_vetter.api.datasets as datasets_api
from tess_vetter.api.contracts import callable_input_schema_from_signature
from tess_vetter.api.datasets import (
    LOAD_TUTORIAL_TARGET_BOUNDARY_CONTRACT,
    LOAD_TUTORIAL_TARGET_CALL_SCHEMA,
    LOAD_TUTORIAL_TARGET_DATA_ROOT_PARTS,
    LOAD_TUTORIAL_TARGET_REQUIRED_FIELDS,
    LOAD_TUTORIAL_TARGET_SCHEMA_VERSION,
    LOCAL_DATASET_PATTERN_DEFAULTS,
    LOCAL_DATASET_SCHEMA_VERSION,
    LocalDatasetLoadPayload,
    load_local_dataset,
    load_tutorial_target,
)


def _write_sector_csv(path: Path, *, sector: int) -> None:
    content = "\n".join(
        [
            "# tutorial csv",
            "time_btjd,flux,flux_err,quality",
            "0.0,1.0,0.001,0",
            "0.5,1.0,0.001,1",
            "1.0,1.0,0.001,0",
            "",
        ]
    )
    (path / f"sector{sector}_pdcsap.csv").write_text(content)


def _write_sector_tpf_npz(path: Path, *, sector: int) -> None:
    time = np.array([0.0, 0.5], dtype=np.float64)
    flux = np.ones((2, 2, 2), dtype=np.float64)
    flux_err = np.full_like(flux, 1e-3)
    aperture_mask = np.array([[True, False], [False, True]])
    quality = np.array([0, 0], dtype=np.int32)
    wcs_header = {"WCSAXES": 2}
    np.savez(
        path / f"sector{sector}_tpf.npz",
        time=time,
        flux=flux,
        flux_err=flux_err,
        aperture_mask=aperture_mask,
        quality=quality,
        wcs_header=wcs_header,
    )


def test_load_local_dataset_parses_csv_and_npz(tmp_path: Path) -> None:
    _write_sector_csv(tmp_path, sector=1)
    _write_sector_tpf_npz(tmp_path, sector=1)

    ds = load_local_dataset(tmp_path)
    assert ds.schema_version == LOCAL_DATASET_SCHEMA_VERSION
    assert 1 in ds.lc_by_sector
    assert 1 in ds.tpf_by_sector

    lc = ds.lc_by_sector[1]
    # quality==1 cadence should be filtered out
    assert len(lc.time) == 2
    assert len(lc.flux) == 2

    tpf = ds.tpf_by_sector[1]
    assert tpf.flux.shape == (2, 2, 2)
    assert tpf.aperture_mask is not None
    summary = ds.summary()
    assert summary["schema_version"] == LOCAL_DATASET_SCHEMA_VERSION
    assert summary["sectors_lc"] == [1]
    assert summary["sectors_tpf"] == [1]
    assert summary["artifacts"] == ["files"]
    assert summary["root"] == str(tmp_path.resolve())


def test_load_local_dataset_payload_contract(tmp_path: Path) -> None:
    _write_sector_csv(tmp_path, sector=2)
    _write_sector_tpf_npz(tmp_path, sector=2)

    payload: LocalDatasetLoadPayload = load_local_dataset(tmp_path).load_payload()
    assert payload["schema_version"] == LOCAL_DATASET_SCHEMA_VERSION
    assert payload["root"] == tmp_path.resolve()
    assert sorted(payload["lc_by_sector"]) == [2]
    assert sorted(payload["tpf_by_sector"]) == [2]
    assert payload["artifacts"]["files"] == sorted(p.name for p in tmp_path.iterdir() if p.is_file())


def test_pattern_overrides_do_not_mutate_defaults(tmp_path: Path) -> None:
    (tmp_path / "sector9_custom.csv").write_text(
        "\n".join(
            [
                "# tutorial csv",
                "time_btjd,flux,flux_err,quality",
                "0.0,1.0,0.001,0",
                "",
            ]
        )
    )

    defaults_before = dict(LOCAL_DATASET_PATTERN_DEFAULTS)
    ds = load_local_dataset(tmp_path, pattern_overrides={"lc_csv": "sector{sector}_custom.csv"})
    assert 9 in ds.lc_by_sector
    assert defaults_before == LOCAL_DATASET_PATTERN_DEFAULTS


def test_load_tutorial_target_contract_constants_are_stable() -> None:
    assert LOAD_TUTORIAL_TARGET_SCHEMA_VERSION == 1
    assert LOAD_TUTORIAL_TARGET_REQUIRED_FIELDS == ("name",)
    assert LOAD_TUTORIAL_TARGET_DATA_ROOT_PARTS == ("docs", "tutorials", "data")
    assert LOAD_TUTORIAL_TARGET_BOUNDARY_CONTRACT == {
        "schema_version": LOAD_TUTORIAL_TARGET_SCHEMA_VERSION,
        "required_fields": ("name",),
        "data_root_parts": LOAD_TUTORIAL_TARGET_DATA_ROOT_PARTS,
    }

    expected = {
        "type": "object",
        "properties": {"name": {}},
        "additionalProperties": False,
        "required": ["name"],
    }
    assert expected == LOAD_TUTORIAL_TARGET_CALL_SCHEMA
    assert callable_input_schema_from_signature(load_tutorial_target) == LOAD_TUTORIAL_TARGET_CALL_SCHEMA


def test_load_tutorial_target_delegates_to_local_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Path] = {}
    sentinel = datasets_api.LocalDataset(
        schema_version=LOCAL_DATASET_SCHEMA_VERSION,
        root=Path("/tmp/example"),
    )

    def _stub(path: str | Path, *, pattern_overrides: object = None) -> datasets_api.LocalDataset:
        assert pattern_overrides is None
        captured["path"] = Path(path)
        return sentinel

    monkeypatch.setattr(datasets_api, "load_local_dataset", _stub)
    out = datasets_api.load_tutorial_target("tic_123")
    expected_base = Path(datasets_api.__file__).resolve().parents[3].joinpath(
        *LOAD_TUTORIAL_TARGET_DATA_ROOT_PARTS
    )
    assert out is sentinel
    assert captured["path"] == expected_base / "tic_123"
