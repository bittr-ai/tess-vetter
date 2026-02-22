from __future__ import annotations

from pathlib import Path

import numpy as np

from tess_vetter.api.datasets import (
    LOCAL_DATASET_PATTERN_DEFAULTS,
    LOCAL_DATASET_SCHEMA_VERSION,
    LocalDatasetLoadPayload,
    load_local_dataset,
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
