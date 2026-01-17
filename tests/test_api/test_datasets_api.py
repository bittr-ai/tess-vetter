from __future__ import annotations

from pathlib import Path

import numpy as np

from bittr_tess_vetter.api.datasets import load_local_dataset


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
    assert ds.schema_version == 1
    assert 1 in ds.lc_by_sector
    assert 1 in ds.tpf_by_sector

    lc = ds.lc_by_sector[1]
    # quality==1 cadence should be filtered out
    assert len(lc.time) == 2
    assert len(lc.flux) == 2

    tpf = ds.tpf_by_sector[1]
    assert tpf.flux.shape == (2, 2, 2)
    assert tpf.aperture_mask is not None
    assert ds.summary()["sectors_lc"] == [1]
    assert ds.summary()["sectors_tpf"] == [1]

