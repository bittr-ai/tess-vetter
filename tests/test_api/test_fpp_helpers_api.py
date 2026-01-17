from __future__ import annotations

from pathlib import Path

import numpy as np

import bittr_tess_vetter.api as btv


def test_load_contrast_curve_exofop_tbl_parses_numeric_rows(tmp_path: Path) -> None:
    p = tmp_path / "cc.tbl"
    p.write_text(
        "\n".join(
            [
                "# comment",
                "not,a,number",
                "0.10 1.5 0.1",
                "0.50, 4.2, 0.2",
                "1.00 6.0",
                "",
            ]
        )
    )

    cc = btv.load_contrast_curve_exofop_tbl(p, filter="Kcont")
    assert cc.filter == "Kcont"
    assert cc.separation_arcsec.shape == cc.delta_mag.shape
    assert cc.separation_arcsec.size == 3
    assert float(cc.separation_arcsec[0]) == 0.10
    assert float(cc.delta_mag[-1]) == 6.0


def test_hydrate_cache_from_dataset_puts_sector_keys(tmp_path: Path) -> None:
    ds = btv.LocalDataset(schema_version=1, root=tmp_path)
    ds.lc_by_sector[1] = btv.LightCurve(
        time=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        flux=np.ones(3, dtype=np.float64),
        flux_err=np.full(3, 1e-3, dtype=np.float64),
    )
    ds.lc_by_sector[2] = btv.LightCurve(
        time=np.array([10.0, 10.5], dtype=np.float64),
        flux=np.ones(2, dtype=np.float64),
        flux_err=np.full(2, 1e-3, dtype=np.float64),
    )

    cache_dir = tmp_path / "cache"
    cache = btv.hydrate_cache_from_dataset(dataset=ds, tic_id=188646744, cache_dir=cache_dir)

    k1 = btv.make_data_ref(188646744, 1, "pdcsap")
    k2 = btv.make_data_ref(188646744, 2, "pdcsap")
    assert cache.has(k1)
    assert cache.has(k2)
