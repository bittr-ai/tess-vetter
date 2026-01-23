from __future__ import annotations

import numpy as np

from bittr_tess_vetter.platform.io.mast_client import MASTClient


class _FakeTPF:
    def __init__(self) -> None:
        self.time = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        self.flux = np.ones((10, 3, 3), dtype=np.float64)
        self.flux_err = np.ones((10, 3, 3), dtype=np.float64) * 0.1
        self.quality = np.zeros(10, dtype=np.int32)
        self.wcs = None
        self.pipeline_mask = np.ones((3, 3), dtype=bool)


class _FakeRow:
    def __init__(self, *, exptime: float, distance: float) -> None:
        self.exptime = exptime
        self.distance = distance
        self._downloaded = False

    def download(self):
        self._downloaded = True
        return _FakeTPF()


class _FakeSearchResult:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> _FakeRow:
        return self._rows[idx]


class _FakeLightkurve:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows

    def search_targetpixelfile(self, *args, **kwargs):
        return _FakeSearchResult(self._rows)


def test_download_tpf_prefers_120s_when_exptime_none(monkeypatch) -> None:
    rows = [_FakeRow(exptime=20.0, distance=10.0), _FakeRow(exptime=120.0, distance=5.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    time, flux, flux_err, wcs, aperture, quality = client.download_tpf(1, sector=1, exptime=None)
    assert time.shape == (10,)
    assert flux.shape == (10, 3, 3)
    assert flux_err is not None
    assert aperture is not None
    assert quality is not None
    assert wcs is None

    # Should select the 120s product.
    assert rows[1]._downloaded is True
    assert rows[0]._downloaded is False


def test_download_tpf_filters_by_requested_exptime(monkeypatch) -> None:
    rows = [_FakeRow(exptime=20.0, distance=10.0), _FakeRow(exptime=120.0, distance=5.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    _time, _flux, _flux_err, _wcs, _aperture, _quality = client.download_tpf(
        1, sector=1, exptime=20.0
    )

    assert rows[0]._downloaded is True
    assert rows[1]._downloaded is False

