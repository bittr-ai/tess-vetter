from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
import pytest

from tess_vetter.platform.io.mast_client import MASTClient, MASTClientError


class _FakeTPF:
    def __init__(self) -> None:
        self.time = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        self.flux = np.ones((10, 3, 3), dtype=np.float64)
        self.flux_err = np.ones((10, 3, 3), dtype=np.float64) * 0.1
        self.quality = np.zeros(10, dtype=np.int32)
        self.wcs = None
        self.pipeline_mask = np.ones((3, 3), dtype=bool)
        primary = fits.PrimaryHDU()
        table = fits.BinTableHDU.from_columns([fits.Column(name="TIME", format="D", array=self.time)])
        self.hdu = fits.HDUList([primary, table])


class _FakeAbsoluteBjdTPF(_FakeTPF):
    def __init__(self) -> None:
        super().__init__()
        self.time = np.linspace(2459001.0, 2459010.0, 10, dtype=np.float64)
        self.hdu[1].data["TIME"] = self.time


class _FakeRefOffsetTPF(_FakeTPF):
    def __init__(self) -> None:
        super().__init__()
        self.time = np.linspace(1.0, 10.0, 10, dtype=np.float64)
        self.hdu[1].data["TIME"] = self.time
        self.hdu[1].header["BJDREFI"] = 2458000
        self.hdu[1].header["BJDREFF"] = 0.0


class _FakeUtcTimesysTPF(_FakeTPF):
    def __init__(self) -> None:
        super().__init__()
        self.hdu[1].header["TIMESYS"] = "UTC"
        self.hdu[1].header["BJDREFI"] = 2457000
        self.hdu[1].header["BJDREFF"] = 0.0


class _FakeRow:
    def __init__(self, *, exptime: float, distance: float) -> None:
        self.exptime = exptime
        self.distance = distance
        self._downloaded = False

    def download(self):
        self._downloaded = True
        return _FakeTPF()


class _FakeAbsoluteBjdRow(_FakeRow):
    def download(self):
        self._downloaded = True
        return _FakeAbsoluteBjdTPF()


class _FakeRefOffsetRow(_FakeRow):
    def download(self):
        self._downloaded = True
        return _FakeRefOffsetTPF()


class _FakeUtcTimesysRow(_FakeRow):
    def download(self):
        self._downloaded = True
        return _FakeUtcTimesysTPF()


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


class _ArrayWithValue:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class _FakeCachedTPF:
    def __init__(self) -> None:
        time = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        self.time = _ArrayWithValue(time)
        self.flux = _ArrayWithValue(np.ones((10, 3, 3), dtype=np.float64))
        self.flux_err = _ArrayWithValue(np.ones((10, 3, 3), dtype=np.float64) * 0.1)
        self.quality = np.zeros(10, dtype=np.int32)
        self.wcs = None
        self.pipeline_mask = np.ones((3, 3), dtype=bool)


class _FakeLightkurveReader:
    def read(self, _path: str):
        return _FakeCachedTPF()


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


def test_download_tpf_normalizes_absolute_bjd_time(monkeypatch) -> None:
    rows = [_FakeAbsoluteBjdRow(exptime=120.0, distance=1.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    time, *_rest = client.download_tpf(1, sector=1, exptime=None)
    assert time[0] == 2001.0
    assert time[-1] == 2010.0


def test_download_tpf_uses_bjdref_when_time_is_relative(monkeypatch) -> None:
    rows = [_FakeRefOffsetRow(exptime=120.0, distance=1.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    time, *_rest = client.download_tpf(1, sector=1, exptime=None)
    assert time[0] == 1001.0
    assert time[-1] == 1010.0


def test_download_tpf_strict_timesys_policy_rejects_non_tdb(monkeypatch) -> None:
    rows = [_FakeUtcTimesysRow(exptime=120.0, distance=1.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient(tpf_timesys_policy="strict")
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    with pytest.raises(MASTClientError, match="TIMESYS=UTC"):
        client.download_tpf(1, sector=1, exptime=None)


def test_download_tpf_warn_timesys_policy_converts_non_tdb(monkeypatch, caplog) -> None:
    rows = [_FakeUtcTimesysRow(exptime=120.0, distance=1.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient(tpf_timesys_policy="warn")
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    time, *_rest = client.download_tpf(1, sector=1, exptime=None)
    assert "TIMESYS=UTC" in caplog.text
    assert not np.allclose(time, np.linspace(0.0, 1.0, 10, dtype=np.float64))


def test_download_tpf_off_timesys_policy_does_not_warn(monkeypatch, caplog) -> None:
    rows = [_FakeUtcTimesysRow(exptime=120.0, distance=1.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient(tpf_timesys_policy="off")
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    _time, *_rest = client.download_tpf(1, sector=1, exptime=None)
    assert "TIMESYS=UTC" not in caplog.text


def _write_cached_tpf_with_timesys(path: Path, *, timesys: str) -> None:
    time = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    flux = np.ones((10, 9), dtype=np.float64)
    qual = np.zeros(10, dtype=np.int32)
    cols = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="FLUX", format="9D", dim="(3,3)", array=flux),
        fits.Column(name="QUALITY", format="J", array=qual),
    ]
    primary = fits.PrimaryHDU()
    table = fits.BinTableHDU.from_columns(cols)
    table.header["TIMESYS"] = timesys
    table.header["BJDREFI"] = 2457000
    table.header["BJDREFF"] = 0.0
    fits.HDUList([primary, table]).writeto(path, overwrite=True)


def test_download_tpf_cached_strict_timesys_policy_rejects_non_tdb(
    monkeypatch, tmp_path: Path
) -> None:
    mast_root = tmp_path / "mastDownload" / "TESS"
    target_dir = mast_root / "tess2018-s0001-0000000000000001-0123-a_tp"
    target_dir.mkdir(parents=True)
    fits_path = target_dir / "tess2018-s0001-0000000000000001-0123-a_tp.fits"
    _write_cached_tpf_with_timesys(fits_path, timesys="UTC")

    client = MASTClient(cache_dir=str(tmp_path), tpf_timesys_policy="strict")
    client._cache_index_built = False
    client._cache_dirs_by_tic.clear()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: _FakeLightkurveReader())

    with pytest.raises(MASTClientError, match="TIMESYS=UTC"):
        client.download_tpf_cached(1, sector=1)


def test_download_tpf_cached_warn_timesys_policy_logs_and_converts(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    mast_root = tmp_path / "mastDownload" / "TESS"
    target_dir = mast_root / "tess2018-s0001-0000000000000001-0123-a_tp"
    target_dir.mkdir(parents=True)
    fits_path = target_dir / "tess2018-s0001-0000000000000001-0123-a_tp.fits"
    _write_cached_tpf_with_timesys(fits_path, timesys="UTC")

    client = MASTClient(cache_dir=str(tmp_path), tpf_timesys_policy="warn")
    client._cache_index_built = False
    client._cache_dirs_by_tic.clear()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: _FakeLightkurveReader())

    time, *_rest = client.download_tpf_cached(1, sector=1)
    assert "TIMESYS=UTC" in caplog.text
    assert not np.allclose(time, np.linspace(0.0, 1.0, 10, dtype=np.float64))


def test_download_tpf_cached_off_timesys_policy_no_warning(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    mast_root = tmp_path / "mastDownload" / "TESS"
    target_dir = mast_root / "tess2018-s0001-0000000000000001-0123-a_tp"
    target_dir.mkdir(parents=True)
    fits_path = target_dir / "tess2018-s0001-0000000000000001-0123-a_tp.fits"
    _write_cached_tpf_with_timesys(fits_path, timesys="UTC")

    client = MASTClient(cache_dir=str(tmp_path), tpf_timesys_policy="off")
    client._cache_index_built = False
    client._cache_dirs_by_tic.clear()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: _FakeLightkurveReader())

    _time, *_rest = client.download_tpf_cached(1, sector=1)
    assert "TIMESYS=UTC" not in caplog.text


def test_client_without_override_rereads_timesys_policy_env(monkeypatch) -> None:
    rows = [_FakeUtcTimesysRow(exptime=120.0, distance=1.0)]
    fake_lk = _FakeLightkurve(rows)
    client = MASTClient()
    monkeypatch.setattr(client, "_ensure_lightkurve", lambda: fake_lk)

    monkeypatch.setenv("BTV_TPF_TIMESYS_POLICY", "off")
    _time, *_rest = client.download_tpf(1, sector=1, exptime=None)

    monkeypatch.setenv("BTV_TPF_TIMESYS_POLICY", "strict")
    with pytest.raises(MASTClientError, match="TIMESYS=UTC"):
        client.download_tpf(1, sector=1, exptime=None)
