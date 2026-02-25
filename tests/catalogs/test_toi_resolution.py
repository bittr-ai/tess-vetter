from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tess_vetter.platform.catalogs.exofop_toi_table import ExoFOPToiTable
from tess_vetter.platform.catalogs.models import SourceRecord
from tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    lookup_tic_coordinates,
    resolve_toi_to_tic_ephemeris_depth,
)


def _source(name: str, query: str) -> SourceRecord:
    return SourceRecord(name=name, version="test", retrieved_at=datetime.now(UTC), query=query)


def test_resolve_toi_to_tic_ephemeris_depth_success(monkeypatch: pytest.MonkeyPatch) -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "tic_id", "period", "epoch", "duration", "depth_ppm"],
        rows=[
            {
                "toi": "5807.01",
                "tic_id": "188646744",
                "period": "6.1234",
                "epoch": "2459001.5",
                "duration": "2.5",
                "depth_ppm": "430.0",
            }
        ],
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table_for_toi",
        lambda *_args, **_kwargs: table,
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table",
        lambda **_: table,
    )

    result = resolve_toi_to_tic_ephemeris_depth("TOI-5807.01")
    assert result.status == LookupStatus.OK
    assert result.tic_id == 188646744
    assert result.period_days == pytest.approx(6.1234)
    assert result.t0_btjd == pytest.approx(2001.5)
    assert result.duration_hours == pytest.approx(2.5)
    assert result.depth_ppm == pytest.approx(430.0)


def test_resolve_toi_to_tic_ephemeris_depth_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "tic_id"],
        rows=[{"toi": "100.01", "tic_id": "1"}],
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table_for_toi",
        lambda *_args, **_kwargs: table,
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table",
        lambda **_: table,
    )

    result = resolve_toi_to_tic_ephemeris_depth("TOI-999.01")
    assert result.status == LookupStatus.DATA_UNAVAILABLE
    assert result.tic_id is None


def test_resolve_toi_to_tic_ephemeris_depth_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_timeout(**_: object) -> ExoFOPToiTable:
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table_for_toi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table",
        _raise_timeout,
    )
    result = resolve_toi_to_tic_ephemeris_depth("123.01")
    assert result.status == LookupStatus.TIMEOUT


def test_resolve_toi_to_tic_ephemeris_depth_falls_back_to_full_table(monkeypatch: pytest.MonkeyPatch) -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "tic_id", "period", "epoch", "duration", "depth_ppm"],
        rows=[
            {
                "toi": "5807.01",
                "tic_id": "188646744",
                "period": "6.1234",
                "epoch": "2459001.5",
                "duration": "2.5",
                "depth_ppm": "430.0",
            }
        ],
    )

    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table_for_toi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("scoped timeout")),
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table",
        lambda **_: table,
    )

    result = resolve_toi_to_tic_ephemeris_depth("TOI-5807.01")
    assert result.status == LookupStatus.OK
    assert result.tic_id == 188646744
    assert result.message is not None
    assert "TOI-scoped fetch failed" in result.message


def test_resolve_toi_to_tic_ephemeris_depth_preserves_btjd_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "tic_id", "period", "epoch_btjd", "duration", "depth_ppm"],
        rows=[
            {
                "toi": "5807.01",
                "tic_id": "188646744",
                "period": "6.1234",
                "epoch_btjd": "2001.5",
                "duration": "2.5",
                "depth_ppm": "430.0",
            }
        ],
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table_for_toi",
        lambda *_args, **_kwargs: table,
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution.fetch_exofop_toi_table",
        lambda **_: table,
    )

    result = resolve_toi_to_tic_ephemeris_depth("TOI-5807.01")
    assert result.status == LookupStatus.OK
    assert result.t0_btjd == pytest.approx(2001.5)


def test_lookup_tic_coordinates_fallback_to_exofop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution._lookup_tic_coords_from_mast",
        lambda tic_id, **_: (None, None, _source("mast_tic", f"TIC {tic_id}"), LookupStatus.DATA_UNAVAILABLE, "no"),
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution._lookup_tic_coords_from_exofop",
        lambda tic_id, **_: (123.4, -45.6, _source("exofop_toi_table", f"TIC {tic_id}"), LookupStatus.OK, None),
    )

    result = lookup_tic_coordinates(42)
    assert result.status == LookupStatus.OK
    assert result.ra_deg == pytest.approx(123.4)
    assert result.dec_deg == pytest.approx(-45.6)
    assert result.source_record is not None
    assert result.source_record.name == "exofop_toi_table"


def test_lookup_tic_coordinates_surfaces_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution._lookup_tic_coords_from_mast",
        lambda tic_id, **_: (
            None,
            None,
            _source("mast_tic", f"TIC {tic_id}"),
            LookupStatus.TIMEOUT,
            "timeout",
        ),
    )
    monkeypatch.setattr(
        "tess_vetter.platform.catalogs.toi_resolution._lookup_tic_coords_from_exofop",
        lambda tic_id, **_: (
            None,
            None,
            _source("exofop_toi_table", f"TIC {tic_id}"),
            LookupStatus.DATA_UNAVAILABLE,
            "missing",
        ),
    )

    result = lookup_tic_coordinates(99)
    assert result.status == LookupStatus.TIMEOUT
    assert result.message == "timeout"
