from __future__ import annotations

import numpy as np
import pytest

from tess_vetter.platform.catalogs.exoplanet_archive import (
    ExoplanetArchiveClient,
    KnownPlanet,
    KnownPlanetsResult,
)


def _planet(name: str, period: float) -> KnownPlanet:
    return KnownPlanet(
        name=name,
        period=float(period),
        period_err=None,
        t0=2000.0,
        t0_err=None,
        duration_hours=2.0,
        depth_ppm=300.0,
        radius_earth=2.0,
        status="CONFIRMED",
        disposition="CP",
        reference=None,
        discovery_facility="TESS",
    )


def test_match_known_planet_ephemeris_confirmed_same_planet(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ExoplanetArchiveClient()
    monkeypatch.setattr(
        client,
        "get_known_planets",
        lambda **_kwargs: KnownPlanetsResult(
            tic_id=100,
            n_planets=1,
            planets=[_planet("TOI-411 c", 9.57307)],
            toi_id="TOI-411",
        ),
    )

    match = client.match_known_planet_ephemeris(tic_id=100, period_days=9.57308)
    assert match.status == "confirmed_same_planet"
    assert match.matched_planet is not None
    assert match.matched_planet.name == "TOI-411 c"


def test_match_known_planet_ephemeris_same_star_different_period(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ExoplanetArchiveClient()
    monkeypatch.setattr(
        client,
        "get_known_planets",
        lambda **_kwargs: KnownPlanetsResult(
            tic_id=100,
            n_planets=2,
            planets=[_planet("TOI-411 b", 4.0), _planet("TOI-411 c", 12.0)],
            toi_id="TOI-411",
        ),
    )

    match = client.match_known_planet_ephemeris(
        tic_id=100,
        period_days=9.57,
        period_tolerance_days=0.001,
        period_tolerance_fraction=0.0001,
    )
    assert match.status == "confirmed_same_star_different_period"
    assert match.matched_planet is None


def test_match_known_planet_ephemeris_no_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ExoplanetArchiveClient()
    monkeypatch.setattr(
        client,
        "get_known_planets",
        lambda **_kwargs: KnownPlanetsResult(
            tic_id=100,
            n_planets=0,
            planets=[],
            toi_id="TOI-411",
        ),
    )

    match = client.match_known_planet_ephemeris(tic_id=100, period_days=9.57)
    assert match.status == "no_confirmed_match"
    assert match.confirmed_planets == []


def test_match_known_planet_ephemeris_ambiguous_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ExoplanetArchiveClient()
    monkeypatch.setattr(
        client,
        "get_known_planets",
        lambda **_kwargs: KnownPlanetsResult(
            tic_id=100,
            n_planets=2,
            planets=[_planet("TOI-XYZ b", 9.570), _planet("TOI-XYZ c", 9.575)],
            toi_id="TOI-XYZ",
        ),
    )

    match = client.match_known_planet_ephemeris(
        tic_id=100,
        period_days=9.573,
        period_tolerance_days=0.01,
        period_tolerance_fraction=0.0,
    )
    assert match.status == "ambiguous_multi_match"
    assert len(match.matched_planets) == 2


def test_bjd_to_btjd_conversion_handles_absolute_and_offset_values() -> None:
    client = ExoplanetArchiveClient()
    assert client._bjd_to_btjd(2459001.5) == pytest.approx(2001.5)
    assert client._bjd_to_btjd(2001.5) == pytest.approx(2001.5)


def test_parse_ps_record_preserves_row_with_missing_transit_epoch() -> None:
    client = ExoplanetArchiveClient()
    row = {
        "pl_name": "TOI-123 b",
        "pl_orbper": 5.0,
        "pl_tranmid": None,
    }
    parsed = client._parse_ps_record(row)
    assert parsed is not None
    assert parsed.name == "TOI-123 b"
    assert np.isnan(parsed.t0)
