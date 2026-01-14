from __future__ import annotations

from pathlib import Path

from bittr_tess_vetter.api.ephemeris_match import (
    EphemerisEntry,
    EphemerisIndex,
    load_index,
    run_ephemeris_matching,
    save_index,
)


def test_ephemeris_match_strong_1_to_1() -> None:
    index = EphemerisIndex(
        [
            EphemerisEntry(
                source_id="TOI-123",
                source_type="toi",
                period=5.0,
                t0=1325.5,
                ra_deg=10.0,
                dec_deg=20.0,
            )
        ]
    )
    index.build_index()

    result = run_ephemeris_matching(
        candidate_period=5.0,
        candidate_t0=1325.5,
        index=index,
        candidate_ra=10.0,
        candidate_dec=20.0,
    )

    assert result.match_class == "EPHEMERIS_MATCH_STRONG"
    assert result.best_match is not None
    assert result.best_match.harmonic_relation == "1:1"
    assert result.best_match.separation_arcsec is not None
    assert result.best_match.match_score > 0.99


def test_ephemeris_match_harmonic_2_to_1() -> None:
    index = EphemerisIndex(
        [
            EphemerisEntry(
                source_id="EB-456",
                source_type="eb",
                period=5.0,
                t0=100.0,
            )
        ]
    )
    index.build_index()

    result = run_ephemeris_matching(
        candidate_period=10.0,
        candidate_t0=100.0,
        index=index,
    )

    assert result.best_match is not None
    assert result.best_match.harmonic_relation == "2:1"
    assert result.match_class in ("EPHEMERIS_MATCH_STRONG", "EPHEMERIS_MATCH_WEAK")


def test_ephemeris_index_save_load_roundtrip(tmp_path: Path) -> None:
    index = EphemerisIndex(
        [
            EphemerisEntry(
                source_id="TOI-1",
                source_type="toi",
                period=1.234,
                t0=2000.0,
                metadata={"tic_id": "123"},
            )
        ]
    )
    index.build_index()

    path = tmp_path / "index.json"
    save_index(index, path)

    loaded = load_index(path)
    assert len(loaded) == 1
    assert loaded.entries[0].source_id == "TOI-1"
    assert loaded.entries[0].metadata["tic_id"] == "123"
