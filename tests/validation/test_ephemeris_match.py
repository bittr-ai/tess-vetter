from __future__ import annotations

from pathlib import Path

import pytest

from tess_vetter.validation.ephemeris_match import (
    EphemerisEntry,
    EphemerisIndex,
    EphemerisMatch,
    build_index_from_csv,
    classify_matches,
    compute_harmonic_match,
    load_index,
    run_ephemeris_matching,
    save_index,
    wrap_t0,
)


def test_entry_and_wrap_t0_reject_non_positive_period() -> None:
    with pytest.raises(ValueError, match="period must be positive"):
        EphemerisEntry(source_id="bad", source_type="toi", period=0.0, t0=1.0)

    with pytest.raises(ValueError, match="period must be positive"):
        wrap_t0(1.0, 0.0)


def test_compute_harmonic_match_returns_none_when_no_relation() -> None:
    match = compute_harmonic_match(
        query_period=5.0,
        query_t0=0.0,
        entry_period=7.0,
        entry_t0=1.0,
        harmonics=[1, 2, 3],
        period_tolerance=1e-4,
        t0_tolerance=1e-4,
    )
    assert match is None


def test_compute_harmonic_match_finds_inverse_harmonic_1_to_k() -> None:
    match = compute_harmonic_match(
        query_period=5.0,
        query_t0=12.5,
        entry_period=10.0,
        entry_t0=12.5,
        harmonics=[2],
        period_tolerance=1e-6,
        t0_tolerance=1e-6,
    )
    assert match is not None
    relation, period_residual, t0_residual = match
    assert relation == "1:2"
    assert period_residual == pytest.approx(0.0)
    assert t0_residual == pytest.approx(0.0)


def test_compute_harmonic_match_uses_default_harmonics() -> None:
    match = compute_harmonic_match(
        query_period=5.0,
        query_t0=50.0,
        entry_period=5.0,
        entry_t0=50.0,
    )
    assert match is not None
    assert match[0] == "1:1"


def test_query_deduplicates_by_source_id_and_respects_max_results() -> None:
    index = EphemerisIndex(
        [
            EphemerisEntry(source_id="dup", source_type="toi", period=5.0, t0=100.0),
            EphemerisEntry(source_id="dup", source_type="toi", period=10.0, t0=100.0),
            EphemerisEntry(source_id="other", source_type="eb", period=10.0, t0=100.0),
        ]
    )

    matches = index.query(
        period=10.0,
        t0=100.0,
        harmonics=[1, 2],
        period_tolerance=1e-6,
        t0_tolerance=1e-6,
        max_results=1,
    )

    assert len(matches) == 1
    assert matches[0].source_entry.source_id == "dup"
    assert matches[0].harmonic_relation in {"1:1", "2:1"}


def test_index_add_entry_and_query_empty_or_nonmatch_paths() -> None:
    empty_index = EphemerisIndex()
    assert empty_index.query(period=2.0, t0=0.0) == []

    index = EphemerisIndex()
    index.add_entry(EphemerisEntry(source_id="x", source_type="toi", period=2.0, t0=101.0))

    # period matches search window, but phase mismatch with strict tolerance should skip append path
    assert (
        index.query(
            period=2.0,
            t0=0.0,
            t0_tolerance=1e-8,
        )
        == []
    )


def test_query_adds_sky_separation_only_when_all_coordinates_present() -> None:
    index = EphemerisIndex(
        [
            EphemerisEntry(
                source_id="with_coords",
                source_type="toi",
                period=5.0,
                t0=100.0,
                ra_deg=10.0,
                dec_deg=20.0,
            ),
            EphemerisEntry(
                source_id="without_coords",
                source_type="toi",
                period=5.0,
                t0=100.0,
                ra_deg=10.0,
                dec_deg=None,
            ),
        ]
    )

    matches = index.query(
        period=5.0,
        t0=100.0,
        harmonics=[1],
        period_tolerance=1e-6,
        t0_tolerance=1e-6,
        query_ra=10.0,
        query_dec=20.0,
    )
    by_id = {m.source_entry.source_id: m for m in matches}

    assert by_id["with_coords"].separation_arcsec == pytest.approx(0.0)
    assert by_id["without_coords"].separation_arcsec is None


def test_classify_matches_covers_none_weak_and_strong() -> None:
    cls, best = classify_matches([])
    assert cls == "NONE"
    assert best is None

    entry = EphemerisEntry(source_id="s", source_type="toi", period=2.0, t0=1.0)
    weak = EphemerisMatch(
        source_entry=entry,
        harmonic_relation="1:1",
        period_residual=0.0,
        t0_residual=0.0,
        match_score=0.6,
    )
    strong = EphemerisMatch(
        source_entry=entry,
        harmonic_relation="1:1",
        period_residual=0.0,
        t0_residual=0.0,
        match_score=0.95,
    )

    cls_weak, best_weak = classify_matches([weak], strong_threshold=0.9, weak_threshold=0.5)
    assert cls_weak == "EPHEMERIS_MATCH_WEAK"
    assert best_weak is weak

    cls_strong, best_strong = classify_matches([weak, strong], strong_threshold=0.9, weak_threshold=0.5)
    assert cls_strong == "EPHEMERIS_MATCH_STRONG"
    assert best_strong is strong

    cls_none_with_match, best_none_with_match = classify_matches([weak], strong_threshold=0.99, weak_threshold=0.8)
    assert cls_none_with_match == "NONE"
    assert best_none_with_match is weak


def test_run_ephemeris_matching_defaults_and_provenance() -> None:
    index = EphemerisIndex(
        [EphemerisEntry(source_id="a", source_type="toi", period=3.0, t0=10.0)]
    )
    result = run_ephemeris_matching(
        candidate_period=3.0,
        candidate_t0=10.0,
        index=index,
        candidate_id="cand-1",
    )

    assert result.candidate_id == "cand-1"
    assert result.match_class == "EPHEMERIS_MATCH_STRONG"
    assert result.best_match is not None
    assert result.provenance["harmonics"] == [1, 2, 3]
    assert result.provenance["index_size"] == 1
    assert result.provenance["num_matches"] == 1


def test_save_load_roundtrip_and_invalid_json(tmp_path: Path) -> None:
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
    path = tmp_path / "nested" / "index.json"
    save_index(index, path)

    loaded = load_index(path)
    assert len(loaded) == 1
    assert loaded.entries[0].metadata["tic_id"] == "123"

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"version":"1.0"}')
    with pytest.raises(ValueError, match="missing 'entries'"):
        load_index(invalid)


def test_build_index_from_csv_happy_path_and_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "index.csv"
    csv_path.write_text(
        "\n".join(
            [
                "source_id,source_type,period,t0,duration_hours,ra_deg,dec_deg,tag,blank",
                "TOI-1,toi,2.5,100.0,1.2,10.0,20.0,keep,",
            ]
        )
    )

    index = build_index_from_csv(csv_path)
    assert len(index) == 1
    entry = index.entries[0]
    assert entry.duration_hours == pytest.approx(1.2)
    assert entry.ra_deg == pytest.approx(10.0)
    assert entry.dec_deg == pytest.approx(20.0)
    assert entry.metadata == {"tag": "keep"}


def test_build_index_from_csv_rejects_missing_columns_and_empty_csv(tmp_path: Path) -> None:
    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("source_id,source_type,period\nTOI-1,toi,2.0\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        build_index_from_csv(missing_columns)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="Empty or invalid CSV file"):
        build_index_from_csv(empty)
