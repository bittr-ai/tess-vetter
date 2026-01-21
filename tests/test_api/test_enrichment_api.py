from __future__ import annotations

from bittr_tess_vetter.api.enrichment import candidate_key_from_row, validate_enriched_row


def test_candidate_key_from_row() -> None:
    row = {"tic_id": 123, "period_days": 10.0, "t0_btjd": 100.0}
    assert candidate_key_from_row(row) == "123|10.0|100.0"


def test_validate_enriched_row_happy_path() -> None:
    row = {
        "tic_id": 123,
        "toi": None,
        "period_days": 10.0,
        "t0_btjd": 100.0,
        "duration_hours": 2.0,
        "depth_ppm": 500.0,
        "status": "OK",
        "error_class": None,
        "error": None,
        "candidate_key": "123|10.0|100.0",
        "pipeline_version": "x",
        "feature_schema_version": "6.0.0",
        "feature_config": {"bulk_mode": True},
        "inputs_summary": {},
        "missing_feature_families": [],
        "item_wall_ms": 1.0,
    }
    validate_enriched_row(row)


def test_validate_enriched_row_detects_bad_candidate_key() -> None:
    row = {
        "tic_id": 123,
        "period_days": 10.0,
        "t0_btjd": 100.0,
        "duration_hours": 2.0,
        "candidate_key": "wrong",
        "status": "OK",
        "pipeline_version": "x",
        "feature_schema_version": "6.0.0",
        "feature_config": {},
        "inputs_summary": {},
        "missing_feature_families": [],
        "item_wall_ms": 1.0,
    }
    try:
        validate_enriched_row(row)
    except ValueError as e:
        assert "candidate_key" in str(e)
    else:
        raise AssertionError("Expected ValueError")

