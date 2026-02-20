from __future__ import annotations

from tess_vetter.platform.catalogs.exofop_toi_table import (
    ExoFOPToiTable,
    query_exofop_toi_rows,
)


def test_query_exofop_toi_rows_skips_non_numeric_when_numeric_filter_applies() -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "period", "tfopwg_disposition"],
        rows=[
            {"toi": "100.01", "period": "5.0", "tfopwg_disposition": "PC"},
            {"toi": "100.02", "period": "bad", "tfopwg_disposition": "PC"},
            {"toi": "100.03", "period": "", "tfopwg_disposition": "PC"},
        ],
    )

    result = query_exofop_toi_rows(table, period_min=4.0)

    assert [row["toi"] for row in result.rows] == ["100.01"]
    assert result.stats.source_rows == 3
    assert result.stats.matched_rows_before_limit == 1
    assert result.stats.returned_rows == 1
    assert result.stats.skipped_non_numeric_rows == 2


def test_query_exofop_toi_rows_disposition_filters_sort_and_limit() -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "snr", "tfopwg_disposition"],
        rows=[
            {"toi": "200.01", "snr": "12.0", "tfopwg_disposition": "KP"},
            {"toi": "200.02", "snr": "8.0", "tfopwg_disposition": "FP"},
            {"toi": "200.03", "snr": "15.0", "tfopwg_disposition": "PC"},
            {"toi": "200.04", "snr": "9.0", "tfopwg_disposition": "PC"},
        ],
    )

    result = query_exofop_toi_rows(
        table,
        include_dispositions={"PC", "KP"},
        exclude_known_planets=True,
        exclude_false_positives=True,
        sort_by="snr",
        sort_descending=True,
        max_results=1,
    )

    assert [row["toi"] for row in result.rows] == ["200.03"]
    assert result.stats.source_rows == 4
    assert result.stats.matched_rows_before_limit == 2
    assert result.stats.returned_rows == 1
    assert result.stats.filtered_by_disposition_rows == 2


def test_query_exofop_toi_rows_exclude_known_planets_filters_cp() -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["toi", "tfopwg_disposition"],
        rows=[
            {"toi": "300.01", "tfopwg_disposition": "CP"},
            {"toi": "300.02", "tfopwg_disposition": "PC"},
        ],
    )

    result = query_exofop_toi_rows(
        table,
        include_dispositions={"CP", "PC"},
        exclude_known_planets=True,
    )

    assert [row["toi"] for row in result.rows] == ["300.02"]
