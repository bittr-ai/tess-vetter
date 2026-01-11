from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from bittr_tess_vetter.validation.checks_catalog import run_exofop_toi_lookup, run_nearby_eb_search
from bittr_tess_vetter.catalogs.exofop_toi_table import ExoFOPToiTable


@dataclass
class FakeResponse:
    text: str
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_run_nearby_eb_search_parses_votable_and_returns_separation() -> None:
    votable = """<?xml version="1.0"?>
<VOTABLE>
  <RESOURCE>
    <TABLE>
      <FIELD name="TIC" />
      <FIELD name="Per" />
      <FIELD name="RAJ2000" />
      <FIELD name="DEJ2000" />
      <DATA>
        <TABLEDATA>
          <TR><TD>123</TD><TD>2.0</TD><TD>10.0</TD><TD>-20.0</TD></TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""

    def http_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse(votable)

    result = run_nearby_eb_search(
        ra_deg=10.0,
        dec_deg=-20.0,
        candidate_period_days=2.0,
        http_get=http_get,
    )

    assert result.id == "V06"
    assert result.passed is None
    assert result.details["status"] == "ok"
    assert result.details["n_ebs_found"] == 1
    match = result.details["matches"][0]
    assert match["tic_id"] == 123
    assert match["delta_1x"] == 0.0
    assert abs(match["sep_arcsec"]) < 1e-9


def test_run_nearby_eb_search_zero_matches_returns_ok() -> None:
    votable = """<?xml version="1.0"?>
<VOTABLE>
  <RESOURCE>
    <TABLE>
      <FIELD name="TIC" />
      <FIELD name="Per" />
      <FIELD name="RAJ2000" />
      <FIELD name="DEJ2000" />
      <DATA><TABLEDATA></TABLEDATA></DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""

    def http_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse(votable)

    result = run_nearby_eb_search(ra_deg=10.0, dec_deg=-20.0, http_get=http_get)
    assert result.details["status"] == "ok"
    assert result.details["n_ebs_found"] == 0


def test_run_exofop_toi_lookup_can_filter_by_toi() -> None:
    table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["tic_id", "toi", "tfopwg_disposition"],
        rows=[
            {"tic_id": "123", "toi": "100.01", "tfopwg_disposition": "PC"},
            {"tic_id": "123", "toi": "101.01", "tfopwg_disposition": "FP"},
        ],
    )
    with patch("bittr_tess_vetter.catalogs.exofop_toi_table.fetch_exofop_toi_table", return_value=table):
        result = run_exofop_toi_lookup(tic_id=123, toi=101.01, http_get=None)
    assert result.id == "V07"
    assert result.details["status"] == "ok"
    assert result.details["found"] is True
    assert result.details["row"]["tfopwg_disposition"] == "FP"


def test_run_exofop_toi_lookup_uses_stale_cache_on_fetch_failure() -> None:
    stale_table = ExoFOPToiTable(
        fetched_at_unix=0.0,
        headers=["tic_id", "toi", "tfopwg_disposition"],
        rows=[{"tic_id": "123", "toi": "101.01", "tfopwg_disposition": "FP"}],
    )

    with patch(
        "bittr_tess_vetter.catalogs.exofop_toi_table.fetch_exofop_toi_table",
        side_effect=[RuntimeError("network down"), stale_table],
    ):
        result = run_exofop_toi_lookup(tic_id=123, toi=101.01, http_get=None)

    assert result.details["status"] == "ok"
    assert result.details["found"] is True
    assert result.details["used_stale_cache"] is True
    assert result.details["fetch_error"] == "network down"
