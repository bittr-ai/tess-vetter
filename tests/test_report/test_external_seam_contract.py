from __future__ import annotations

import copy

from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.report import ReportPayloadModel, build_report
from bittr_tess_vetter.report._custom_view_paths import is_allowed_custom_view_path
from bittr_tess_vetter.report._data import _canonical_sha256


def _make_minimal_lc() -> LightCurve:
    return LightCurve(
        time=[0.0, 0.1, 0.2, 0.3],
        flux=[1.0, 0.999, 1.001, 1.0],
        flux_err=[0.0001, 0.0001, 0.0001, 0.0001],
    )


def _make_candidate() -> Candidate:
    return Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )


def test_external_seam_top_level_contract_keys_are_present() -> None:
    payload = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False).to_json()

    assert set(payload.keys()) == {"schema_version", "summary", "plot_data", "custom_views", "payload_meta"}
    ReportPayloadModel.model_validate(payload)


def test_external_seam_payload_meta_custom_view_fields_are_required() -> None:
    payload = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False).to_json()
    meta = payload["payload_meta"]

    assert "custom_views_version" in meta
    assert "custom_views_hash" in meta
    assert "custom_view_hashes_by_id" in meta
    assert "custom_views_includes_ad_hoc" in meta


def test_external_seam_json_pointer_policy_for_custom_view_paths() -> None:
    assert is_allowed_custom_view_path("/plot_data/phase_folded/bin_flux")
    assert is_allowed_custom_view_path("/summary/lc_summary/snr")

    assert not is_allowed_custom_view_path("plot_data.phase_folded.bin_flux")
    assert not is_allowed_custom_view_path("/payload_meta/summary_hash")


def test_external_seam_summary_plot_hashes_ignore_payload_meta_edits() -> None:
    payload = build_report(_make_minimal_lc(), _make_candidate(), include_additional_plots=False).to_json()

    summary_hash_before = _canonical_sha256(payload["summary"])
    plot_hash_before = _canonical_sha256(payload["plot_data"])
    stored_summary_hash_before = payload["payload_meta"]["summary_hash"]
    stored_plot_hash_before = payload["payload_meta"]["plot_data_hash"]

    mutated = copy.deepcopy(payload)
    mutated["payload_meta"]["summary_version"] = "99"
    mutated["payload_meta"]["plot_data_version"] = "99"
    mutated["payload_meta"]["custom_views_version"] = "99"
    mutated["payload_meta"]["contract_version"] = "99"

    assert _canonical_sha256(mutated["summary"]) == summary_hash_before
    assert _canonical_sha256(mutated["plot_data"]) == plot_hash_before
    assert mutated["payload_meta"]["summary_hash"] == stored_summary_hash_before
    assert mutated["payload_meta"]["plot_data_hash"] == stored_plot_hash_before
