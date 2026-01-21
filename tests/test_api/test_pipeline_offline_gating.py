from __future__ import annotations

import pytest

import bittr_tess_vetter.api.io as btv_io
import bittr_tess_vetter.pipeline as pipeline
from bittr_tess_vetter.features import FeatureConfig


def test_enrich_candidate_offline_requires_local_data_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("MASTClient should not be constructed when network_ok=False")

    monkeypatch.setattr(btv_io, "MASTClient", _boom)  # type: ignore[assignment]

    raw, row = pipeline.enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=False, bulk_mode=True),
    )

    assert row["status"] == "ERROR"
    assert row["error_class"] == "OfflineNoLocalDataError"
    assert "local_data_path" in (row["error"] or "")


def test_enrich_candidate_no_download_requires_local_data_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("MASTClient should not be constructed when no_download=True")

    monkeypatch.setattr(btv_io, "MASTClient", _boom)  # type: ignore[assignment]

    _raw, row = pipeline.enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=True, no_download=True, bulk_mode=True),
    )

    assert row["status"] == "ERROR"
    assert row["error_class"] == "NoDownloadError"
    assert "local_data_path" in (row["error"] or "")
