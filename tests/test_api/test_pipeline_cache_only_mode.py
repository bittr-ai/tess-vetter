from __future__ import annotations

from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate


def test_cache_only_mode_allows_offline_without_local_data(monkeypatch) -> None:
    """When no_download=True and cache_dir is set, pipeline should not hard-error
    with OfflineNoLocalDataError before attempting cache-only lookup.
    """

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
            pass

        def search_lightcurve_cached(self, tic_id: int, sector=None, author=None):  # noqa: ANN001
            return []

    # Patch the API facade import used by the pipeline.
    monkeypatch.setattr("tess_vetter.api.io.MASTClient", _FakeClient, raising=True)

    config = FeatureConfig(network_ok=False, no_download=True, cache_dir="/tmp/fake_cache")
    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=None,
        config=config,
    )
    # It should proceed into cache-only lookup and ultimately error due to no cached data.
    assert row["status"] == "ERROR"
    assert row["error_class"] in ("LightCurveNotFoundError", "NoSectorsSelectedError", "MASTClientError")
