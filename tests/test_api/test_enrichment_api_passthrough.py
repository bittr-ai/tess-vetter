from __future__ import annotations

from bittr_tess_vetter.api import enrichment as enrichment_api
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import (
    EnrichmentSummary as PipelineEnrichmentSummary,
)
from bittr_tess_vetter.pipeline import (
    enrich_candidate as pipeline_enrich_candidate,
)
from bittr_tess_vetter.pipeline import (
    enrich_worklist as pipeline_enrich_worklist,
)
from bittr_tess_vetter.pipeline import (
    make_candidate_key as pipeline_make_candidate_key,
)


def test_enrichment_exports_pipeline_symbols() -> None:
    assert enrichment_api.enrich_candidate is pipeline_enrich_candidate
    assert enrichment_api.enrich_worklist is pipeline_enrich_worklist
    assert enrichment_api.make_candidate_key is pipeline_make_candidate_key
    assert enrichment_api.EnrichmentSummary is PipelineEnrichmentSummary


def test_normalize_feature_config_passthrough_instance() -> None:
    cfg = FeatureConfig(bulk_mode=True, network_ok=False, no_download=True)
    out = enrichment_api.normalize_feature_config(cfg)
    assert out is cfg


def test_normalize_feature_config_constructs_from_mapping() -> None:
    out = enrichment_api.normalize_feature_config(
        {
            "bulk_mode": True,
            "network_ok": False,
            "allow_20s": True,
            "no_download": True,
            "require_tpf": False,
            "enable_t0_refine": False,
            "cache_dir": None,
            "local_data_path": None,
        }
    )
    assert isinstance(out, FeatureConfig)
    assert out.bulk_mode is True
    assert out.network_ok is False
    assert out.allow_20s is True
    assert out.no_download is True

