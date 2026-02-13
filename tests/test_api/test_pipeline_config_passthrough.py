from __future__ import annotations

import importlib

import numpy as np

from bittr_tess_vetter.api.pipeline import PipelineConfig
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.validation.result_schema import VettingBundleResult

generate_report_api = importlib.import_module("bittr_tess_vetter.api.generate_report")
vet_api = importlib.import_module("bittr_tess_vetter.api.vet")


class _FakePipeline:
    def __init__(self, *args, checks=None, registry=None, config=None, **kwargs):
        self.config = config

    def run(self, *args, **kwargs):
        return VettingBundleResult(
            results=[],
            warnings=[],
            provenance={},
            inputs_summary={},
        )

    def run_many(self, *args, **kwargs):
        return (
            [
                VettingBundleResult(
                    results=[],
                    warnings=[],
                    provenance={},
                    inputs_summary={},
                )
            ],
            [],
        )


def _candidate() -> Candidate:
    return Candidate(
        ephemeris=Ephemeris(period_days=10.0, t0_btjd=100.0, duration_hours=2.0),
        depth_ppm=1000.0,
    )


def _light_curve() -> LightCurve:
    time = np.linspace(0.0, 1.0, 32)
    return LightCurve(time=time, flux=np.ones_like(time), flux_err=np.ones_like(time) * 1e-4)


def test_vet_candidate_accepts_pipeline_config_and_stamps_provenance(monkeypatch) -> None:
    monkeypatch.setattr("bittr_tess_vetter.api.pipeline.VettingPipeline", _FakePipeline)

    cfg = PipelineConfig(timeout_seconds=7.5, random_seed=11, extra_params={"k": "v"})
    bundle = vet_api.vet_candidate(
        _light_curve(),
        _candidate(),
        checks=["V01"],
        pipeline_config=cfg,
    )

    assert bundle.provenance["pipeline_config"]["timeout_seconds"] == 7.5
    assert bundle.provenance["pipeline_config"]["random_seed"] == 11
    assert bundle.provenance["pipeline_config"]["extra_params"]["k"] == "v"


def test_vet_many_accepts_pipeline_config_and_stamps_provenance(monkeypatch) -> None:
    monkeypatch.setattr("bittr_tess_vetter.api.pipeline.VettingPipeline", _FakePipeline)

    cfg = PipelineConfig(timeout_seconds=3.0, fail_fast=True, extra_params={"x": 1})
    bundles, _ = vet_api.vet_many(
        _light_curve(),
        [_candidate()],
        checks=["V01"],
        pipeline_config=cfg,
    )

    assert bundles[0].provenance["pipeline_config"]["timeout_seconds"] == 3.0
    assert bundles[0].provenance["pipeline_config"]["fail_fast"] is True
    assert bundles[0].provenance["pipeline_config"]["extra_params"]["x"] == 1


def test_resolve_effective_pipeline_config_defaults_request_timeout() -> None:
    resolved = generate_report_api._resolve_effective_pipeline_config(
        pipeline_config=None,
        check_timeout_seconds=12.0,
    )

    assert resolved.timeout_seconds == 12.0
    assert resolved.extra_params["request_timeout_seconds"] == 12.0


def test_resolve_effective_pipeline_config_respects_explicit_values() -> None:
    cfg = PipelineConfig(
        timeout_seconds=5.0,
        extra_params={"request_timeout_seconds": 1.0, "k": "v"},
    )
    resolved = generate_report_api._resolve_effective_pipeline_config(
        pipeline_config=cfg,
        check_timeout_seconds=12.0,
    )

    assert resolved.timeout_seconds == 5.0
    assert resolved.extra_params["request_timeout_seconds"] == 1.0
    assert resolved.extra_params["k"] == "v"
