from __future__ import annotations

import json
from dataclasses import dataclass
import importlib

import numpy as np

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.features.evidence import is_skip_block
from bittr_tess_vetter.pipeline import enrich_candidate
from bittr_tess_vetter.features import FEATURE_SCHEMA_VERSION, FeatureConfig


@dataclass
class _FakeBundle:
    results: list[object]
    warnings: list[str]
    inputs_summary: dict[str, object] = None  # type: ignore[assignment]
    provenance: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.inputs_summary is None:
            self.inputs_summary = {}
        if self.provenance is None:
            self.provenance = {"duration_ms": 0.0}


class _FakeCheck:
    def __init__(self, check_id: str, metrics: dict[str, object] | None = None) -> None:
        self._check_id = check_id
        self._metrics = metrics or {}

    def model_dump(self) -> dict[str, object]:
        return {"id": self._check_id, "metrics": dict(self._metrics)}


def _make_lightcurve(*, tic_id: int, sector: int) -> LightCurveData:
    # Ensure time spans the candidate t0 window so sector gating passes.
    time = np.arange(0.0, 30.0, 0.02, dtype=np.float64)
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.full_like(time, 2e-4, dtype=np.float64)
    quality = np.zeros_like(time, dtype=np.int32)
    valid_mask = np.ones_like(time, dtype=bool)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=tic_id,
        sector=sector,
        cadence_seconds=120.0,
    )


def test_enrich_candidate_offline_local_path_produces_stable_skip_blocks(monkeypatch) -> None:
    # Offline enrichment should:
    # - load light curves via local_data_path (patched)
    # - avoid network and TPF families
    # - return explicit skip blocks for those families
    # - always produce JSON-serializable row + raw
    tic_id = 42
    lc = _make_lightcurve(tic_id=tic_id, sector=1)

    def _fake_load_local(tic: int, _path: str, *, requested_sectors=None):
        assert tic == tic_id
        return [lc], [1]

    def _fake_vet_candidate(*args, **kwargs):
        # Keep checks minimal and deterministic; features should still be extractable.
        # Include V09 metrics to ensure the feature builder can parse optional V09 inputs.
        checks = [
            _FakeCheck("V01", {"delta_sigma": 0.0}),
            _FakeCheck("V02", {"secondary_depth_sigma": 0.0}),
            _FakeCheck("V09", {"distance_to_target_pixels": 0.3, "localization_reliable": True}),
        ]
        return _FakeBundle(results=checks, warnings=[])

    monkeypatch.setattr("bittr_tess_vetter.pipeline._load_lightcurves_from_local", _fake_load_local)
    vet_mod = importlib.import_module("bittr_tess_vetter.api.vet")
    monkeypatch.setattr(vet_mod, "vet_candidate", _fake_vet_candidate)

    cfg = FeatureConfig(
        network_ok=False,
        local_data_path="/dev/null",
        enable_candidate_evidence=False,
        enable_pixel_timeseries=False,
        enable_ghost_reliability=False,
        enable_host_plausibility=False,
        enable_sector_quality=False,
    )

    raw, row = enrich_candidate(
        tic_id=tic_id,
        toi="TOI-TEST",
        period_days=5.0,
        t0_btjd=1.0,
        duration_hours=2.0,
        depth_ppm=1000.0,
        config=cfg,
        sectors=[1],
    )

    assert row["status"] == "OK"
    assert row["tic_id"] == tic_id
    assert row["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert row["candidate_key"].startswith(f"{tic_id}|")

    assert is_skip_block(raw["candidate_evidence"])
    assert raw["candidate_evidence"]["reason"] == "disabled_by_config"

    # With no cache_dir/network, TPF families are explicitly skipped.
    assert is_skip_block(raw["localization"])
    assert raw["localization"]["reason"] == "tpf_unavailable"
    assert is_skip_block(raw["pixel_host_hypotheses"])
    assert raw["pixel_host_hypotheses"]["reason"] == "tpf_unavailable"
    assert is_skip_block(raw["sector_quality_report"])
    assert raw["sector_quality_report"]["reason"] == "tpf_unavailable"

    # Invariants: row/raw are JSON serializable.
    json.dumps(raw)
    json.dumps(row)


def test_enrich_candidate_local_data_missing_sets_error_status(monkeypatch) -> None:
    def _fake_load_local(_tic: int, _path: str, *, requested_sectors=None):
        raise FileNotFoundError("no local data")

    monkeypatch.setattr("bittr_tess_vetter.pipeline._load_lightcurves_from_local", _fake_load_local)

    cfg = FeatureConfig(network_ok=False, local_data_path="/dev/null")

    raw, row = enrich_candidate(
        tic_id=123,
        toi=None,
        period_days=5.0,
        t0_btjd=1.0,
        duration_hours=2.0,
        depth_ppm=1000.0,
        config=cfg,
        sectors=[1],
    )

    assert row["status"] == "ERROR"
    assert row["error_class"] in {"LocalDataNotFoundError", "LocalDataNotFoundError"}
    assert "local" in str(row["error"]).lower() or "no local" in str(row["error"]).lower()

    # Ensure raw still contains audit-friendly skip blocks
    assert "provenance" in raw
    json.dumps(raw)
    json.dumps(row)
