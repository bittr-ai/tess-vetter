from __future__ import annotations

from dataclasses import dataclass

import pytest

from tess_vetter.api.pipeline import VettingPipeline
from tess_vetter.code_mode.errors import map_legacy_error_to_prd_code
from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate, enrich_worklist
from tess_vetter.validation.registry import CheckRegistry, CheckRequirements, CheckTier
from tess_vetter.validation.result_schema import ok_result


@pytest.mark.parametrize(
    ("legacy_error", "expected_prd"),
    [
        ("LocalDataNotFoundError", "SCHEMA_VIOLATION_OUTPUT"),
        ("NoDownloadError", "POLICY_DENIED"),
        ("OfflineNoLocalDataError", "POLICY_DENIED"),
        ("MissingOptionalDependencyError", "DEPENDENCY_MISSING"),
        ("KeyError", "SCHEMA_VIOLATION_INPUT"),
        ("NetworkTimeoutError", "TIMEOUT_EXCEEDED"),
    ],
)
def test_map_legacy_error_to_prd_code_known_cases(legacy_error: str, expected_prd: str) -> None:
    assert map_legacy_error_to_prd_code(legacy_error) == expected_prd


def test_map_legacy_error_to_prd_code_reason_flags_and_fallback() -> None:
    assert map_legacy_error_to_prd_code(None, reason_flag="EXTRA_MISSING:tls") == "DEPENDENCY_MISSING"
    assert map_legacy_error_to_prd_code(None, reason_flag="NETWORK_DISABLED") == "POLICY_DENIED"
    assert map_legacy_error_to_prd_code(None, reason_flag="INSUFFICIENT_DATA") == "SCHEMA_VIOLATION_INPUT"
    assert map_legacy_error_to_prd_code(None, reason_flag="NO_APERTURE_MASK") == "SCHEMA_VIOLATION_INPUT"
    assert (
        map_legacy_error_to_prd_code(None, reason_flag="NO_SECTOR_MEASUREMENTS")
        == "SCHEMA_VIOLATION_INPUT"
    )
    assert map_legacy_error_to_prd_code(None, reason_flag="INSUFFICIENT_SECTORS") == "SCHEMA_VIOLATION_INPUT"
    assert map_legacy_error_to_prd_code(None, reason_flag="NETWORK_TIMEOUT") == "TIMEOUT_EXCEEDED"
    assert map_legacy_error_to_prd_code(None, reason_flag="NETWORK_ERROR") == "POLICY_DENIED"
    assert map_legacy_error_to_prd_code(None, reason_flag=" network_timeout ") == "TIMEOUT_EXCEEDED"
    assert map_legacy_error_to_prd_code("UnknownLegacyError") == "SCHEMA_VIOLATION_OUTPUT"


def test_enrich_candidate_error_includes_prd_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_load_local(_tic_id: int, _path: str, *, requested_sectors=None):
        raise FileNotFoundError("no local data for TIC")

    monkeypatch.setattr("tess_vetter.pipeline._load_lightcurves_from_local", _fake_load_local)

    raw, row = enrich_candidate(
        tic_id=123,
        toi=None,
        period_days=5.0,
        t0_btjd=1.0,
        duration_hours=2.0,
        depth_ppm=1000.0,
        config=FeatureConfig(network_ok=False, local_data_path="/tmp/missing"),
        sectors=[1],
    )

    assert row["status"] == "ERROR"
    # Legacy fields remain unchanged for compatibility.
    assert row["error_class"] == "LocalDataNotFoundError"
    assert raw["provenance"]["error_class"] == "LocalDataNotFoundError"
    # Additive PRD metadata for code-mode consumers.
    assert raw["provenance"]["error_code"] == "SCHEMA_VIOLATION_OUTPUT"


@dataclass
class _FakeCheck:
    _id: str
    _requirements: CheckRequirements
    _runner: object

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._id

    @property
    def tier(self) -> CheckTier:
        return CheckTier.AUX

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return []

    def run(self, inputs, config):
        return self._runner(inputs, config)


def test_api_pipeline_skip_dependency_includes_prd_error_code() -> None:
    registry = CheckRegistry()
    registry.register(
        _FakeCheck(
            _id="V99",
            _requirements=CheckRequirements(optional_deps=("tls",)),
            _runner=lambda _inputs, _config: ok_result("V99", "V99", metrics={}),
        )
    )

    pipeline = VettingPipeline(registry=registry)
    bundle = pipeline.run(lc=object(), candidate=object())

    result = bundle.results[0]
    assert result.status == "skipped"
    assert result.flags == ["SKIPPED:EXTRA_MISSING:tls"]
    assert result.provenance["error_code"] == "DEPENDENCY_MISSING"


def test_api_pipeline_error_includes_prd_error_code_and_legacy_flag() -> None:
    registry = CheckRegistry()
    registry.register(
        _FakeCheck(
            _id="V98",
            _requirements=CheckRequirements(),
            _runner=lambda _inputs, _config: (_ for _ in ()).throw(KeyError("bad-input")),
        )
    )

    pipeline = VettingPipeline(registry=registry)
    bundle = pipeline.run(lc=object(), candidate=object())

    result = bundle.results[0]
    assert result.status == "error"
    assert result.flags[0] == "ERROR:KeyError"
    assert result.provenance["error_code"] == "SCHEMA_VIOLATION_INPUT"


def test_enrich_worklist_fallback_row_includes_provenance_error_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured_rows: list[dict] = []

    def _fake_append_jsonl(_out_path, row):
        captured_rows.append(dict(row))

    def _raise_enrichment_error(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tess_vetter.api.jsonl.append_jsonl", _fake_append_jsonl)
    monkeypatch.setattr("tess_vetter.pipeline.enrich_candidate", _raise_enrichment_error)

    summary = enrich_worklist(
        worklist_iter=iter(
            [
                {
                    "tic_id": 123,
                    "period_days": 2.5,
                    "t0_btjd": 1.0,
                    "duration_hours": 3.0,
                    "depth_ppm": 1000.0,
                }
            ]
        ),
        output_path=tmp_path / "out.jsonl",
        config=FeatureConfig(),
        progress_interval=0,
    )

    assert summary.processed == 0
    assert summary.errors == 1
    assert len(captured_rows) == 1
    assert captured_rows[0]["status"] == "ERROR"
    assert captured_rows[0]["error_class"] == "RuntimeError"
    assert captured_rows[0]["provenance"]["error_code"] == "SCHEMA_VIOLATION_OUTPUT"
