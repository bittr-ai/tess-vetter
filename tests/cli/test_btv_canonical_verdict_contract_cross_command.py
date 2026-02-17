from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("command_name", "payload", "mirror_keys"),
    [
        pytest.param(
            "activity",
            {
                "schema_version": "cli.activity.v1",
                "activity": {"rotation_period": 6.25},
                "result": {
                    "activity": {"rotation_period": 6.25},
                    "verdict": "QUIET",
                    "verdict_source": "$.activity.variability_class",
                },
                "verdict": "QUIET",
                "verdict_source": "$.activity.variability_class",
            },
            [("activity", "activity")],
            id="activity",
        ),
        pytest.param(
            "systematics-proxy",
            {
                "schema_version": "cli.systematics_proxy.v1",
                "systematics_proxy": {"score": 0.35},
                "result": {
                    "systematics_proxy": {"score": 0.35},
                    "verdict": "CLEAN",
                    "verdict_source": "$.systematics_proxy.score",
                },
                "verdict": "CLEAN",
                "verdict_source": "$.systematics_proxy.score",
            },
            [("systematics_proxy", "systematics_proxy")],
            id="systematics-proxy",
        ),
        pytest.param(
            "model-compete",
            {
                "schema_version": "cli.model_compete.v1",
                "result": {
                    "model_competition": {"model_competition_label": "TRANSIT"},
                    "verdict": "TRANSIT",
                    "verdict_source": "$.result.interpretation_label",
                },
                "verdict": "TRANSIT",
                "verdict_source": "$.result.interpretation_label",
            },
            [],
            id="model-compete",
        ),
        pytest.param(
            "timing",
            {
                "schema_version": "cli.timing.v1",
                "next_actions": [{"code": "TIMING_MEASURABLE"}],
                "result": {
                    "next_actions": [{"code": "TIMING_MEASURABLE"}],
                    "verdict": "TIMING_MEASURABLE",
                    "verdict_source": "$.result.next_actions[0].code",
                },
                "verdict": "TIMING_MEASURABLE",
                "verdict_source": "$.result.next_actions[0].code",
            },
            [("next_actions", "next_actions")],
            id="timing",
        ),
        pytest.param(
            "ephemeris-reliability",
            {
                "schema_version": "cli.ephemeris_reliability.v1",
                "result": {
                    "label": "ok",
                    "verdict": "ok",
                    "verdict_source": "$.result.label",
                },
                "verdict": "ok",
                "verdict_source": "$.result.label",
            },
            [],
            id="ephemeris-reliability",
        ),
        pytest.param(
            "localize-host",
            {
                "schema_version": "cli.localize_host.v1",
                "result": {
                    "consensus_label": "ON_TARGET",
                    "verdict": "ON_TARGET",
                    "verdict_source": "$.result.consensus_label",
                },
                "verdict": "ON_TARGET",
                "verdict_source": "$.result.consensus_label",
            },
            [],
            id="localize-host",
        ),
        pytest.param(
            "dilution",
            {
                "schema_version": "cli.dilution.v1",
                "scenarios": [{"id": "host"}],
                "result": {
                    "scenarios": [{"id": "host"}],
                    "verdict": "AMBIGUOUS_HOST",
                    "verdict_source": "$.provenance.host_ambiguous",
                },
                "verdict": "AMBIGUOUS_HOST",
                "verdict_source": "$.provenance.host_ambiguous",
            },
            [("scenarios", "scenarios")],
            id="dilution",
        ),
        pytest.param(
            "rv-feasibility",
            {
                "schema_version": "cli.rv_feasibility.v1",
                "activity": {"rotation_period": 6.1},
                "rv_feasibility": {"verdict": "MODERATE_RV_FEASIBILITY"},
                "result": {
                    "activity": {"rotation_period": 6.1},
                    "rv_feasibility": {"verdict": "MODERATE_RV_FEASIBILITY"},
                    "verdict": "MODERATE_RV_FEASIBILITY",
                    "verdict_source": "$.result.rv_feasibility.verdict",
                },
                "verdict": "MODERATE_RV_FEASIBILITY",
                "verdict_source": "$.result.rv_feasibility.verdict",
            },
            [("activity", "activity"), ("rv_feasibility", "rv_feasibility")],
            id="rv-feasibility",
        ),
        pytest.param(
            "measure-sectors",
            {
                "schema_version": "cli.measure_sectors.v1",
                "sector_measurements": [{"sector": 1, "depth_ppm": 500.0, "depth_err_ppm": 50.0}],
                "consistency": {"verdict": "CONSISTENT"},
                "recommended_sectors": [1],
                "result": {
                    "sector_measurements": [{"sector": 1, "depth_ppm": 500.0, "depth_err_ppm": 50.0}],
                    "consistency": {"verdict": "CONSISTENT"},
                    "recommended_sectors": [1],
                    "verdict": "CONSISTENT",
                    "verdict_source": "$.consistency.verdict",
                },
                "verdict": "CONSISTENT",
                "verdict_source": "$.consistency.verdict",
            },
            [
                ("sector_measurements", "sector_measurements"),
                ("consistency", "consistency"),
                ("recommended_sectors", "recommended_sectors"),
            ],
            id="measure-sectors",
        ),
        pytest.param(
            "detrend-grid",
            {
                "schema_version": "cli.detrend_grid.v1",
                "stable": True,
                "best_variant": {"variant_id": "v1"},
                "result": {
                    "stable": True,
                    "best_variant": {"variant_id": "v1"},
                    "verdict": "STABLE",
                    "verdict_source": "$.stable",
                },
                "verdict": "STABLE",
                "verdict_source": "$.stable",
            },
            [("stable", "stable"), ("best_variant", "best_variant")],
            id="detrend-grid",
        ),
    ],
)
def test_fixture_payloads_include_canonical_verdict_contract(
    command_name: str,
    payload: dict[str, object],
    mirror_keys: list[tuple[str, str]],
) -> None:
    _ = command_name
    assert isinstance(payload.get("schema_version"), str)
    assert isinstance(payload.get("result"), dict)
    result_payload = payload["result"]
    assert "verdict" in payload
    assert "verdict_source" in payload
    assert "verdict" in result_payload
    assert "verdict_source" in result_payload
    assert result_payload["verdict"] == payload["verdict"]
    assert result_payload["verdict_source"] == payload["verdict_source"]

    for top_level_key, result_key in mirror_keys:
        assert top_level_key in payload
        assert result_key in result_payload
        assert payload[top_level_key] == result_payload[result_key]
