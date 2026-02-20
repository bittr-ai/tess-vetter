from __future__ import annotations

from tess_vetter.features import FeatureConfig
from tess_vetter.features.builder import build_features


def test_builder_exports_lc_diagnostics_and_check_scalars() -> None:
    raw = {
        "target": {"tic_id": 123},
        "ephemeris": {
            "period_days": 5.0,
            "t0_btjd": 1.0,
            "duration_hours": 2.0,
            "sectors": [1],
            "cadence_seconds": 120.0,
        },
        "depth_ppm": {"input_depth_ppm": 1000.0},
        "check_results": [
            {"id": "PF01", "status": "ok", "metrics": {"snr": 8.0, "snr_proxy": 8.0}},
            {"id": "V09", "status": "ok", "metrics": {"localization_reliable": True}},
            {"id": "V10", "status": "ok", "metrics": {"aperture_depth_sign_flip": False}},
            {
                "id": "V11",
                "status": "ok",
                "metrics": {"secondary_primary_ratio": 0.8, "fred": 12.3},
            },
            {
                "id": "V11b",
                "status": "ok",
                "metrics": {
                    "sig_pri": 10.0,
                    "sig_sec": 3.0,
                    "sig_ter": 4.2,
                    "chi": 7.7,
                    "fred": 1.2,
                },
            },
            {
                "id": "V13",
                "status": "ok",
                "metrics": {"missing_frac_max": 0.25, "n_epochs_missing_ge_0p25": 2},
            },
            {"id": "V15", "status": "ok", "metrics": {"asymmetry_sigma": 1.5}},
        ],
        "pixel_host_hypotheses": {"skipped": True, "reason": "no_tpf"},
        "localization": {"skipped": True, "reason": "no_tpf"},
        "sector_quality_report": {"skipped": True, "reason": "no_tpf"},
        "candidate_evidence": {"skipped": True, "reason": "no_network"},
        "ephemeris_specificity": {
            "smooth_score": 11.0,
            "null_pvalue": 0.02,
            "few_point_fraction": 0.3,
        },
        "alias_diagnostics": {"alias_class": "ALIAS_STRONG"},
        "systematics_proxy": {"score": 0.7},
        "lc_stats": {"lc_cadence_seconds": 120.0, "lc_n_valid": 12345},
        "provenance": {"pipeline_version": "test", "code_hash": "test"},
    }

    row = build_features(raw, FeatureConfig(network_ok=False, bulk_mode=True))

    assert row["smooth_score"] == 11.0
    assert row["null_pvalue"] == 0.02
    assert row["few_point_fraction"] == 0.3
    assert row["systematics_proxy_score"] == 0.7
    assert row["alias_class"] == "STRONG_ALIAS"

    assert row["v09_localization_reliable"] is True
    assert row["v10_aperture_depth_sign_flip"] is False
    assert row["v13_missing_frac_max"] == 0.25
    assert row["v13_n_epochs_missing_ge_0p25"] == 2
    assert row["v15_asymmetry_sigma"] == 1.5

    assert row["v11b_sig_ter"] == 4.2
    assert row["v11b_chi"] == 7.7

    assert row["lc_cadence_seconds"] == 120.0
    assert row["lc_n_valid"] == 12345

