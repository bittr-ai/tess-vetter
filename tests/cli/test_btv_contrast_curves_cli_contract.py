from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from bittr_tess_vetter.cli.contrast_curves_cli import (
    contrast_curve_summary_command,
    contrast_curves_command,
)
from bittr_tess_vetter.exofop.types import ExoFopFileRow


def test_btv_contrast_curve_summary_success_contract(monkeypatch, tmp_path: Path) -> None:
    cc_file = tmp_path / "curve.tbl"
    cc_file.write_text("0.10 2.0\n0.50 4.5\n1.00 6.1\n", encoding="utf-8")

    out_path = tmp_path / "summary.json"
    runner = CliRunner()
    result = runner.invoke(
        contrast_curve_summary_command,
        [
            "TOI-123.01",
            "--file",
            str(cc_file),
            "--filter",
            "Kcont",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.contrast_curve_summary.v1"
    assert payload["inputs_summary"]["toi"] == "TOI-123.01"
    assert payload["sensitivity"]["n_points"] == 3
    assert payload["ruling_summary"]["status"] == "ok"
    assert payload["verdict"] in {
        "CONTRAST_CURVES_STRONG",
        "CONTRAST_CURVES_MODERATE",
        "CONTRAST_CURVES_LIMITED",
    }
    assert payload["verdict_source"] == "$.ruling_summary.quality_assessment"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["result"]["sensitivity"] == payload["sensitivity"]


def test_btv_contrast_curve_summary_rejects_mismatched_positional_and_option_toi(tmp_path: Path) -> None:
    cc_file = tmp_path / "curve.tbl"
    cc_file.write_text("0.1 2.0\n0.5 4.0\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        contrast_curve_summary_command,
        [
            "TOI-5807.01",
            "--toi",
            "TOI-4510.01",
            "--file",
            str(cc_file),
        ],
    )

    assert result.exit_code == 1
    assert "must match" in result.output


def test_btv_contrast_curves_success_contract(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    target_root = cache_dir / "exofop" / "tic_123" / "files"
    target_root.mkdir(parents=True, exist_ok=True)

    file_one = target_root / "TOI5807I-dc20240727-Kcont_plot.tbl"
    file_two = target_root / "TOI123_contrast.csv"
    file_one.write_text("0.1 2.5\n0.5 4.0\n1.0 5.2\n", encoding="utf-8")
    file_two.write_text("0.2,2.0\n0.7,4.8\n1.5,5.5\n", encoding="utf-8")

    class _FakeExoFopClient:
        def __init__(self, *, cache_dir: Path, cookie_jar_path: Path | None = None) -> None:
            self.cache_dir = Path(cache_dir)
            self.cookie_jar_path = cookie_jar_path

        def resolve_target(self, target: str):
            return SimpleNamespace(tic_id=123, toi=target)

        def file_list(self, *, tic_id: int, force_refresh: bool = False):
            _ = tic_id, force_refresh
            return [
                ExoFopFileRow(
                    file_id=1,
                    tic_id=123,
                    toi="123.01",
                    filename="TOI5807I-dc20240727-Kcont_plot.tbl",
                    type="Image",
                    description="Palomar-5m AO sensitivity table",
                ),
                ExoFopFileRow(
                    file_id=2,
                    tic_id=123,
                    toi="123.01",
                    filename="TOI123_contrast.csv",
                    type="Image",
                    description="Contrast limits",
                ),
            ]

        def fetch_files(self, *, target: int, selectors, force_refresh: bool = False):
            _ = target, force_refresh
            assert selectors is not None
            assert selectors.max_files == 12
            return SimpleNamespace(
                cache_root=self.cache_dir / "exofop" / "tic_123",
                manifest_path=self.cache_dir / "exofop" / "tic_123" / "manifest.json",
                files_downloaded=[file_one, file_two],
                files_skipped=[],
                warnings=[],
            )

    monkeypatch.setattr("bittr_tess_vetter.cli.contrast_curves_cli.ExoFopClient", _FakeExoFopClient)

    out_path = tmp_path / "contrast_curves.json"
    runner = CliRunner()
    result = runner.invoke(
        contrast_curves_command,
        [
            "TOI-123.01",
            "--network-ok",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.contrast_curves.v2"
    assert payload["tic_id"] == 123
    assert payload["toi"] == "TOI-123.01"
    assert payload["inputs_summary"]["tic_id"] == 123
    assert payload["inputs_summary"]["input_resolution"]["source"] == "toi_catalog"
    assert len(payload["observations"]) == 2
    assert payload["combined_exclusion"]["n_observations"] == 2
    assert payload["ruling_summary"]["status"] == "ok"
    assert payload["summary"]["availability"] == "available"
    assert payload["summary"]["n_observations"] == 2
    assert payload["summary"]["selected_curve"]["source"] == "exofop"
    assert payload["fpp_recommendations"]["primary"]["observation_index"] in {0, 1}
    assert isinstance(payload["fpp_recommendations"]["all_candidates"], list)
    assert payload["result"]["observations"] == payload["observations"]
    assert payload["result"]["combined_exclusion"] == payload["combined_exclusion"]
    assert payload["result"]["ruling_summary"] == payload["ruling_summary"]
    assert payload["result"]["fpp_recommendations"] == payload["fpp_recommendations"]
    assert payload["result"]["summary"] == payload["summary"]
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]


def test_btv_contrast_curves_no_data_behavior(monkeypatch, tmp_path: Path) -> None:
    class _FakeExoFopClient:
        def __init__(self, *, cache_dir: Path, cookie_jar_path: Path | None = None) -> None:
            self.cache_dir = Path(cache_dir)
            self.cookie_jar_path = cookie_jar_path

        def resolve_target(self, target: str):
            return SimpleNamespace(tic_id=777, toi=target)

        def file_list(self, *, tic_id: int, force_refresh: bool = False):
            _ = tic_id, force_refresh
            return []

        def fetch_files(self, *, target: int, selectors, force_refresh: bool = False):
            _ = target, selectors, force_refresh
            return SimpleNamespace(
                cache_root=self.cache_dir / "exofop" / "tic_777",
                manifest_path=self.cache_dir / "exofop" / "tic_777" / "manifest.json",
                files_downloaded=[],
                files_skipped=[],
                warnings=[],
            )

    monkeypatch.setattr("bittr_tess_vetter.cli.contrast_curves_cli.ExoFopClient", _FakeExoFopClient)

    out_path = tmp_path / "contrast_curves_no_data.json"
    runner = CliRunner()
    result = runner.invoke(
        contrast_curves_command,
        [
            "TOI-777.01",
            "--network-ok",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["observations"] == []
    assert payload["combined_exclusion"] is None
    assert payload["ruling_summary"]["status"] == "no_data"
    assert payload["verdict"] == "NO_CONTRAST_CURVES"
    assert payload["verdict_source"] == "$.ruling_summary.status"
    assert payload["summary"]["availability"] == "none"
    assert payload["fpp_recommendations"]["primary"] is None


def test_btv_contrast_curves_skips_non_table_image_artifacts(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    target_root = cache_dir / "exofop" / "tic_123" / "files"
    target_root.mkdir(parents=True, exist_ok=True)

    good_tbl = target_root / "TOI5807I-dc20240727-Kcont_plot.tbl"
    bad_ps = target_root / "TOI5807I-dc20240727-Kcont_plot.ps"
    good_tbl.write_text("0.000,0.000,0.0\n0.111,2.088,0.1\n0.514,6.991,0.2\n1.025,8.461,0.2\n", encoding="utf-8")
    bad_ps.write_text("%!PS-Adobe-3.0\n100 200 moveto\n300 400 lineto\n", encoding="utf-8")

    class _FakeExoFopClient:
        def __init__(self, *, cache_dir: Path, cookie_jar_path: Path | None = None) -> None:
            self.cache_dir = Path(cache_dir)
            self.cookie_jar_path = cookie_jar_path

        def resolve_target(self, target: str):
            return SimpleNamespace(tic_id=123, toi=target)

        def file_list(self, *, tic_id: int, force_refresh: bool = False):
            _ = tic_id, force_refresh
            return [
                ExoFopFileRow(
                    file_id=1,
                    tic_id=123,
                    toi="123.01",
                    filename="TOI5807I-dc20240727-Kcont_plot.tbl",
                    type="Image",
                    description="Palomar-5m AO sensitivity table",
                ),
                ExoFopFileRow(
                    file_id=2,
                    tic_id=123,
                    toi="123.01",
                    filename="TOI5807I-dc20240727-Kcont_plot.ps",
                    type="Image",
                    description="Palomar-5m AO Sensitivity Curve",
                ),
            ]

        def fetch_files(self, *, target: int, selectors, force_refresh: bool = False):
            _ = target, selectors, force_refresh
            return SimpleNamespace(
                cache_root=self.cache_dir / "exofop" / "tic_123",
                manifest_path=self.cache_dir / "exofop" / "tic_123" / "manifest.json",
                files_downloaded=[good_tbl, bad_ps],
                files_skipped=[],
                warnings=[],
            )

    monkeypatch.setattr("bittr_tess_vetter.cli.contrast_curves_cli.ExoFopClient", _FakeExoFopClient)

    out_path = tmp_path / "contrast_curves_skip_ps.json"
    runner = CliRunner()
    result = runner.invoke(
        contrast_curves_command,
        [
            "TOI-123.01",
            "--network-ok",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["observations"]) == 1
    assert payload["observations"][0]["filename"].endswith(".tbl")
    assert payload["ruling_summary"]["max_delta_mag_at_0p5_arcsec"] == pytest.approx(6.820672, rel=0, abs=1e-6)
    parse_failures = payload["provenance"]["parse_failures"]
    assert any(item["file"].endswith(".ps") for item in parse_failures)
