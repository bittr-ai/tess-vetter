from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bittr_tess_vetter.followup.service import run_followup


def test_run_followup_no_network_does_not_fetch_obsnotes(monkeypatch, tmp_path: Path) -> None:
    class _FakeClient:
        def __init__(self, *, cache_dir: Path):
            self.cache_dir = cache_dir

        def fetch_files(self, *, target: int, selectors, force_refresh: bool = False):  # noqa: ARG002
            raise AssertionError("fetch_files should not be called in --no-network mode")

        def obs_notes(self, *, tic_id: int, force_refresh: bool = False):  # noqa: ARG002
            raise AssertionError("obs_notes should not be called in --no-network mode")

    monkeypatch.setattr("bittr_tess_vetter.followup.service.ExoFopClient", _FakeClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.followup.service.detect_render_capabilities",
        lambda: SimpleNamespace(can_render_pdf=False, preferred_renderer=None),
    )

    request = SimpleNamespace(
        tic_id=123,
        toi="TOI-123.01",
        network_ok=False,
        cache_dir=tmp_path,
        render_images=False,
        include_raw_spectra=False,
        max_files=None,
        skip_notes=False,
        notes_file=None,
    )
    payload = run_followup(request)

    assert payload["summary"]["n_vetting_notes"] == 0
    assert "Network disabled; skipping ExoFOP observation notes fetch." in payload["provenance_extra"]["warnings"]


def test_run_followup_reads_cached_manifest_and_headers_only(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "exofop" / "tic_321"
    files_dir = cache_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    fits_file = files_dir / "spec_1.fits"
    fits_file.write_bytes(b"fits-bytes")
    (cache_root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file_id": 1,
                        "filename": "spec_1.fits",
                        "type": "Spectrum",
                        "path": "files/spec_1.fits",
                    },
                    {
                        "file_id": 2,
                        "filename": "img_1.png",
                        "type": "Image",
                        "path": "files/img_1.png",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class _FakeClient:
        def __init__(self, *, cache_dir: Path):
            self.cache_dir = cache_dir

        def fetch_files(self, *, target: int, selectors, force_refresh: bool = False):  # noqa: ARG002
            raise AssertionError("fetch_files should not be called in --no-network mode")

    monkeypatch.setattr("bittr_tess_vetter.followup.service.ExoFopClient", _FakeClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.followup.service.detect_render_capabilities",
        lambda: SimpleNamespace(can_render_pdf=True, preferred_renderer="pdftoppm"),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.followup.service.extract_fits_header",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=SimpleNamespace(status="ok", reason=None),
            header={"INSTRUME": "TESTSPEC"},
        ),
    )

    request = SimpleNamespace(
        tic_id=321,
        toi="TOI-321.01",
        network_ok=False,
        cache_dir=tmp_path,
        render_images=True,
        include_raw_spectra=False,
        max_files=None,
        skip_notes=True,
        notes_file=None,
    )
    payload = run_followup(request)

    assert payload["summary"]["n_files"] == 1
    assert payload["files"][0]["filename"] == "img_1.png"
    assert payload["summary"]["files_source"] == "cache_manifest"
    assert payload["provenance_extra"]["capabilities"]["image_rendering"]["available"] is True

