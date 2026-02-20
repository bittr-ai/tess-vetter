from __future__ import annotations

from pathlib import Path

from tess_vetter.followup.processing import (
    classify_followup_file,
    detect_render_capabilities,
    extract_fits_header,
)


def test_classify_followup_file_uses_type_and_extension() -> None:
    classified = classify_followup_file(exofop_type="Image", filename="TOI123_cc.fits")

    assert classified.exofop_group == "imaging"
    assert classified.extension == "fits"
    assert classified.format == "fits"


def test_detect_render_capabilities_prefers_pdftoppm(monkeypatch) -> None:
    def _which(name: str) -> str | None:
        if name == "pdftoppm":
            return "/usr/bin/pdftoppm"
        if name == "gs":
            return "/usr/bin/gs"
        return None

    monkeypatch.setattr("tess_vetter.followup.processing.shutil.which", _which)
    caps = detect_render_capabilities()

    assert caps.can_render_pdf is True
    assert caps.preferred_renderer == "/usr/bin/pdftoppm"


def test_detect_render_capabilities_marks_missing_tools(monkeypatch) -> None:
    monkeypatch.setattr("tess_vetter.followup.processing.shutil.which", lambda _name: None)

    caps = detect_render_capabilities()

    assert caps.can_render_pdf is False
    assert caps.preferred_renderer is None


def test_extract_fits_header_missing_file_returns_failed_status(tmp_path: Path) -> None:
    result = extract_fits_header(tmp_path / "missing.fits")

    assert result.status.status == "failed"
    assert result.status.reason == "FILE_NOT_FOUND"
    assert result.header == {}
