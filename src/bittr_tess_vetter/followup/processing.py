from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from bittr_tess_vetter.followup.types import (
    FitsHeaderExtractionResult,
    FollowupFileClassification,
    FollowupFileProcessingStatus,
    RenderCapabilities,
)

_TYPE_TO_GROUP: dict[str, str] = {
    "image": "imaging",
    "spectrum": "spectroscopy",
    "time series": "timeseries",
    "timeseries": "timeseries",
    "stellar": "stellar",
    "planet": "planet",
}

_EXT_TO_FORMAT: dict[str, str] = {
    "fits": "fits",
    "fit": "fits",
    "fz": "fits",
    "pdf": "pdf",
    "ps": "postscript",
    "eps": "postscript",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "tbl": "table",
    "csv": "table",
    "dat": "table",
    "txt": "text",
    "zip": "archive",
    "gz": "archive",
}


def classify_followup_file(*, exofop_type: str | None, filename: str | Path) -> FollowupFileClassification:
    file_name = Path(filename).name
    type_text = str(exofop_type or "").strip()
    type_norm = type_text.lower()
    group = _TYPE_TO_GROUP.get(type_norm, "unknown")

    suffix = Path(file_name).suffix.lower().lstrip(".")
    fmt = _EXT_TO_FORMAT.get(suffix, "unknown")

    return FollowupFileClassification(
        filename=file_name,
        exofop_type=(type_text or None),
        exofop_group=group,
        extension=suffix,
        format=fmt,
    )


def detect_render_capabilities() -> RenderCapabilities:
    pdftoppm_path = shutil.which("pdftoppm")
    gs_path = shutil.which("gs")
    preferred_renderer = pdftoppm_path or gs_path
    return RenderCapabilities(
        pdftoppm_path=pdftoppm_path,
        gs_path=gs_path,
        can_render_pdf=preferred_renderer is not None,
        preferred_renderer=preferred_renderer,
    )


def extract_fits_header(
    path: str | Path,
    *,
    keys: list[str] | tuple[str, ...] | None = None,
) -> FitsHeaderExtractionResult:
    file_path = Path(path)
    if not file_path.exists():
        status = FollowupFileProcessingStatus(
            filename=file_path.name,
            status="failed",
            reason="FILE_NOT_FOUND",
        )
        return FitsHeaderExtractionResult(path=file_path, header={}, status=status)

    try:
        from astropy.io import fits
    except Exception:
        status = FollowupFileProcessingStatus(
            filename=file_path.name,
            status="skipped",
            reason="ASTROPY_UNAVAILABLE",
        )
        return FitsHeaderExtractionResult(path=file_path, header={}, status=status)

    try:
        header = fits.getheader(file_path)
        raw = dict(header) if keys is None else {str(k): header.get(k) for k in keys}
        serializable = {str(k): _coerce_header_value(v) for k, v in raw.items()}
        status = FollowupFileProcessingStatus(filename=file_path.name, status="ok")
        return FitsHeaderExtractionResult(path=file_path, header=serializable, status=status)
    except Exception as exc:
        status = FollowupFileProcessingStatus(
            filename=file_path.name,
            status="failed",
            reason=type(exc).__name__,
            details={"error": str(exc)},
        )
        return FitsHeaderExtractionResult(path=file_path, header={}, status=status)


def _coerce_header_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "classify_followup_file",
    "detect_render_capabilities",
    "extract_fits_header",
]
