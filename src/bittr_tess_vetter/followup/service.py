from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bittr_tess_vetter.exofop import ExoFopClient, ExoFopSelectors
from bittr_tess_vetter.followup.processing import (
    classify_followup_file,
    detect_render_capabilities,
    extract_fits_header,
)


def run_followup(request: Any) -> dict[str, Any]:
    cache_dir = Path(request.cache_dir) if request.cache_dir is not None else (Path.cwd() / ".btv_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = ExoFopClient(cache_dir=cache_dir)

    warnings: list[str] = []
    files_source = "none"

    if bool(request.network_ok):
        try:
            selectors = ExoFopSelectors(max_files=int(request.max_files) if request.max_files is not None else None)
            client.fetch_files(target=int(request.tic_id), selectors=selectors, force_refresh=False)
        except Exception as exc:
            warnings.append(f"ExoFOP fetch failed: {type(exc).__name__}: {exc}")
    else:
        warnings.append("Network disabled; using cached follow-up artifacts only.")

    manifest_path = cache_dir / "exofop" / f"tic_{int(request.tic_id)}" / "manifest.json"
    raw_files = _load_manifest_files(manifest_path=manifest_path, warnings=warnings)
    if raw_files:
        files_source = "cache_manifest"

    files: list[dict[str, Any]] = []
    for row in raw_files:
        kind = str(row.get("type") or "Unknown")
        if (not bool(request.include_raw_spectra)) and kind.lower() == "spectrum":
            continue
        file_entry = {
            "file_id": row.get("file_id"),
            "filename": row.get("filename"),
            "type": kind,
            "date_utc": row.get("date_utc"),
            "description": row.get("description"),
            "path": row.get("path"),
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
        }
        classified = classify_followup_file(exofop_type=kind, filename=str(row.get("filename") or ""))
        file_entry["classification"] = {
            "group": classified.exofop_group,
            "extension": classified.extension,
            "format": classified.format,
        }
        if (not bool(request.include_raw_spectra)) and classified.format == "fits" and row.get("path"):
            local_path = (manifest_path.parent / str(row["path"])).resolve()
            hdr = extract_fits_header(local_path, keys=("INSTRUME", "TELESCOP", "OBJECT", "DATE-OBS"))
            file_entry["fits_header_status"] = hdr.status.status
            file_entry["fits_header_reason"] = hdr.status.reason
            if hdr.header:
                file_entry["fits_header"] = hdr.header
        files.append(file_entry)

    if request.max_files is not None:
        files = files[: int(request.max_files)]

    vetting_notes: list[str] = []
    notes_source: str | None = None
    if not bool(request.skip_notes):
        if request.notes_file is not None:
            notes_file = Path(request.notes_file)
            try:
                vetting_notes = [line.strip() for line in notes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                notes_source = str(notes_file)
            except Exception as exc:
                warnings.append(f"Failed to read notes file: {type(exc).__name__}: {exc}")
        else:
            if bool(request.network_ok):
                try:
                    notes_rows = client.obs_notes(tic_id=int(request.tic_id), force_refresh=False)
                    vetting_notes = [str(r.text).strip() for r in notes_rows if str(r.text).strip()]
                    notes_source = "exofop_obsnotes"
                except Exception as exc:
                    warnings.append(f"Obs notes unavailable: {type(exc).__name__}: {exc}")
            else:
                warnings.append("Network disabled; skipping ExoFOP observation notes fetch.")

    render_caps = detect_render_capabilities()
    capabilities = {
        "image_rendering": {
            "requested": bool(request.render_images),
            "available": bool(render_caps.can_render_pdf),
            "used": False,
            "reason": None if bool(render_caps.can_render_pdf) else "renderer_missing",
            "preferred_renderer": render_caps.preferred_renderer,
        },
        "spectra_content": {
            "mode": "raw" if bool(request.include_raw_spectra) else "headers_only",
            "raw_available": bool(request.include_raw_spectra),
            "reason": None if bool(request.include_raw_spectra) else "headers_only_mode",
        },
    }
    if bool(request.render_images) and not bool(render_caps.can_render_pdf):
        warnings.append("Image rendering requested but neither pdftoppm nor gs is available.")

    return {
        "files": files,
        "vetting_notes": vetting_notes,
        "summary": {
            "n_files": int(len(files)),
            "n_vetting_notes": int(len(vetting_notes)),
            "files_source": files_source,
            "notes_source": notes_source,
        },
        "provenance_extra": {
            "cache_manifest": str(manifest_path) if manifest_path.exists() else None,
            "warnings": warnings,
            "capabilities": capabilities,
        },
    }


def _load_manifest_files(*, manifest_path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Failed to read cached manifest: {type(exc).__name__}: {exc}")
        return []
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return []
    out: list[dict[str, Any]] = []
    for row in files:
        if isinstance(row, dict):
            out.append(dict(row))
    return out
