from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from bittr_tess_vetter.exofop.types import (
    ExoFopFetchResult,
    ExoFopFileRow,
    ExoFopSelectors,
    ExoFopTarget,
)


class ExoFopClient:
    BASE_URL = "https://exofop.ipac.caltech.edu/tess/"

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        cookie_jar_path: str | Path | None = None,
        user_agent: str = "bittr-tess-vetter exofop-client",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cookie_jar_path = Path(cookie_jar_path) if cookie_jar_path else None
        self.user_agent = user_agent
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._cookie_loaded = False
        if self.cookie_jar_path is not None:
            self._cookie_loaded = self._load_cookie_jar(self.cookie_jar_path)

    # -----------------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------------

    @property
    def cookie_jar_loaded(self) -> bool:
        return self._cookie_loaded

    def resolve_target(self, target: str | int) -> ExoFopTarget:
        if isinstance(target, int):
            return ExoFopTarget(tic_id=int(target), toi=None, aliases={})
        target_str = str(target).strip()
        if target_str.isdigit():
            return ExoFopTarget(tic_id=int(target_str), toi=None, aliases={})

        # Use ExoFOP redirect helper in JSON mode.
        url = self._url("gototicid.php")
        params = {"target": target_str, "jsonext": "1"}
        payload = self._get_json(url, params=params)

        summary = payload.get("summary") if isinstance(payload, dict) else None
        if isinstance(summary, dict) and summary.get("status") == "OK":
            tic_id = int(summary.get("TIC"))
            aliases: dict[str, str] = {}
            for k in ("GAIADR2", "GAIADR3", "KIC", "EPIC"):
                if summary.get(k):
                    aliases[k.lower()] = str(summary.get(k))
            return ExoFopTarget(tic_id=tic_id, toi=None, aliases=aliases)

        # Fallback to simpler json.
        params = {"target": target_str, "json": "1"}
        payload = self._get_json(url, params=params)
        if isinstance(payload, dict) and payload.get("status") == "OK":
            tic_id = int(payload.get("TIC"))
            aliases = {}
            for k in ("GaiaDR2", "GAIADR2", "KIC", "EPIC"):
                if payload.get(k):
                    aliases[k.lower()] = str(payload.get(k))
            return ExoFopTarget(tic_id=tic_id, toi=None, aliases=aliases)

        raise ValueError(
            f"ExoFOP could not resolve target '{target_str}': {payload.get('message') if isinstance(payload, dict) else payload}"
        )

    def file_list(self, *, tic_id: int, force_refresh: bool = False) -> list[ExoFopFileRow]:
        target_dir = self._target_dir(tic_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "filelist.csv"
        if out_path.exists() and not force_refresh:
            return self._parse_filelist_csv(
                out_path.read_text(encoding="utf-8", errors="replace"), fallback_tic_id=int(tic_id)
            )

        url = self._url("download_filelist.php")
        text = self._get_text(url, params={"id": str(int(tic_id)), "output": "csv"})
        self._write_text_atomic(out_path, text)
        return self._parse_filelist_csv(text, fallback_tic_id=int(tic_id))

    def index_target(
        self,
        *,
        target: str | int,
        include_summaries: bool = True,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult:
        exo = self.resolve_target(target)
        target_dir = self._target_dir(exo.tic_id)
        summaries_dir = target_dir / "summaries"
        files_dir = target_dir / "files"
        archives_dir = target_dir / "archives"
        for d in (target_dir, summaries_dir, files_dir, archives_dir):
            d.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        files_downloaded: list[Path] = []
        files_skipped: list[str] = []

        # Always ensure filelist exists.
        file_rows = self.file_list(tic_id=exo.tic_id, force_refresh=force_refresh)
        filelist_path = target_dir / "filelist.csv"

        summary_paths: dict[str, Path] = {}
        if include_summaries:
            summary_specs = {
                "spect": ("download_spect.php", {"target": str(exo.tic_id), "output": "csv"}),
                "imaging": ("download_imaging.php", {"target": str(exo.tic_id), "output": "csv"}),
                "tseries": ("download_tseries.php", {"target": str(exo.tic_id), "output": "csv"}),
                "uploads": ("download_uploads.php", {"target": str(exo.tic_id), "output": "csv"}),
                "stellar": ("download_stellar.php", {"id": str(exo.tic_id), "output": "pipe"}),
                "planet": ("download_planet.php", {"id": str(exo.tic_id), "output": "pipe"}),
            }
            for key, (endpoint, params) in summary_specs.items():
                out_path = summaries_dir / f"{key}.{self._ext_for_output(params.get('output'))}"
                summary_paths[key] = out_path
                if out_path.exists() and not force_refresh:
                    continue
                try:
                    text = self._get_text(self._url(endpoint), params=params)
                    self._write_text_atomic(out_path, text)
                except Exception as e:
                    warnings.append(f"{endpoint} failed: {type(e).__name__}: {e}")

        manifest_path = target_dir / "manifest.json"
        self._write_manifest(
            manifest_path=manifest_path,
            target=exo,
            filelist_path=filelist_path,
            summary_paths=summary_paths,
            files_dir=files_dir,
            file_rows=file_rows,
            downloaded_files=files_downloaded,
            skipped=files_skipped,
            warnings=warnings,
        )

        return ExoFopFetchResult(
            target=exo,
            cache_root=target_dir,
            manifest_path=manifest_path,
            filelist_path=filelist_path,
            summary_paths=summary_paths,
            files_downloaded=files_downloaded,
            files_skipped=files_skipped,
            warnings=warnings,
        )

    def fetch_files(
        self,
        *,
        target: str | int,
        selectors: ExoFopSelectors | None = None,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult:
        result = self.index_target(target=target, include_summaries=True, force_refresh=force_refresh)
        tic_id = result.target.tic_id
        target_dir = result.cache_root
        files_dir = target_dir / "files"
        archives_dir = target_dir / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)
        files_dir.mkdir(parents=True, exist_ok=True)

        rows = self.file_list(tic_id=tic_id, force_refresh=force_refresh)
        selected = self._filter_rows(rows, selectors)
        selected_names = {r.filename for r in selected}

        # Download zip once (or reuse cached) and extract requested files.
        zip_path = archives_dir / "files.zip"
        if (not zip_path.exists()) or force_refresh:
            zip_url = self._url("download_files_zip.php")
            content = self._get_bytes(zip_url, params={"id": str(int(tic_id))})
            self._write_bytes_atomic(zip_path, content)

        extracted = self._extract_selected(zip_path=zip_path, out_dir=files_dir, names=selected_names)

        # Update manifest with new file metadata.
        warnings = list(result.warnings)
        files_skipped: list[str] = []
        for r in selected:
            if r.filename not in extracted:
                files_skipped.append(r.filename)
        self._write_manifest(
            manifest_path=result.manifest_path,
            target=result.target,
            filelist_path=result.filelist_path,
            summary_paths=result.summary_paths,
            files_dir=files_dir,
            file_rows=rows,
            downloaded_files=sorted(extracted.values()),
            skipped=files_skipped,
            warnings=warnings,
        )

        return ExoFopFetchResult(
            target=result.target,
            cache_root=result.cache_root,
            manifest_path=result.manifest_path,
            filelist_path=result.filelist_path,
            summary_paths=result.summary_paths,
            files_downloaded=sorted(extracted.values()),
            files_skipped=files_skipped,
            warnings=warnings,
        )

    def fetch(
        self,
        *,
        target: str | int,
        selectors: ExoFopSelectors | None = None,
        include_summaries: bool = True,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult:
        # Convenience wrapper: index first; optionally download files
        indexed = self.index_target(
            target=target, include_summaries=include_summaries, force_refresh=force_refresh
        )
        if selectors is None:
            return indexed
        return self.fetch_files(target=target, selectors=selectors, force_refresh=force_refresh)

    # -----------------------------------------------------------------------------
    # Networking helpers
    # -----------------------------------------------------------------------------

    def _url(self, endpoint: str) -> str:
        return self.BASE_URL + endpoint

    def _request(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"request failed: {last_exc}")

    def _get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        resp = self._request("GET", url, params=params)
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def _get_bytes(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        resp = self._request("GET", url, params=params)
        return resp.content

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        text = self._get_text(url, params=params)
        return json.loads(text)

    # -----------------------------------------------------------------------------
    # Cookie jar support
    # -----------------------------------------------------------------------------

    def _load_cookie_jar(self, path: Path) -> bool:
        # ExoFOP docs show wget/curl cookie jars; these are commonly Netscape/Mozilla format.
        import http.cookiejar

        jar = http.cookiejar.MozillaCookieJar(str(path))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
        except Exception:
            return False
        # Transfer into requests session cookie jar
        for c in jar:
            self.session.cookies.set_cookie(c)
        return True

    # -----------------------------------------------------------------------------
    # Parsing / filtering
    # -----------------------------------------------------------------------------

    def _parse_filelist_csv(self, text: str, *, fallback_tic_id: int | None = None) -> list[ExoFopFileRow]:
        # ExoFOP uses CSV when output=csv; tolerate stray BOM or whitespace.
        buf = io.StringIO(text.lstrip("\ufeff"))
        reader = csv.DictReader(buf)
        rows: list[ExoFopFileRow] = []
        for r in reader:
            if not r:
                continue
            filename = (r.get("File Name") or r.get("FileName") or r.get("file_name") or "").strip()
            if not filename:
                continue
            file_id_raw = (r.get("File ID") or r.get("FileID") or r.get("file_id") or "").strip()
            # ExoFOP filelist schemas vary. Some exports include a TIC/TIC ID column, while
            # others omit it entirely when querying by TIC id. In the latter case, fall back
            # to the caller-provided tic_id.
            tic_raw = (r.get("TIC") or r.get("TIC ID") or r.get("tic") or "").strip()
            toi = (r.get("TOI") or r.get("toi") or "").strip() or None
            typ = (r.get("Type") or r.get("type") or "").strip() or "Unknown"
            date = (r.get("Date") or r.get("date") or "").strip() or None
            user = (r.get("User") or r.get("user") or "").strip() or None
            group = (r.get("Group") or r.get("group") or "").strip() or None
            tag = (r.get("Tag") or r.get("tag") or "").strip() or None
            desc = (r.get("Description") or r.get("description") or "").strip() or None

            try:
                file_id = int(float(file_id_raw))
            except Exception:
                continue
            if tic_raw:
                try:
                    tic_id = int(float(tic_raw))
                except Exception:
                    tic_id = int(fallback_tic_id) if fallback_tic_id is not None else 0
            else:
                tic_id = int(fallback_tic_id) if fallback_tic_id is not None else 0

            if tic_id <= 0:
                # Without a TIC, we can't reliably associate the entry to a target.
                continue

            rows.append(
                ExoFopFileRow(
                    file_id=file_id,
                    tic_id=tic_id,
                    toi=toi,
                    filename=filename,
                    type=typ,
                    date_utc=date,
                    user=user,
                    group=group,
                    tag=tag,
                    description=desc,
                )
            )
        return rows

    def _filter_rows(
        self, rows: list[ExoFopFileRow], selectors: ExoFopSelectors | None
    ) -> list[ExoFopFileRow]:
        if selectors is None:
            return rows
        out = rows
        if selectors.types:
            allowed = {t.lower() for t in selectors.types}
            out = [r for r in out if r.type.lower() in allowed]
        if selectors.tag_ids:
            allowed_tags = {int(t) for t in selectors.tag_ids}

            def _tag_ok(tag: str | None) -> bool:
                if tag is None:
                    return False
                try:
                    return int(float(tag)) in allowed_tags
                except Exception:
                    return False

            out = [r for r in out if _tag_ok(r.tag)]
        if selectors.filename_regex:
            rx = re.compile(selectors.filename_regex)
            out = [r for r in out if rx.search(r.filename)]
        if selectors.max_files is not None and selectors.max_files >= 0:
            out = out[: int(selectors.max_files)]
        return out

    # -----------------------------------------------------------------------------
    # Files / extraction / hashing
    # -----------------------------------------------------------------------------

    def _extract_selected(self, *, zip_path: Path, out_dir: Path, names: set[str]) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not names:
            return out
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = {Path(n).name: n for n in zf.namelist()}
            for want in names:
                member = members.get(Path(want).name)
                if member is None:
                    continue
                dest = out_dir / Path(want).name
                if dest.exists() and dest.stat().st_size > 0:
                    out[want] = dest
                    continue
                # Extract to temp then move
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with zf.open(member) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
                os.replace(tmp, dest)
                out[want] = dest
        return out

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # -----------------------------------------------------------------------------
    # Cache layout / manifest
    # -----------------------------------------------------------------------------

    def _target_dir(self, tic_id: int) -> Path:
        return self.cache_dir / "exofop" / f"tic_{int(tic_id)}"

    def _write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _write_bytes_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _ext_for_output(self, output: Any) -> str:
        out = str(output or "").lower()
        if out == "csv":
            return "csv"
        if out == "pipe":
            return "pipe"
        return "txt"

    def _write_manifest(
        self,
        *,
        manifest_path: Path,
        target: ExoFopTarget,
        filelist_path: Path,
        summary_paths: dict[str, Path],
        files_dir: Path,
        file_rows: list[ExoFopFileRow],
        downloaded_files: Iterable[Path],
        skipped: list[str],
        warnings: list[str],
    ) -> None:
        files_meta: list[dict[str, Any]] = []
        downloaded_set = {Path(p).name for p in downloaded_files}
        for row in file_rows:
            local_path = files_dir / Path(row.filename).name
            entry: dict[str, Any] = {
                "file_id": row.file_id,
                "filename": row.filename,
                "type": row.type,
                "date_utc": row.date_utc,
                "tag": row.tag,
                "description": row.description,
            }
            if local_path.exists() and local_path.name in downloaded_set:
                try:
                    entry["path"] = str(local_path.relative_to(manifest_path.parent))
                except Exception:
                    entry["path"] = str(local_path)
                entry["bytes"] = int(local_path.stat().st_size)
                entry["sha256"] = self._sha256(local_path)
            files_meta.append(entry)

        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": asdict(target),
            "auth": {"cookie_jar_used": bool(self.cookie_jar_path), "cookie_loaded": self.cookie_jar_loaded},
            "sources": {"base_url": self.BASE_URL},
            "filelist": {"path": str(filelist_path.relative_to(manifest_path.parent)), "n_rows": len(file_rows)},
            "summaries": {k: str(v.relative_to(manifest_path.parent)) for k, v in summary_paths.items()},
            "files": files_meta,
            "skipped": skipped,
            "warnings": warnings,
            "errors": [],
        }
        self._write_text_atomic(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
