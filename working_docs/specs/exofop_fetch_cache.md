# Spec: ExoFOP Fetch + Cache (Reusable Artifact Ingestion)

Status: draft  
Owner: `bittr-tess-vetter`  
Motivation driver: robust, reproducible ingestion of ExoFOP-TESS follow-up artifacts (contrast curves, spectra, time-series LCs, notes) for downstream validation (e.g., TRICERATOPS+), without manual downloads.

## 1) Problem Statement

We routinely need ExoFOP follow-up artifacts to:
- constrain false positive scenarios (e.g., speckle/AO contrast curves),
- incorporate external photometry (TRICERATOPS+),
- attach provenance to vetting decisions and reports,
- run analyses reproducibly after restarts.

Today, users often download files manually from ExoFOP and copy them into `working_docs/...`, which is:
- error-prone (missing/incorrect files; wrong TIC; wrong bandpass),
- non-reproducible (no manifest; no pinned versions),
- slow (no caching; repeated downloads),
- fragile to ExoFOP service quirks (e.g., logged-in-only/proprietary artifacts).

## 2) Goals / Non-Goals

### Goals
- Provide a **single, reusable API** in `bittr-tess-vetter` to:
  1) resolve target identifiers (TIC/TOI/name),
  2) list available ExoFOP artifacts (file list + summary tables),
  3) download artifacts in bulk or selectively,
  4) cache results locally with deterministic paths,
  5) emit a machine-readable manifest usable by other tools (e.g., `astro-arc-tess` MCP tools).
- Support both public and authenticated (cookie) downloads.
- Be resilient: retries, partial success, clear error reporting.

### Non-Goals (for v1)
- No GUI; no notebook-first UX requirements.
- No attempt to “validate” content correctness (beyond basic checksums/size/type sniffing).
- No mandatory integration into every vetting pipeline path (opt-in consumption by callers).

## 3) User Stories

1) “Given `tic_id=365938305`, download all ExoFOP files and produce a manifest.”
2) “Given `TOI-1262`, only download Gemini speckle contrast curves and TRES spectra.”
3) “Given a TFOPWG member cookie jar, download proprietary files and cache them.”
4) “Given an existing cache, run again and perform **zero network calls** unless refresh is requested.”
5) “Given cached contrast curves, provide a helper that returns a normalized contrast-curve object consumable by TRICERATOPS+.”

## 4) Scope: ExoFOP Endpoints

This spec targets the documented PHP endpoints (base `https://exofop.ipac.caltech.edu/tess/`):

### Identifier resolution / target overview
- `gototicid.php?target=...&json` (or `jsonext`)
- `target.php?id=<tic>&json` (optional; metadata)

### Summary tables (text-returning)
- `download_spect.php?target=...&output=csv`
- `download_imaging.php?target=...&output=csv`
- `download_tseries.php?target=...&output=csv`
- `download_obsnotes.php?tid=<tic>&output=pipe` (or by tag/id)
- `download_stellar.php?id=<tic>` (pipe)
- `download_planet.php?id=<tic>` (pipe)
- `download_uploads.php?target=<tic|toi|name>&output=csv`

### Files (download payloads)
- `download_filelist.php?id=<tic>&output=csv` (index: Type, File Name, File ID, Date, Tag, Description, …)
- `download_files.php?id=<tic>` (tar)
- `download_files_zip.php?id=<tic>` (zip)

If ExoFOP provides a “download a single file by file_id” endpoint, add it as an optimization in v1.1; v1 can do targeted downloads by fetching `download_files_zip.php` and extracting only requested files.

## 5) Proposed API Surface (Python)

Module location (suggested):
- `bittr_tess_vetter/exofop/client.py`
- `bittr_tess_vetter/exofop/types.py`
- `bittr_tess_vetter/exofop/cache.py`

### Core dataclasses / types
```python
@dataclass(frozen=True)
class ExoFopTarget:
    tic_id: int
    toi: str | None
    aliases: dict[str, str]  # e.g., {"hd": "..."}

@dataclass(frozen=True)
class ExoFopFileRow:
    file_id: int
    tic_id: int
    toi: str | None
    filename: str
    type: str  # "Image"|"Spectrum"|...
    date_utc: str | None
    tag: str | int | None
    description: str | None

@dataclass(frozen=True)
class ExoFopFetchResult:
    target: ExoFopTarget
    cache_root: Path
    manifest_path: Path
    files_downloaded: list[Path]
    files_skipped: list[str]
    warnings: list[str]
```

### Client
```python
class ExoFopClient:
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        cookie_jar_path: str | Path | None = None,
        user_agent: str = "bittr-tess-vetter exofop-client",
        timeout_seconds: float = 30.0,
    ): ...

    def resolve_target(self, target: str) -> ExoFopTarget: ...
    def file_list(self, *, tic_id: int) -> list[ExoFopFileRow]: ...

    def download_all_files_zip(self, *, tic_id: int, force: bool = False) -> Path: ...
    def fetch(
        self,
        *,
        target: str | int,
        selectors: "ExoFopSelectors" | None = None,
        include_summaries: bool = True,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult: ...
```

### Selectors
```python
@dataclass(frozen=True)
class ExoFopSelectors:
    types: set[str] | None = None          # e.g. {"Image","Spectrum"}
    filename_regex: str | None = None      # e.g. "sensitivity\\.(dat|pdf)$"
    tag_ids: set[int] | None = None
    max_files: int | None = None
```

## 6) Cache Layout + Manifest

Cache root: `<cache_dir>/exofop/`

Per target:
- `<cache_dir>/exofop/tic_<tic_id>/`
  - `manifest.json`
  - `filelist.csv` (raw index)
  - `summaries/`:
    - `spect.csv`, `imaging.csv`, `tseries.csv`, `obsnotes.pipe`, `stellar.pipe`, `planet.pipe`, `uploads.csv`
  - `files/`:
    - downloaded payloads (`.pdf`, `.dat`, `.fits`, etc.)
  - `archives/`:
    - `files.zip` (optional retained) or `files.tar`

### `manifest.json` schema (v1)
```json
{
  "schema_version": 1,
  "generated_at": "ISO-8601",
  "target": {"tic_id": 365938305, "toi": "1262.01", "aliases": {"hd": "HD 104654"}},
  "auth": {"cookie_jar_used": true},
  "sources": {
    "base_url": "https://exofop.ipac.caltech.edu/tess/",
    "endpoints": ["download_filelist.php", "download_files_zip.php", "..."]
  },
  "filelist": {"path": "filelist.csv", "n_rows": 123},
  "summaries": {"spect": "...", "imaging": "...", "tseries": "..."},
  "files": [
    {"file_id": 441891, "filename": "TIC173640199-01_...png", "type": "Light_Curve", "path": "files/...png", "sha256": "...", "bytes": 12345}
  ],
  "warnings": ["..."],
  "errors": []
}
```

## 7) Authentication (Cookie Jar)

Support a user-supplied cookie jar file created via the documented `curl/wget` flow.

Requirements:
- No credential prompts in library code.
- Cookie jar file is read-only; users manage refresh/expiry themselves.
- Manifest records whether auth cookie jar was used.

## 8) Reliability / Error Handling

- All network calls have timeouts and retries (bounded).
- Partial success is allowed: produce manifest even if some endpoints fail.
- Clearly distinguish:
  - “not found / no data” vs “auth required” vs “network timeout”.
- Ensure file writes are atomic:
  - download to temp file → checksum → move into place.

## 9) Integrations

### 9.1 TRICERATOPS+ consumption
Provide helper to locate and normalize contrast curves:
```python
def find_contrast_curves(manifest_path: Path) -> list[Path]: ...
def load_exofop_contrast_curve_dat(path: Path) -> ContrastCurve: ...
```

### 9.2 `astro-arc-tess` (MCP tools)
`astro-arc-tess` can:
- call this module directly (if both repos co-installed), or
- consume the manifest + cached paths via its own tooling layer.

Minimum interoperability requirement:
- stable on-disk paths + `manifest.json` schema.

## 10) CLI (optional but recommended)

Expose a small CLI for ops:
- `python -m bittr_tess_vetter.exofop fetch --tic 365938305 --types Image Spectrum --regex sensitivity\\.(dat|pdf)$`
- `... --cookie-jar ./mycookie.txt`
- `... --force-refresh`

Outputs:
- prints cache path + manifest path
- non-zero exit on total failure

## 11) Testing Plan

Unit tests (no network):
- Parse/validate `download_filelist.php` CSV (sample fixture).
- Selector filtering behavior (types/regex/tags).
- Manifest writing and deterministic paths.
- Archive extraction and file matching.

Integration tests (optional; gated):
- Use recorded HTTP fixtures or a “live” mode behind an env var.

## 12) Milestones

### v1 (MVP)
- `resolve_target`, `file_list`, `download_files_zip`, `fetch` + cache + manifest.
- Basic selectors + deterministic cache layout.

### v1.1
- Optional single-file download endpoint if available.
- Better “auth required” classification (HTTP status parsing).
- Add convenience helpers for TRICERATOPS+ inputs (contrast curves + external LCs).

