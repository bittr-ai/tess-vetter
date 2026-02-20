from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExoFopTarget:
    tic_id: int
    toi: str | None = None
    aliases: dict[str, str] | None = None


@dataclass(frozen=True)
class ExoFopFileRow:
    file_id: int
    tic_id: int
    toi: str | None
    filename: str
    type: str
    date_utc: str | None = None
    user: str | None = None
    group: str | None = None
    tag: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ExoFopSelectors:
    types: set[str] | None = None
    filename_regex: str | None = None
    tag_ids: set[int] | None = None
    max_files: int | None = None


@dataclass(frozen=True)
class ExoFopFetchResult:
    target: ExoFopTarget
    cache_root: Path
    manifest_path: Path
    filelist_path: Path
    summary_paths: dict[str, Path]
    files_downloaded: list[Path]
    files_skipped: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ExoFopObsNoteRow:
    tic_id: int
    author: str | None
    date_utc: str | None
    data_tag: str | None
    text: str
    note_id: int | None = None
    group: str | None = None
    raw: dict[str, str] | None = None


@dataclass(frozen=True)
class ExoFopToiRow:
    tic_id: int
    toi: str | None
    tfopwg_disposition: str | None
    planet_disposition: str | None
    comments: str | None
    raw: dict[str, str]
