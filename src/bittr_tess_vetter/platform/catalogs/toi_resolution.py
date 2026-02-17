from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from bittr_tess_vetter.platform.catalogs.exofop_toi_table import (
    fetch_exofop_toi_table,
    fetch_exofop_toi_table_for_toi,
)
from bittr_tess_vetter.platform.catalogs.models import SourceRecord


class LookupStatus(str, Enum):
    OK = "ok"
    DATA_UNAVAILABLE = "data_unavailable"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"


class ToiResolutionResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: LookupStatus
    toi_query: str
    tic_id: int | None = None
    matched_toi: str | None = None
    period_days: float | None = None
    t0_btjd: float | None = None
    duration_hours: float | None = None
    depth_ppm: float | None = None
    missing_fields: list[str] = Field(default_factory=list)
    source_record: SourceRecord
    raw_row: dict[str, str] | None = None
    message: str | None = None


class TICCoordinateLookupResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: LookupStatus
    tic_id: int
    ra_deg: float | None = None
    dec_deg: float | None = None
    source_record: SourceRecord | None = None
    attempts: list[SourceRecord] = Field(default_factory=list)
    message: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "timeout" in name or "timed out" in msg or "timeout" in msg


def _status_from_exception(exc: Exception) -> LookupStatus:
    return LookupStatus.TIMEOUT if _is_timeout_error(exc) else LookupStatus.RUNTIME_ERROR


def _to_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except Exception:
            continue
    return None


def _to_int(row: dict[str, str], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return int(float(text))
        except Exception:
            continue
    return None


def _normalize_toi_text(toi: str | float) -> str:
    if isinstance(toi, float):
        return f"{toi:.2f}".rstrip("0").rstrip(".")
    text = str(toi).strip().upper().replace("TOI-", "").replace("TOI", "")
    return text.strip()


def _parse_toi_number(toi: str | float) -> float | None:
    try:
        return float(_normalize_toi_text(toi))
    except Exception:
        return None


def _to_btjd(epoch: float | None) -> float | None:
    if epoch is None:
        return None
    # ExoFOP exports commonly use BJD (absolute) or BTJD/BJD-2457000.
    if epoch > 2_400_000:
        return float(epoch - 2_457_000.0)
    return float(epoch)


def _first_matching_toi_row(rows: list[dict[str, str]], toi_query: str | float) -> dict[str, str] | None:
    toi_num = _parse_toi_number(toi_query)
    toi_text = _normalize_toi_text(toi_query)
    for row in rows:
        row_toi = str(row.get("toi") or "").strip()
        if not row_toi:
            continue
        if toi_num is not None:
            try:
                if abs(float(row_toi) - toi_num) < 1e-6:
                    return row
            except Exception:
                pass
        if row_toi.upper().replace("TOI-", "").replace("TOI", "").strip() == toi_text:
            return row
    return None


def resolve_toi_to_tic_ephemeris_depth(
    toi: str | float,
    *,
    cache_ttl_seconds: int = 24 * 3600,
    disk_cache_dir: str | Path | None = None,
) -> ToiResolutionResult:
    toi_query = str(toi)
    single_source = SourceRecord(
        name="exofop_toi_table_single",
        version="download_toi.php?toi=<TOI>",
        retrieved_at=_now(),
        query=f"toi={toi_query}",
    )
    full_source = SourceRecord(
        name="exofop_toi_table",
        version="download_toi.php",
        retrieved_at=_now(),
        query=f"toi={toi_query}",
    )

    matched: dict[str, str] | None = None
    resolved_source = single_source
    errors: list[str] = []

    try:
        scoped = fetch_exofop_toi_table_for_toi(
            toi_query,
            cache_ttl_seconds=int(cache_ttl_seconds),
            disk_cache_dir=disk_cache_dir,
        )
        matched = _first_matching_toi_row([dict(r) for r in scoped.rows], toi)
    except Exception as exc:
        errors.append(f"TOI-scoped fetch failed: {type(exc).__name__}: {exc}")

    if matched is None:
        resolved_source = full_source
        try:
            table = fetch_exofop_toi_table(
                cache_ttl_seconds=int(cache_ttl_seconds),
                disk_cache_dir=disk_cache_dir,
            )
        except Exception as exc:
            if errors:
                message = "; ".join(errors + [f"full-table fetch failed: {type(exc).__name__}: {exc}"])
            else:
                message = f"Failed to fetch ExoFOP TOI table: {type(exc).__name__}: {exc}"
            return ToiResolutionResult(
                status=_status_from_exception(exc),
                toi_query=toi_query,
                source_record=full_source,
                message=message,
            )
        matched = _first_matching_toi_row([dict(r) for r in table.rows], toi)
        if matched is None:
            return ToiResolutionResult(
                status=LookupStatus.DATA_UNAVAILABLE,
                toi_query=toi_query,
                source_record=full_source.model_copy(
                    update={"query": f"toi={toi_query} -> no row match in ExoFOP table"}
                ),
                message=f"TOI '{toi_query}' was not found in ExoFOP TOI table",
            )

    tic_id = _to_int(matched, ("tic_id", "tic", "ticid"))
    period_days = _to_float(matched, ("period_days", "period", "per"))
    t0_raw = _to_float(matched, ("epoch_btjd", "epoch_bjd", "epoch", "t0_btjd", "t0"))
    duration_hours = _to_float(
        matched,
        ("duration_hours", "duration_hr", "duration_hrs", "duration", "dur"),
    )
    depth_ppm = _to_float(matched, ("depth_ppm", "depth", "dep_ppt", "dep_ppm"))

    missing_fields: list[str] = []
    if tic_id is None:
        missing_fields.append("tic_id")
    if period_days is None:
        missing_fields.append("period_days")
    if t0_raw is None:
        missing_fields.append("t0_btjd")
    if duration_hours is None:
        missing_fields.append("duration_hours")
    if depth_ppm is None:
        missing_fields.append("depth_ppm")

    required_core_missing = any(
        field in missing_fields for field in ("tic_id", "period_days", "t0_btjd", "duration_hours")
    )
    status = LookupStatus.DATA_UNAVAILABLE if required_core_missing else LookupStatus.OK
    msg = None
    if missing_fields:
        msg = f"Matched TOI row but missing fields: {', '.join(missing_fields)}"
    if errors:
        prefix = "; ".join(errors)
        msg = f"{prefix}; {msg}" if msg else prefix

    return ToiResolutionResult(
        status=status,
        toi_query=toi_query,
        tic_id=tic_id,
        matched_toi=str(matched.get("toi") or toi_query),
        period_days=period_days,
        t0_btjd=_to_btjd(t0_raw),
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        missing_fields=missing_fields,
        source_record=resolved_source.model_copy(
            update={"query": f"toi={toi_query} -> matched toi={matched.get('toi')}"}
        ),
        raw_row={str(k): str(v) for k, v in matched.items()},
        message=msg,
    )


def _lookup_tic_coords_from_mast(
    tic_id: int,
    *,
    cache_dir: str | None = None,
) -> tuple[float | None, float | None, SourceRecord, LookupStatus, str | None]:
    from bittr_tess_vetter.platform.io.mast_client import (
        MASTClient,
        MASTClientError,
        TargetNotFoundError,
    )

    source = SourceRecord(
        name="mast_tic",
        version="Catalogs.query_criteria",
        retrieved_at=_now(),
        query=f"TIC ID={int(tic_id)}",
    )
    try:
        client = MASTClient(cache_dir=cache_dir)
        target = client.get_target_info(int(tic_id))
    except TargetNotFoundError as exc:
        return None, None, source, LookupStatus.DATA_UNAVAILABLE, str(exc)
    except MASTClientError as exc:
        return None, None, source, _status_from_exception(exc), str(exc)
    except Exception as exc:
        return None, None, source, _status_from_exception(exc), f"{type(exc).__name__}: {exc}"

    ra = float(target.ra) if target.ra is not None else None
    dec = float(target.dec) if target.dec is not None else None
    if ra is None or dec is None:
        return ra, dec, source, LookupStatus.DATA_UNAVAILABLE, "TIC query returned no RA/Dec"
    return ra, dec, source, LookupStatus.OK, None


def _lookup_tic_coords_from_exofop(
    tic_id: int,
    *,
    cache_ttl_seconds: int,
    disk_cache_dir: str | Path | None,
) -> tuple[float | None, float | None, SourceRecord, LookupStatus, str | None]:
    source = SourceRecord(
        name="exofop_toi_table",
        version="download_toi.php",
        retrieved_at=_now(),
        query=f"tic_id={int(tic_id)}",
    )
    try:
        table = fetch_exofop_toi_table(
            cache_ttl_seconds=int(cache_ttl_seconds),
            disk_cache_dir=disk_cache_dir,
        )
    except Exception as exc:
        return None, None, source, _status_from_exception(exc), f"{type(exc).__name__}: {exc}"

    rows = table.entries_for_tic(int(tic_id))
    if not rows:
        return None, None, source, LookupStatus.DATA_UNAVAILABLE, "No ExoFOP TOI rows for TIC"

    for row in rows:
        r = dict(row)
        ra = _to_float(r, ("ra_deg", "ra", "raj2000"))
        dec = _to_float(r, ("dec_deg", "dec", "dej2000"))
        if ra is not None and dec is not None:
            return ra, dec, source, LookupStatus.OK, None

    return None, None, source, LookupStatus.DATA_UNAVAILABLE, "ExoFOP rows missing RA/Dec fields"


def lookup_tic_coordinates(
    tic_id: int,
    *,
    cache_ttl_seconds: int = 24 * 3600,
    disk_cache_dir: str | Path | None = None,
    mast_cache_dir: str | None = None,
) -> TICCoordinateLookupResult:
    tic = int(tic_id)
    attempts: list[SourceRecord] = []

    ra, dec, source, status, msg = _lookup_tic_coords_from_mast(tic, cache_dir=mast_cache_dir)
    attempts.append(source.model_copy(update={"query": f"{source.query}; status={status.value}"}))
    if status == LookupStatus.OK and ra is not None and dec is not None:
        return TICCoordinateLookupResult(
            status=LookupStatus.OK,
            tic_id=tic,
            ra_deg=ra,
            dec_deg=dec,
            source_record=source,
            attempts=attempts,
        )

    ra2, dec2, source2, status2, msg2 = _lookup_tic_coords_from_exofop(
        tic,
        cache_ttl_seconds=cache_ttl_seconds,
        disk_cache_dir=disk_cache_dir,
    )
    attempts.append(
        source2.model_copy(update={"query": f"{source2.query}; status={status2.value}"})
    )
    if status2 == LookupStatus.OK and ra2 is not None and dec2 is not None:
        return TICCoordinateLookupResult(
            status=LookupStatus.OK,
            tic_id=tic,
            ra_deg=ra2,
            dec_deg=dec2,
            source_record=source2,
            attempts=attempts,
        )

    if status in {LookupStatus.TIMEOUT, LookupStatus.RUNTIME_ERROR}:
        return TICCoordinateLookupResult(
            status=status,
            tic_id=tic,
            attempts=attempts,
            message=msg,
        )
    if status2 in {LookupStatus.TIMEOUT, LookupStatus.RUNTIME_ERROR}:
        return TICCoordinateLookupResult(
            status=status2,
            tic_id=tic,
            attempts=attempts,
            message=msg2,
        )

    return TICCoordinateLookupResult(
        status=LookupStatus.DATA_UNAVAILABLE,
        tic_id=tic,
        attempts=attempts,
        message=msg2 or msg or "RA/Dec unavailable for TIC",
    )


__all__ = [
    "LookupStatus",
    "ToiResolutionResult",
    "TICCoordinateLookupResult",
    "resolve_toi_to_tic_ephemeris_depth",
    "lookup_tic_coordinates",
]
