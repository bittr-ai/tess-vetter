"""Shared report-file and sector-loading helpers for diagnostics CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bittr_tess_vetter.cli.common_cli import EXIT_DATA_UNAVAILABLE, BtvCliError, load_json_file
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


@dataclass(frozen=True)
class ResolvedDiagnosticsInputs:
    tic_id: int
    period_days: float
    t0_btjd: float
    duration_hours: float
    depth_ppm: float | None
    sectors_used: list[int] | None
    input_resolution: dict[str, Any]
    report_file_path: str


def _extract_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_obj = payload.get("report")
    if isinstance(report_obj, dict):
        return report_obj
    return payload


def _extract_report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    report_payload = _extract_report_payload(payload)
    summary = report_payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def _extract_report_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    report_payload = _extract_report_payload(payload)
    provenance = report_payload.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _coerce_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"Report file has invalid {field_name}: {value!r}") from exc


def _coerce_required_float(value: Any, *, field_name: str) -> float:
    out = _coerce_optional_float(value, field_name=field_name)
    if out is None:
        raise BtvCliError(f"Report file is missing required field: {field_name}")
    return out


def _coerce_required_int(value: Any, *, field_name: str) -> int:
    if value is None:
        raise BtvCliError(f"Report file is missing required field: {field_name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"Report file has invalid {field_name}: {value!r}") from exc


def _coerce_optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(result)) if result else None


def _extract_report_sectors_used(payload: dict[str, Any]) -> list[int] | None:
    # Prefer report-level provenance, then wrapper/root provenance if available.
    report_provenance = _extract_report_provenance(payload)
    sectors = _coerce_optional_int_list(report_provenance.get("sectors_used"))
    if sectors:
        return sectors

    summary = _extract_report_summary(payload)
    sectors = _coerce_optional_int_list(summary.get("sectors_used"))
    if sectors:
        return sectors

    root_provenance = payload.get("provenance")
    if isinstance(root_provenance, dict):
        sectors = _coerce_optional_int_list(root_provenance.get("sectors_used"))
        if sectors:
            return sectors

    return None


def resolve_inputs_from_report_file(report_file: str) -> ResolvedDiagnosticsInputs:
    report_path = Path(report_file).expanduser()
    payload = load_json_file(report_path, label="report file")
    summary = _extract_report_summary(payload)
    ephemeris_raw = summary.get("ephemeris")
    if not isinstance(ephemeris_raw, dict):
        raise BtvCliError("Report file is missing summary.ephemeris object")

    tic_id = _coerce_required_int(summary.get("tic_id"), field_name="summary.tic_id")
    period_days = _coerce_required_float(ephemeris_raw.get("period_days"), field_name="summary.ephemeris.period_days")
    t0_btjd = _coerce_required_float(ephemeris_raw.get("t0_btjd"), field_name="summary.ephemeris.t0_btjd")
    duration_hours = _coerce_required_float(
        ephemeris_raw.get("duration_hours"),
        field_name="summary.ephemeris.duration_hours",
    )
    depth_ppm = _coerce_optional_float(summary.get("input_depth_ppm"), field_name="summary.input_depth_ppm")
    sectors_used = _extract_report_sectors_used(payload)

    resolved_path = str(report_path.resolve())
    input_resolution = {
        "source": "report_file",
        "resolved_from": "report_file",
        "inputs": {
            "tic_id": int(tic_id),
            "period_days": float(period_days),
            "t0_btjd": float(t0_btjd),
            "duration_hours": float(duration_hours),
            "depth_ppm": float(depth_ppm) if depth_ppm is not None else None,
        },
        "overrides": [],
        "errors": [],
        "report_file": resolved_path,
    }

    return ResolvedDiagnosticsInputs(
        tic_id=int(tic_id),
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        depth_ppm=float(depth_ppm) if depth_ppm is not None else None,
        sectors_used=sectors_used,
        input_resolution=input_resolution,
        report_file_path=resolved_path,
    )


def choose_effective_sectors(
    *,
    sectors_arg: tuple[int, ...],
    report_sectors_used: list[int] | None,
) -> tuple[list[int] | None, bool, str]:
    if sectors_arg:
        return [int(s) for s in sectors_arg], True, "cli"
    if report_sectors_used:
        return [int(s) for s in report_sectors_used], False, "report_file"
    return None, False, "none"


def load_lightcurves_with_sector_policy(
    *,
    tic_id: int,
    sectors: list[int] | None,
    flux_type: str,
    explicit_sectors: bool,
    network_ok: bool = True,
) -> tuple[list[Any], str]:
    client = MASTClient()

    if explicit_sectors:
        if not sectors:
            raise BtvCliError("Internal error: explicit_sectors=True requires non-empty sectors list")
        lightcurves: list[Any] = []
        missing_sectors: list[int] = []
        for sector in sectors:
            try:
                lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type=str(flux_type).lower(),
                    )
                )
            except Exception:
                missing_sectors.append(int(sector))

        if missing_sectors:
            missing = ", ".join(str(s) for s in sorted(set(missing_sectors)))
            raise BtvCliError(
                (
                    f"Cache-only sector load failed for TIC {int(tic_id)}. "
                    f"Missing cached light curve for sector(s): {missing}. "
                    "Populate cache for the requested sectors, or remove --sectors "
                    "to allow MAST discovery/download."
                ),
                exit_code=EXIT_DATA_UNAVAILABLE,
            )

        return lightcurves, "cache_only_explicit_sectors"

    normalized_flux_type = str(flux_type).lower()
    if sectors:
        requested_sectors = sorted({int(s) for s in sectors})
        cached_lightcurves: list[Any] = []
        missing_sectors: list[int] = []
        for sector in requested_sectors:
            try:
                cached_lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type=normalized_flux_type,
                    )
                )
            except Exception:
                missing_sectors.append(int(sector))

        if not missing_sectors:
            return sorted(cached_lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "cache_first_filtered"

        if not bool(network_ok):
            missing = ", ".join(str(s) for s in sorted(set(missing_sectors)))
            raise BtvCliError(
                (
                    f"Cache-only load failed for TIC {int(tic_id)} with --no-network. "
                    f"Missing cached light curve for sector(s): {missing}."
                ),
                exit_code=EXIT_DATA_UNAVAILABLE,
            )

        fetched_missing = client.download_all_sectors(
            tic_id=int(tic_id),
            flux_type=normalized_flux_type,
            sectors=missing_sectors,
        )
        lightcurves = cached_lightcurves + fetched_missing
        if not lightcurves:
            raise LightCurveNotFoundError(f"No sectors available for TIC {int(tic_id)}")
        if cached_lightcurves:
            return sorted(lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "cache_then_mast_filtered"
        return sorted(lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "mast_filtered"

    search_cached = getattr(client, "search_lightcurve_cached", None)
    if callable(search_cached):
        cached_results: list[Any] = []
        try:
            cached_results = list(search_cached(tic_id=int(tic_id)))
        except Exception:
            cached_results = []

        cached_sectors = sorted(
            {
                int(getattr(row, "sector"))
                for row in cached_results
                if getattr(row, "sector", None) is not None
            }
        )
        if cached_sectors:
            cached_lightcurves: list[Any] = []
            for sector in cached_sectors:
                try:
                    cached_lightcurves.append(
                        client.download_lightcurve_cached(
                            tic_id=int(tic_id),
                            sector=int(sector),
                            flux_type=normalized_flux_type,
                        )
                    )
                except Exception:
                    continue
            if cached_lightcurves:
                return sorted(cached_lightcurves, key=lambda lc: int(getattr(lc, "sector", 0))), "cache_discovery"

    if not bool(network_ok):
        raise BtvCliError(
            (
                f"No cached sectors available for TIC {int(tic_id)} with --no-network. "
                "Provide --sectors for known cached sectors or enable --network-ok."
            ),
            exit_code=EXIT_DATA_UNAVAILABLE,
        )

    lightcurves = client.download_all_sectors(
        tic_id=int(tic_id),
        flux_type=normalized_flux_type,
        sectors=None,
    )
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {int(tic_id)}")
    if sectors:
        return lightcurves, "mast_filtered"
    return lightcurves, "mast_discovery"


__all__ = [
    "ResolvedDiagnosticsInputs",
    "choose_effective_sectors",
    "load_lightcurves_with_sector_policy",
    "resolve_inputs_from_report_file",
]
