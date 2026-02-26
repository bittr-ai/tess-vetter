"""V3 `btv fpp` CLI with plan/run/sweep/summary/explain surface."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import click

from tess_vetter.api.fpp import calculate_fpp
from tess_vetter.api.triceratops_cache import stage_triceratops_runtime_artifacts
from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    emit_progress,
    load_json_file,
    resolve_optional_output_path,
)
from tess_vetter.cli.fpp_cli import (
    _apply_replicate_output_controls,
    _build_cache_for_fpp,
    _derive_fpp_verdict,
    _estimate_detrended_depth_ppm,
    _is_degenerate_fpp_result,
    _load_auto_stellar_inputs,
    _load_cli_contrast_curve,
    _load_report_inputs,
    _looks_like_timeout,
    _runtime_artifacts_ready,
)
from tess_vetter.cli.stellar_inputs import resolve_stellar_inputs
from tess_vetter.cli.vet_cli import (
    _normalize_detrend_method,
    _resolve_candidate_inputs,
    _validate_detrend_args,
)
from tess_vetter.platform.io import LightCurveNotFoundError, PersistentCache, TargetNotFoundError

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

_PLAN_SCHEMA = "cli.fpp.plan.v1"
_RUN_SCHEMA = "cli.fpp.v4"

_MODE_CHOICES = ("quick", "balanced", "strict")
_PROFILE_CHOICES = ("low", "medium", "high")
_POINT_CHOICES = ("windowed", "expanded")
_SCENARIO_SEED_CHOICES = ("fixed_across_matrix", "scenario_index_offset", "scenario_hash_offset")
_REPLICATE_SEED_CHOICES = ("replicate_index_offset", "deterministic_hash")

_FALLBACK_TO_POINTS = {
    "reduce_points_to_2000": 2000,
    "reduce_points_to_1500": 1500,
    "reduce_points_to_1000": 1000,
    "reduce_points_to_750": 750,
    "reduce_points_to_500": 500,
}
_DEFAULT_FALLBACK = [
    "reduce_points_to_1500",
    "reduce_points_to_1000",
    "reduce_points_to_750",
    "reduce_points_to_500",
    "abort",
]

_MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "quick": {"replicates": 1, "sampler_profile": "low", "point_profile": "windowed"},
    "balanced": {"replicates": 3, "sampler_profile": "medium", "point_profile": "windowed"},
    "strict": {"replicates": 5, "sampler_profile": "high", "point_profile": "expanded"},
}

_SAMPLER_TO_MC_DRAWS = {"low": 50_000, "medium": 100_000, "high": 200_000}
_POINT_TO_MAX_POINTS = {"windowed": 1500, "expanded": 2000}
_MODE_TO_PRESET = {"quick": "fast", "balanced": "standard", "strict": "tutorial"}


@dataclass
class _ResolvedPolicy:
    requested_runtime_policy: dict[str, Any]
    effective_runtime_policy: dict[str, Any]
    resolution_trace: list[dict[str, Any]]
    preset: str
    engine_overrides: dict[str, Any]


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _emit_fpp_replicate_progress(command: str, payload: dict[str, Any]) -> None:
    event = str(payload.get("event") or "")
    idx = payload.get("replicate_index")
    total = payload.get("replicates_total")
    seed = payload.get("seed")
    if event == "replicate_start":
        emit_progress(str(command), "replicate", detail=f"{idx}/{total} seed={seed} start")
    elif event == "replicate_complete":
        status = str(payload.get("status") or "unknown")
        emit_progress(str(command), "replicate", detail=f"{idx}/{total} seed={seed} status={status}")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        if yaml is None:
            raise BtvCliError("PyYAML is required for non-JSON sweep config files.", exit_code=EXIT_RUNTIME_ERROR)
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise BtvCliError("Config file must contain a top-level object.", exit_code=EXIT_INPUT_ERROR)
    return payload


def _load_plan(path: Path) -> dict[str, Any]:
    payload = load_json_file(path, label="fpp-plan")
    if payload.get("schema_version") != _PLAN_SCHEMA:
        raise BtvCliError(
            f"Unsupported plan schema_version '{payload.get('schema_version')}', expected '{_PLAN_SCHEMA}'.",
            exit_code=EXIT_INPUT_ERROR,
        )
    return payload


def _pick_source(cli_value: Any, scenario_value: Any, plan_value: Any, fallback_value: Any) -> tuple[Any, str]:
    if cli_value is not None:
        return cli_value, "cli"
    if scenario_value is not None:
        return scenario_value, "scenario"
    if plan_value is not None:
        return plan_value, "plan_default"
    return fallback_value, "fallback"


def _coerce_int_field(name: str, value: Any, *, min_value: int = 1) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"{name} must be an integer", exit_code=EXIT_INPUT_ERROR) from exc
    if parsed < min_value:
        raise BtvCliError(f"{name} must be >= {min_value}", exit_code=EXIT_INPUT_ERROR)
    return parsed


def _validate_choice(name: str, value: Any, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise BtvCliError(f"{name} must be one of: {', '.join(allowed)}", exit_code=EXIT_INPUT_ERROR)
    return normalized


def _normalize_fallback_policy(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise BtvCliError("fallback_policy must be a list", exit_code=EXIT_INPUT_ERROR)
    out: list[str] = []
    for item in value:
        token = str(item).strip().lower()
        if token not in _FALLBACK_TO_POINTS and token != "abort":
            allowed = list(_FALLBACK_TO_POINTS) + ["abort"]
            raise BtvCliError(
                f"fallback_policy token '{token}' is invalid; allowed: {', '.join(allowed)}",
                exit_code=EXIT_INPUT_ERROR,
            )
        out.append(token)
    return out


def _resolve_policy(
    *,
    cli_policy: dict[str, Any],
    scenario_policy: dict[str, Any],
    plan_policy: dict[str, Any],
) -> _ResolvedPolicy:
    mode_fallback = "balanced"
    mode, mode_source = _pick_source(
        _validate_choice("mode", cli_policy.get("mode"), _MODE_CHOICES),
        _validate_choice("mode", scenario_policy.get("mode"), _MODE_CHOICES),
        _validate_choice("mode", plan_policy.get("mode"), _MODE_CHOICES),
        mode_fallback,
    )

    mode_defaults = _MODE_DEFAULTS[str(mode)]

    sampler_profile, sampler_source = _pick_source(
        _validate_choice("sampler_profile", cli_policy.get("sampler_profile"), _PROFILE_CHOICES),
        _validate_choice("sampler_profile", scenario_policy.get("sampler_profile"), _PROFILE_CHOICES),
        _validate_choice("sampler_profile", plan_policy.get("sampler_profile"), _PROFILE_CHOICES),
        mode_defaults["sampler_profile"],
    )
    point_profile, point_source = _pick_source(
        _validate_choice("point_profile", cli_policy.get("point_profile"), _POINT_CHOICES),
        _validate_choice("point_profile", scenario_policy.get("point_profile"), _POINT_CHOICES),
        _validate_choice("point_profile", plan_policy.get("point_profile"), _POINT_CHOICES),
        mode_defaults["point_profile"],
    )

    replicates, replicates_source = _pick_source(
        _coerce_int_field("replicates", cli_policy.get("replicates"), min_value=1),
        _coerce_int_field("replicates", scenario_policy.get("replicates"), min_value=1),
        _coerce_int_field("replicates", plan_policy.get("replicates"), min_value=1),
        int(mode_defaults["replicates"]),
    )

    fallback_policy, fallback_source = _pick_source(
        _normalize_fallback_policy(cli_policy.get("fallback_policy")),
        _normalize_fallback_policy(scenario_policy.get("fallback_policy")),
        _normalize_fallback_policy(plan_policy.get("fallback_policy")),
        list(_DEFAULT_FALLBACK),
    )

    mc_draws, mc_draws_source = _pick_source(
        _coerce_int_field("mc_draws", cli_policy.get("mc_draws"), min_value=1),
        _coerce_int_field("mc_draws", scenario_policy.get("mc_draws"), min_value=1),
        _coerce_int_field("mc_draws", plan_policy.get("mc_draws"), min_value=1),
        None,
    )
    max_points, max_points_source = _pick_source(
        _coerce_int_field("max_points", cli_policy.get("max_points"), min_value=1),
        _coerce_int_field("max_points", scenario_policy.get("max_points"), min_value=1),
        _coerce_int_field("max_points", plan_policy.get("max_points"), min_value=1),
        None,
    )

    derived_mc_draws = int(_SAMPLER_TO_MC_DRAWS[str(sampler_profile)])
    derived_max_points = int(_POINT_TO_MAX_POINTS[str(point_profile)])

    if mode_source == sampler_source == "cli":
        mode_sampler = str(_MODE_DEFAULTS[str(mode)]["sampler_profile"])
        if str(sampler_profile) != mode_sampler:
            raise BtvCliError(
                "Conflicting same-tier runtime inputs for sampler_profile (mode vs sampler_profile).",
                exit_code=EXIT_INPUT_ERROR,
            )
    if mode_source == point_source == "cli":
        mode_point = str(_MODE_DEFAULTS[str(mode)]["point_profile"])
        if str(point_profile) != mode_point:
            raise BtvCliError(
                "Conflicting same-tier runtime inputs for point_profile (mode vs point_profile).",
                exit_code=EXIT_INPUT_ERROR,
            )

    # Same-tier derived knob conflict rule.
    if mc_draws is not None and mc_draws_source == sampler_source and int(mc_draws) != int(derived_mc_draws):
        raise BtvCliError(
            "Conflicting same-tier runtime inputs for mc_draws (numeric vs sampler profile).",
            exit_code=EXIT_INPUT_ERROR,
        )
    if max_points is not None and max_points_source == point_source and int(max_points) != int(derived_max_points):
        raise BtvCliError(
            "Conflicting same-tier runtime inputs for max_points (numeric vs point profile).",
            exit_code=EXIT_INPUT_ERROR,
        )

    effective_mc_draws = int(mc_draws) if mc_draws is not None else int(derived_mc_draws)
    effective_max_points = int(max_points) if max_points is not None else int(derived_max_points)

    trace: list[dict[str, Any]] = [
        {
            "field": "mode",
            "requested_value": cli_policy.get("mode"),
            "effective_value": mode,
            "source": mode_source,
            "reason": "selected by precedence",
            "collision_group": None,
        },
        {
            "field": "sampler_profile",
            "requested_value": cli_policy.get("sampler_profile"),
            "effective_value": sampler_profile,
            "source": sampler_source,
            "reason": "selected by precedence",
            "collision_group": None,
        },
        {
            "field": "point_profile",
            "requested_value": cli_policy.get("point_profile"),
            "effective_value": point_profile,
            "source": point_source,
            "reason": "selected by precedence",
            "collision_group": None,
        },
        {
            "field": "replicates",
            "requested_value": cli_policy.get("replicates"),
            "effective_value": replicates,
            "source": replicates_source,
            "reason": "selected by precedence",
            "collision_group": None,
        },
        {
            "field": "fallback_policy",
            "requested_value": cli_policy.get("fallback_policy"),
            "effective_value": fallback_policy,
            "source": fallback_source,
            "reason": "selected by precedence",
            "collision_group": None,
        },
        {
            "field": "mc_draws",
            "requested_value": cli_policy.get("mc_draws"),
            "effective_value": effective_mc_draws,
            "source": mc_draws_source if mc_draws is not None else sampler_source,
            "reason": "explicit numeric" if mc_draws is not None else "derived from sampler_profile",
            "collision_group": None,
        },
        {
            "field": "max_points",
            "requested_value": cli_policy.get("max_points"),
            "effective_value": effective_max_points,
            "source": max_points_source if max_points is not None else point_source,
            "reason": "explicit numeric" if max_points is not None else "derived from point_profile",
            "collision_group": None,
        },
    ]

    requested_policy = {
        "mode": cli_policy.get("mode"),
        "replicates": cli_policy.get("replicates"),
        "sampler_profile": cli_policy.get("sampler_profile"),
        "point_profile": cli_policy.get("point_profile"),
        "fallback_policy": cli_policy.get("fallback_policy"),
        "mc_draws": cli_policy.get("mc_draws"),
        "max_points": cli_policy.get("max_points"),
    }
    effective_policy = {
        "mode": mode,
        "replicates": int(replicates),
        "sampler_profile": sampler_profile,
        "point_profile": point_profile,
        "fallback_policy": list(fallback_policy),
        "mc_draws": int(effective_mc_draws),
        "max_points": int(effective_max_points),
    }
    engine_overrides = {"mc_draws": int(effective_mc_draws), "max_points": int(effective_max_points)}
    return _ResolvedPolicy(
        requested_runtime_policy=requested_policy,
        effective_runtime_policy=effective_policy,
        resolution_trace=trace,
        preset=_MODE_TO_PRESET[str(mode)],
        engine_overrides=engine_overrides,
    )


def _execute_with_policy_retry(
    *,
    resolved: _ResolvedPolicy,
    run_attempt,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    fallback_policy = list(resolved.effective_runtime_policy["fallback_policy"])
    overrides = dict(resolved.engine_overrides)
    attempted_overrides: list[dict[str, Any]] = [dict(overrides)]

    for token in fallback_policy:
        if token == "abort":
            break
        if token in _FALLBACK_TO_POINTS:
            alt = int(_FALLBACK_TO_POINTS[token])
            current = overrides.get("max_points")
            if current is None or int(alt) < int(current):
                new_overrides = dict(overrides)
                new_overrides["max_points"] = alt
                attempted_overrides.append(new_overrides)

    for idx, attempt_overrides in enumerate(attempted_overrides, start=1):
        result = run_attempt(attempt_overrides)
        degenerate = _is_degenerate_fpp_result(result)
        attempts.append(
            {
                "attempt": int(idx),
                "max_points": attempt_overrides.get("max_points"),
                "mc_draws": attempt_overrides.get("mc_draws"),
                "degenerate": bool(degenerate),
                "reason": result.get("degenerate_reason"),
            }
        )
        if not degenerate:
            break

    if result is None:
        raise BtvCliError("FPP execution did not return a result.", exit_code=EXIT_RUNTIME_ERROR)
    return result, attempts


def _get_scenario(plan: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenarios = plan.get("scenarios")
    if not isinstance(scenarios, list):
        raise BtvCliError("Plan missing scenarios list.", exit_code=EXIT_INPUT_ERROR)
    for item in scenarios:
        if isinstance(item, dict) and str(item.get("id")) == str(scenario_id):
            return item
    raise BtvCliError(f"Scenario '{scenario_id}' not found in plan.", exit_code=EXIT_INPUT_ERROR)


def _scenario_seed(seed_policy: dict[str, Any], scenario_id: str, scenario_index: int) -> int:
    base_seed = _coerce_int_field("seed_policy.base_seed", seed_policy.get("base_seed"), min_value=0)
    if base_seed is None:
        raise BtvCliError("seed_policy.base_seed is required", exit_code=EXIT_INPUT_ERROR)
    strategy = _validate_choice(
        "seed_policy.scenario_seed_strategy",
        seed_policy.get("scenario_seed_strategy"),
        _SCENARIO_SEED_CHOICES,
    )
    if strategy is None:
        raise BtvCliError("seed_policy.scenario_seed_strategy is required", exit_code=EXIT_INPUT_ERROR)
    if strategy == "fixed_across_matrix":
        return int(base_seed)
    if strategy == "scenario_index_offset":
        return int(base_seed) + int(scenario_index)
    digest = int(hashlib.sha256(str(scenario_id).encode("utf-8")).hexdigest()[:8], 16)
    return int(base_seed) + int(digest % 1_000_000)


def _plan_cache_signature(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm_used: float,
    sectors_loaded: list[int],
    cache_dir: str,
) -> str:
    return _hash_obj(
        {
            "tic_id": int(tic_id),
            "period_days": float(period_days),
            "t0_btjd": float(t0_btjd),
            "duration_hours": float(duration_hours),
            "depth_ppm_used": float(depth_ppm_used),
            "sectors_loaded": [int(s) for s in sectors_loaded],
            "cache_dir": str(cache_dir),
        }
    )


def _run_from_plan(
    *,
    plan: dict[str, Any],
    scenario_id: str,
    cli_policy: dict[str, Any],
    seed: int | None,
    timeout_seconds: float | None,
    replicate_detail: str,
    replicate_errors_limit: int,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    network_ok: bool,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
    command_label: str,
) -> dict[str, Any]:
    scenario = _get_scenario(plan, scenario_id)
    scenario_policy = scenario.get("runtime_policy") if isinstance(scenario.get("runtime_policy"), dict) else {}
    plan_policy = plan.get("runtime_policy_defaults") if isinstance(plan.get("runtime_policy_defaults"), dict) else {}

    resolved = _resolve_policy(cli_policy=cli_policy, scenario_policy=scenario_policy, plan_policy=plan_policy)
    effective_mode = str(resolved.effective_runtime_policy.get("mode", "balanced")).lower()
    if effective_mode in {"balanced", "strict"} and not bool(require_stellar):
        raise BtvCliError(
            "Decision-grade modes (balanced/strict) require stellar inputs. "
            "Use --require-stellar or switch to --mode quick for exploratory runs.",
            exit_code=EXIT_INPUT_ERROR,
        )

    cache = PersistentCache(cache_dir=Path(str(plan["cache_dir"])))
    parsed_contrast_curve, contrast_curve_parse_provenance = _load_cli_contrast_curve(
        contrast_curve=contrast_curve,
        contrast_curve_filter=contrast_curve_filter,
    )

    try:
        resolved_stellar, stellar_resolution = resolve_stellar_inputs(
            tic_id=int(plan["tic_id"]),
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=use_stellar_auto,
            require_stellar=require_stellar,
            auto_loader=(lambda _tic_id: _load_auto_stellar_inputs(_tic_id)) if use_stellar_auto else None,
        )
    except (TargetNotFoundError, LightCurveNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    result, attempts = _execute_with_policy_retry(
        resolved=resolved,
        run_attempt=lambda attempt_overrides: calculate_fpp(
            cache=cache,
            tic_id=int(plan["tic_id"]),
            period=float(plan["period_days"]),
            t0=float(plan["t0_btjd"]),
            depth_ppm=float(plan["depth_ppm_used"]),
            duration_hours=float(plan["duration_hours"]),
            sectors=[int(s) for s in plan["sectors_loaded"]],
            stellar_radius=resolved_stellar.get("radius"),
            stellar_mass=resolved_stellar.get("mass"),
            tmag=resolved_stellar.get("tmag"),
            timeout_seconds=timeout_seconds,
            preset=resolved.preset,
            replicates=int(resolved.effective_runtime_policy["replicates"]),
            seed=seed,
            contrast_curve=parsed_contrast_curve,
            overrides=attempt_overrides,
            allow_network=bool(network_ok),
            progress_hook=lambda payload: _emit_fpp_replicate_progress(command_label, payload),
        ),
    )

    result = _apply_replicate_output_controls(
        result,
        replicate_detail=str(replicate_detail).lower(),
        replicate_errors_limit=int(replicate_errors_limit),
    )
    verdict, verdict_source = _derive_fpp_verdict(result)

    payload: dict[str, Any] = {
        "schema_version": _RUN_SCHEMA,
        "fpp_result": result,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "result": {
            "fpp_result": result,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "provenance": {
            "plan": {
                "schema_version": plan.get("schema_version"),
                "created_at": plan.get("created_at"),
                "signature": plan.get("plan_signature"),
            },
            "inputs": {
                "tic_id": int(plan["tic_id"]),
                "period_days": float(plan["period_days"]),
                "t0_btjd": float(plan["t0_btjd"]),
                "duration_hours": float(plan["duration_hours"]),
                "depth_ppm": float(plan["depth_ppm_used"]),
                "depth_ppm_catalog": plan.get("inputs", {}).get("depth_ppm_catalog"),
                "sectors": [int(s) for s in plan["sectors_loaded"]],
                "sectors_loaded": [int(s) for s in plan["sectors_loaded"]],
            },
            "stellar": stellar_resolution,
            "contrast_curve": {
                "path": str(contrast_curve) if contrast_curve is not None else None,
                "filter": str(parsed_contrast_curve.filter) if parsed_contrast_curve is not None else None,
                "parse_provenance": contrast_curve_parse_provenance,
            },
            "runtime": {
                "scenario_id": str(scenario_id),
                "preset": resolved.preset,
                "seed_requested": seed,
                "seed_effective": result.get("base_seed", seed),
                "timeout_seconds": timeout_seconds,
                "policy_resolution": {
                    "requested_runtime_policy": resolved.requested_runtime_policy,
                    "effective_runtime_policy": resolved.effective_runtime_policy,
                    "resolution_trace": resolved.resolution_trace,
                },
                "degenerate_guard": {
                    "attempts": attempts,
                    "guard_triggered": bool(attempts and attempts[0].get("degenerate")),
                    "fallback_succeeded": bool(any(not a.get("degenerate") for a in attempts[1:])),
                    "final_selected_attempt": int(attempts[-1]["attempt"]) if attempts else 1,
                },
                "replicate_detail": str(replicate_detail).lower(),
                "replicate_errors_limit": int(replicate_errors_limit),
                "network_ok": bool(network_ok),
            },
        },
    }
    payload["provenance"]["stellar"]["quality"] = (
        "decision_grade" if bool(require_stellar) else "exploratory"
    )
    return payload


@click.group("fpp")
def fpp_group() -> None:
    """FPP CLI (v3): plan, run, sweep, summary, explain."""


@click.command("fpp-prepare")
def fpp_prepare_removed_command() -> None:
    """Removed command; use `btv fpp plan`."""
    raise BtvCliError("`btv fpp-prepare` was removed. Use `btv fpp plan`.", exit_code=EXIT_INPUT_ERROR)


@click.command("fpp-run")
def fpp_run_removed_command() -> None:
    """Removed command; use `btv fpp run`."""
    raise BtvCliError("`btv fpp-run` was removed. Use `btv fpp run`.", exit_code=EXIT_INPUT_ERROR)


@fpp_group.command("plan")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label for input resolution.")
@click.option(
    "--report-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional prior CLI report/vet JSON to seed candidate inputs and sectors_used.",
)
@click.option("--detrend", type=str, default=None, help="Optional detrend method.")
@click.option("--detrend-bin-hours", type=float, default=6.0, show_default=True)
@click.option("--detrend-buffer", type=float, default=2.0, show_default=True)
@click.option("--detrend-sigma-clip", type=float, default=5.0, show_default=True)
@click.option("--detrend-cache/--no-detrend-cache", default=False, show_default=True)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option("--cache-only-sectors/--allow-sector-download", default=False, show_default=True)
@click.option("--network-ok/--no-network", default=True, show_default=True)
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--force-restage", is_flag=True, default=False, help="Always re-run staging.")
@click.option("--mode", type=click.Choice(_MODE_CHOICES, case_sensitive=False), default="balanced", show_default=True)
@click.option("--replicates", type=int, default=3, show_default=True)
@click.option("--sampler-profile", type=click.Choice(_PROFILE_CHOICES, case_sensitive=False), default="medium", show_default=True)
@click.option("--point-profile", type=click.Choice(_POINT_CHOICES, case_sensitive=False), default="windowed", show_default=True)
@click.option("--mc-draws", type=int, default=None)
@click.option("--max-points", type=int, default=None)
@click.option("--fallback-step", "fallback_steps", multiple=True, type=str)
@click.option("-o", "--out", "output_plan_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
def fpp_plan_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    report_file: Path | None,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    detrend_cache: bool,
    sectors: tuple[int, ...],
    cache_only_sectors: bool,
    network_ok: bool,
    cache_dir: Path | None,
    timeout_seconds: float | None,
    force_restage: bool,
    mode: str,
    replicates: int,
    sampler_profile: str,
    point_profile: str,
    mc_draws: int | None,
    max_points: int | None,
    fallback_steps: tuple[str, ...],
    output_plan_path: Path,
) -> None:
    """Resolve candidate inputs and stage artifacts into a reusable FPP plan."""
    report_candidate: dict[str, Any] = {}
    report_sectors_used: list[int] | None = None
    if report_file is not None:
        report_candidate, report_sectors_used = _load_report_inputs(report_file)

    candidate_tic_id = tic_id if tic_id is not None else report_candidate.get("tic_id")
    candidate_period_days = period_days if period_days is not None else report_candidate.get("period_days")
    candidate_t0_btjd = t0_btjd if t0_btjd is not None else report_candidate.get("t0_btjd")
    candidate_duration_hours = duration_hours if duration_hours is not None else report_candidate.get("duration_hours")
    candidate_depth_ppm = depth_ppm if depth_ppm is not None else report_candidate.get("depth_ppm")

    requested_sectors = [int(s) for s in sectors] if sectors else None
    effective_sectors = requested_sectors if requested_sectors is not None else report_sectors_used
    cache_only_sector_load = bool(cache_only_sectors) and effective_sectors is not None

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=network_ok,
        toi=toi,
        tic_id=int(candidate_tic_id) if candidate_tic_id is not None else None,
        period_days=float(candidate_period_days) if candidate_period_days is not None else None,
        t0_btjd=float(candidate_t0_btjd) if candidate_t0_btjd is not None else None,
        duration_hours=float(candidate_duration_hours) if candidate_duration_hours is not None else None,
        depth_ppm=float(candidate_depth_ppm) if candidate_depth_ppm is not None else None,
    )

    detrend_method = _normalize_detrend_method(detrend)
    if detrend_method is not None:
        _validate_detrend_args(
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
        )
    detrend_cache_requested = bool(detrend_cache)
    if detrend_cache_requested and detrend_method is None:
        raise BtvCliError("--detrend-cache requires --detrend", exit_code=EXIT_INPUT_ERROR)
    detrend_cache_effective = detrend_cache_requested or (detrend_method is not None)

    depth_source = "catalog"
    depth_ppm_used = resolved_depth_ppm
    detrended_depth_meta: dict[str, Any] | None = None
    if depth_ppm is not None:
        depth_source = "explicit"
        depth_ppm_used = float(depth_ppm)
    elif detrend_method is not None:
        detrended_depth, detrended_depth_meta = _estimate_detrended_depth_ppm(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            catalog_depth_ppm=resolved_depth_ppm,
            sectors=effective_sectors,
            detrend_method=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
            cache_dir=cache_dir,
            cache_only_sectors=cache_only_sector_load,
        )
        if detrended_depth is not None:
            depth_ppm_used = float(detrended_depth)
            depth_source = "detrended"

    if depth_ppm_used is None:
        raise BtvCliError(
            "Missing transit depth. Provide --depth-ppm, enable --detrend, or use --toi with depth metadata.",
            exit_code=EXIT_INPUT_ERROR,
        )

    cache, sectors_loaded = _build_cache_for_fpp(
        tic_id=resolved_tic_id,
        sectors=effective_sectors,
        cache_dir=cache_dir,
        detrend_cache=bool(detrend_cache_effective),
        period_days=float(resolved_period_days),
        t0_btjd=float(resolved_t0_btjd),
        duration_hours=float(resolved_duration_hours),
        depth_ppm=float(depth_ppm_used),
        detrend_method=detrend_method,
        detrend_bin_hours=float(detrend_bin_hours),
        detrend_buffer=float(detrend_buffer),
        detrend_sigma_clip=float(detrend_sigma_clip),
        cache_only_sectors=cache_only_sector_load,
    )

    runtime_artifacts: dict[str, Any]
    if not force_restage:
        ready, details = _runtime_artifacts_ready(
            cache_dir=Path(cache.cache_dir),
            tic_id=int(resolved_tic_id),
            sectors_loaded=[int(s) for s in sectors_loaded],
        )
        if ready:
            runtime_artifacts = {
                **details,
                "target_cached": True,
                "trilegal_cached": True,
                "staged_with_network": bool(network_ok),
                "timeout_seconds_requested": float(timeout_seconds) if timeout_seconds is not None else None,
                "reused": True,
            }
        else:
            runtime_artifacts = {
                **details,
                "target_cached": bool(details.get("target_cached")),
                "trilegal_cached": bool(details.get("trilegal_cached")),
                "staged_with_network": bool(network_ok),
                "timeout_seconds_requested": float(timeout_seconds) if timeout_seconds is not None else None,
                "reused": False,
            }
    else:
        runtime_artifacts = {
            "target_cached": False,
            "trilegal_cached": False,
            "trilegal_csv_path": None,
            "staged_with_network": bool(network_ok),
            "timeout_seconds_requested": float(timeout_seconds) if timeout_seconds is not None else None,
            "reused": False,
        }

    if network_ok and (force_restage or not runtime_artifacts.get("trilegal_cached")):
        stage_result = stage_triceratops_runtime_artifacts(
            cache=cache,
            tic_id=int(resolved_tic_id),
            sectors=[int(s) for s in sectors_loaded],
            timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
        )
        runtime_artifacts.update(
            {
                "target_cached": True,
                "trilegal_cached": bool(stage_result.get("trilegal_csv_path")),
                "trilegal_csv_path": stage_result.get("trilegal_csv_path"),
                "target_cache_hit": bool(stage_result.get("target_cache_hit", False)),
                "trilegal_cache_hit": bool(stage_result.get("trilegal_cache_hit", False)),
                "runtime_seconds": stage_result.get("runtime_seconds"),
                "stage_state_path": stage_result.get("stage_state_path"),
                "reused": False,
            }
        )

    fallback_policy = list(fallback_steps) if fallback_steps else list(_DEFAULT_FALLBACK)
    runtime_policy_defaults = {
        "mode": str(mode).lower(),
        "replicates": int(replicates),
        "sampler_profile": str(sampler_profile).lower(),
        "point_profile": str(point_profile).lower(),
        "fallback_policy": [str(x).lower() for x in fallback_policy],
        "mc_draws": int(mc_draws) if mc_draws is not None else None,
        "max_points": int(max_points) if max_points is not None else None,
    }

    plan_signature = _plan_cache_signature(
        tic_id=int(resolved_tic_id),
        period_days=float(resolved_period_days),
        t0_btjd=float(resolved_t0_btjd),
        duration_hours=float(resolved_duration_hours),
        depth_ppm_used=float(depth_ppm_used),
        sectors_loaded=[int(s) for s in sectors_loaded],
        cache_dir=str(cache.cache_dir),
    )

    payload: dict[str, Any] = {
        "schema_version": _PLAN_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "tic_id": int(resolved_tic_id),
        "period_days": float(resolved_period_days),
        "t0_btjd": float(resolved_t0_btjd),
        "duration_hours": float(resolved_duration_hours),
        "depth_ppm_used": float(depth_ppm_used),
        "sectors_loaded": [int(s) for s in sectors_loaded],
        "cache_dir": str(cache.cache_dir),
        "runtime_artifacts": runtime_artifacts,
        "runtime_policy_defaults": runtime_policy_defaults,
        "scenarios": [{"id": "default", "runtime_policy": {}}],
        "inputs": {
            "depth_ppm_catalog": resolved_depth_ppm,
            "resolved_source": input_resolution.get("source"),
            "resolved_from": input_resolution.get("resolved_from"),
            "requested_sectors": effective_sectors,
            "detrend": {
                "method": detrend_method,
                "cache_applied": bool(detrend_cache_effective),
                "cache_requested": bool(detrend_cache_requested),
                "bin_hours": float(detrend_bin_hours),
                "buffer_factor": float(detrend_buffer),
                "sigma_clip": float(detrend_sigma_clip),
                "depth_source": depth_source,
                "depth_meta": detrended_depth_meta,
            },
        },
        "plan_signature": plan_signature,
    }

    dump_json_output(payload, output_plan_path)


@fpp_group.command("run")
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--scenario-id", type=str, default="default", show_default=True)
@click.option("--mode", type=click.Choice(_MODE_CHOICES, case_sensitive=False), default=None)
@click.option("--replicates", type=int, default=None)
@click.option("--sampler-profile", type=click.Choice(_PROFILE_CHOICES, case_sensitive=False), default=None)
@click.option("--point-profile", type=click.Choice(_POINT_CHOICES, case_sensitive=False), default=None)
@click.option("--mc-draws", type=int, default=None)
@click.option("--max-points", type=int, default=None)
@click.option("--fallback-step", "fallback_steps", multiple=True, type=str)
@click.option("--seed", type=int, default=None)
@click.option("--timeout-seconds", type=float, default=None)
@click.option(
    "--replicate-detail",
    type=click.Choice(["full", "compact"], case_sensitive=False),
    default="full",
    show_default=True,
)
@click.option("--replicate-errors-limit", type=int, default=0, show_default=True)
@click.option("--stellar-radius", type=float, default=None)
@click.option("--stellar-mass", type=float, default=None)
@click.option("--stellar-tmag", type=float, default=None)
@click.option("--stellar-file", type=str, default=None)
@click.option("--use-stellar-auto/--no-use-stellar-auto", default=True, show_default=True)
@click.option("--require-stellar/--no-require-stellar", default=True, show_default=True)
@click.option("--network-ok/--no-network", default=True, show_default=True)
@click.option("--contrast-curve", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--contrast-curve-filter", type=str, default=None)
@click.option("-o", "--out", "output_path_arg", type=str, default="-", show_default=True)
def fpp_run_command_v3(
    plan_path: Path,
    scenario_id: str,
    mode: str | None,
    replicates: int | None,
    sampler_profile: str | None,
    point_profile: str | None,
    mc_draws: int | None,
    max_points: int | None,
    fallback_steps: tuple[str, ...],
    seed: int | None,
    timeout_seconds: float | None,
    replicate_detail: str,
    replicate_errors_limit: int,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    network_ok: bool,
    contrast_curve: Path | None,
    contrast_curve_filter: str | None,
    output_path_arg: str,
) -> None:
    """Run one FPP scenario from a v3 plan artifact."""
    out_path = resolve_optional_output_path(output_path_arg)
    plan = _load_plan(plan_path)

    cli_policy = {
        "mode": str(mode).lower() if mode is not None else None,
        "replicates": replicates,
        "sampler_profile": str(sampler_profile).lower() if sampler_profile is not None else None,
        "point_profile": str(point_profile).lower() if point_profile is not None else None,
        "mc_draws": mc_draws,
        "max_points": max_points,
        "fallback_policy": [str(x).lower() for x in fallback_steps] if fallback_steps else None,
    }

    payload = _run_from_plan(
        plan=plan,
        scenario_id=str(scenario_id),
        cli_policy=cli_policy,
        seed=seed,
        timeout_seconds=timeout_seconds,
        replicate_detail=str(replicate_detail).lower(),
        replicate_errors_limit=int(replicate_errors_limit),
        stellar_radius=stellar_radius,
        stellar_mass=stellar_mass,
        stellar_tmag=stellar_tmag,
        stellar_file=stellar_file,
        use_stellar_auto=use_stellar_auto,
        require_stellar=require_stellar,
        network_ok=bool(network_ok),
        contrast_curve=contrast_curve,
        contrast_curve_filter=contrast_curve_filter,
        command_label="fpp-run",
    )
    dump_json_output(payload, out_path)


@fpp_group.command("summary")
@click.option("--from", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--out", "output_path_arg", type=str, default="-", show_default=True)
def fpp_summary_command(input_path: Path, output_path_arg: str) -> None:
    """Render concise outcome summary from an FPP run payload."""
    payload = load_json_file(input_path, label="fpp-result")
    fpp_result = payload.get("fpp_result") if isinstance(payload, dict) else None
    if not isinstance(fpp_result, dict):
        raise BtvCliError("Invalid result payload: missing fpp_result", exit_code=EXIT_INPUT_ERROR)
    rep = fpp_result.get("replicate_analysis") if isinstance(fpp_result.get("replicate_analysis"), dict) else {}
    out = {
        "schema_version": "cli.fpp.summary.v1",
        "source": str(input_path),
        "summary": {
            "fpp": fpp_result.get("fpp"),
            "disposition": fpp_result.get("disposition"),
            "replicates": fpp_result.get("replicates"),
            "n_success": fpp_result.get("n_success"),
            "n_fail": fpp_result.get("n_fail"),
            "effective_config_hash": fpp_result.get("effective_config_hash"),
            "replicate_summary": rep.get("summary"),
        },
    }
    dump_json_output(out, resolve_optional_output_path(output_path_arg))


@fpp_group.command("explain")
@click.option("--from", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--out", "output_path_arg", type=str, default="-", show_default=True)
def fpp_explain_command(input_path: Path, output_path_arg: str) -> None:
    """Explain policy resolution and fallback behavior for an FPP run payload."""
    payload = load_json_file(input_path, label="fpp-result")
    runtime = (
        payload.get("provenance", {}).get("runtime")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(runtime, dict):
        raise BtvCliError("Invalid result payload: missing provenance.runtime", exit_code=EXIT_INPUT_ERROR)

    out = {
        "schema_version": "cli.fpp.explain.v1",
        "source": str(input_path),
        "scenario_id": runtime.get("scenario_id"),
        "policy_resolution": runtime.get("policy_resolution"),
        "degenerate_guard": runtime.get("degenerate_guard"),
    }
    dump_json_output(out, resolve_optional_output_path(output_path_arg))


@fpp_group.command("sweep")
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--network-ok/--no-network", default=True, show_default=True)
@click.option("--stellar-radius", type=float, default=None)
@click.option("--stellar-mass", type=float, default=None)
@click.option("--stellar-tmag", type=float, default=None)
@click.option("--stellar-file", type=str, default=None)
@click.option("--use-stellar-auto/--no-use-stellar-auto", default=True, show_default=True)
@click.option("--require-stellar/--no-require-stellar", default=True, show_default=True)
def fpp_sweep_command(
    plan_path: Path,
    config_path: Path,
    out_dir: Path,
    timeout_seconds: float | None,
    network_ok: bool,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
) -> None:
    """Run a declarative FPP scenario matrix from a plan artifact."""
    plan = _load_plan(plan_path)
    config = _read_yaml_or_json(config_path)

    for key in ("target", "base_runtime_policy", "matrix", "seed_policy", "execution", "outputs"):
        if key not in config:
            raise BtvCliError(f"Sweep config missing required block '{key}'", exit_code=EXIT_INPUT_ERROR)

    matrix = config.get("matrix")
    if not isinstance(matrix, dict) or not matrix:
        raise BtvCliError("matrix must be a non-empty object", exit_code=EXIT_INPUT_ERROR)

    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise BtvCliError("execution must be an object", exit_code=EXIT_INPUT_ERROR)
    parallelism = str(execution.get("parallelism", "sequential")).lower()
    if parallelism not in {"sequential", "parallel"}:
        raise BtvCliError("execution.parallelism must be 'sequential' or 'parallel'", exit_code=EXIT_INPUT_ERROR)

    seed_policy = config.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise BtvCliError("seed_policy must be an object", exit_code=EXIT_INPUT_ERROR)
    _ = _validate_choice(
        "seed_policy.replicate_seed_strategy",
        seed_policy.get("replicate_seed_strategy"),
        _REPLICATE_SEED_CHOICES,
    )
    lock_seed_to_matrix = bool(seed_policy.get("lock_seed_to_matrix", True))

    base_runtime_policy = config.get("base_runtime_policy")
    if not isinstance(base_runtime_policy, dict):
        raise BtvCliError("base_runtime_policy must be an object", exit_code=EXIT_INPUT_ERROR)

    keys = list(matrix.keys())
    values: list[list[Any]] = []
    for key in keys:
        raw = matrix.get(key)
        if not isinstance(raw, list) or not raw:
            raise BtvCliError(f"matrix.{key} must be a non-empty list", exit_code=EXIT_INPUT_ERROR)
        values.append(raw)

    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_outputs_dir = out_dir / "scenarios"
    scenario_outputs_dir.mkdir(parents=True, exist_ok=True)

    for idx, combo in enumerate(product(*values), start=1):
        combo_policy = dict(zip(keys, combo, strict=True))
        cli_policy = dict(base_runtime_policy)
        cli_policy.update(combo_policy)
        fallback_policy = cli_policy.get("fallback_policy")
        if fallback_policy is not None and not isinstance(fallback_policy, list):
            raise BtvCliError("fallback_policy must be a list", exit_code=EXIT_INPUT_ERROR)
        scenario_id = str(combo_policy.get("scenario_id") or f"scenario_{idx:03d}")
        scenario_seed = _scenario_seed(seed_policy, scenario_id, idx)
        if not lock_seed_to_matrix and combo_policy.get("seed") is not None:
            scenario_seed = int(combo_policy.get("seed"))

        run_payload = _run_from_plan(
            plan=plan,
            scenario_id="default",
            cli_policy={
                "mode": cli_policy.get("mode"),
                "replicates": cli_policy.get("replicates"),
                "sampler_profile": cli_policy.get("sampler_profile"),
                "point_profile": cli_policy.get("point_profile"),
                "mc_draws": cli_policy.get("mc_draws"),
                "max_points": cli_policy.get("max_points"),
                "fallback_policy": cli_policy.get("fallback_policy"),
            },
            seed=int(scenario_seed),
            timeout_seconds=timeout_seconds,
            replicate_detail="full",
            replicate_errors_limit=0,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=bool(use_stellar_auto),
            require_stellar=bool(require_stellar),
            network_ok=bool(network_ok),
            contrast_curve=None,
            contrast_curve_filter=None,
            command_label="fpp-sweep",
        )

        scenario_out = scenario_outputs_dir / f"{scenario_id}.json"
        dump_json_output(run_payload, scenario_out)

        fpp_result = run_payload.get("fpp_result", {})
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_index": idx,
                "seed": int(scenario_seed),
                "fpp": fpp_result.get("fpp"),
                "disposition": fpp_result.get("disposition"),
                "effective_config_hash": fpp_result.get("effective_config_hash"),
                "path": str(scenario_out),
                "policy_resolution": run_payload.get("provenance", {}).get("runtime", {}).get("policy_resolution"),
                "degenerate_guard": run_payload.get("provenance", {}).get("runtime", {}).get("degenerate_guard"),
                "redundancy_group_id": None,
            }
        )

    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("effective_config_hash") or "")
        by_hash.setdefault(key, []).append(row)

    redundancy_groups: list[dict[str, Any]] = []
    group_num = 1
    for hsh, members in by_hash.items():
        if not hsh or len(members) < 2:
            continue
        group_id = f"redundancy_{group_num:03d}"
        group_num += 1
        for member in members:
            member["redundancy_group_id"] = group_id
            policy = member.get("policy_resolution")
            if isinstance(policy, dict):
                trace = policy.get("resolution_trace")
                if isinstance(trace, list):
                    for entry in trace:
                        if isinstance(entry, dict):
                            entry["collision_group"] = group_id
        redundancy_groups.append(
            {
                "group_id": group_id,
                "effective_config_hash": hsh,
                "scenario_ids": [str(m.get("scenario_id")) for m in members],
                "count": len(members),
            }
        )

    matrix_summary = {"schema_version": "cli.fpp.matrix_summary.v1", "rows": rows}
    dump_json_output(matrix_summary, out_dir / "matrix_summary.json")

    sorted_rows = sorted(rows, key=lambda r: float(r.get("fpp") if r.get("fpp") is not None else float("inf")))
    with (out_dir / "matrix_ranked.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scenario_id",
                "scenario_index",
                "seed",
                "fpp",
                "disposition",
                "effective_config_hash",
                "redundancy_group_id",
                "path",
            ],
        )
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    dump_json_output(
        {
            "schema_version": "cli.fpp.sweep_explain.v1",
            "scenarios": [
                {
                    "scenario_id": row.get("scenario_id"),
                    "policy_resolution": row.get("policy_resolution"),
                    "degenerate_guard": row.get("degenerate_guard"),
                    "redundancy_group_id": row.get("redundancy_group_id"),
                }
                for row in rows
            ],
        },
        out_dir / "sweep_explain.json",
    )

    dump_json_output(
        {
            "schema_version": "cli.fpp.matrix_redundancy.v1",
            "groups": redundancy_groups,
        },
        out_dir / "matrix_redundancy.json",
    )


__all__ = [
    "fpp_group",
    "fpp_prepare_removed_command",
    "fpp_run_removed_command",
    "fpp_plan_command",
    "fpp_run_command_v3",
    "fpp_sweep_command",
    "fpp_summary_command",
    "fpp_explain_command",
]
