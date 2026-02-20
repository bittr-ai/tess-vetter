"""Executor for composable workflow pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tess_vetter.cli.common_cli import EXIT_RUNTIME_ERROR, BtvCliError
from tess_vetter.pipeline_composition.ref_resolver import extract_path, resolve_value
from tess_vetter.pipeline_composition.schema import (
    CompositionSpec,
    StepSpec,
    composition_digest,
)

_OP_TO_COMMAND = {
    "vet": "vet",
    "measure_sectors": "measure-sectors",
    "fit": "fit",
    "report": "report",
    "activity": "activity",
    "rv_feasibility": "rv-feasibility",
    "model_compete": "model-compete",
    "timing": "timing",
    "systematics_proxy": "systematics-proxy",
    "ephemeris_reliability": "ephemeris-reliability",
    "resolve_stellar": "resolve-stellar",
    "resolve_neighbors": "resolve-neighbors",
    "localize_host": "localize-host",
    "dilution": "dilution",
    "detrend_grid": "detrend-grid",
    "contrast_curves": "contrast-curves",
    "contrast_curve_summary": "contrast-curve-summary",
    "fpp": "fpp",
    "fpp_prepare": "fpp-prepare",
    "fpp_run": "fpp-run",
}

_RETRYABLE_TOKENS = ("429", "timeout", "timed out", "temporarily unavailable", "connection reset")
_EXECUTOR_DEFAULT_KEYS = {"retry_max_attempts", "retry_initial_seconds"}
_DETREND_INVARIANCE_POLICY_VERSION = "v1"
_DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD = 0.01


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_retryable_error(message: str) -> bool:
    m = message.lower()
    return any(token in m for token in _RETRYABLE_TOKENS)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_concern_flags(step_payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    summary = step_payload.get("summary")
    if isinstance(summary, dict):
        concerns = summary.get("concerns")
        if isinstance(concerns, list):
            flags.extend(str(x) for x in concerns if x is not None)
    result = step_payload.get("result")
    if isinstance(result, dict):
        consensus = result.get("consensus")
        if isinstance(consensus, dict):
            rflags = consensus.get("reliability_flags")
            if isinstance(rflags, list):
                flags.extend(str(x) for x in rflags if x is not None)
        warnings = result.get("warnings")
        if isinstance(warnings, list):
            flags.extend(str(x) for x in warnings if x is not None)
    warnings_top = step_payload.get("warnings")
    if isinstance(warnings_top, list):
        flags.extend(str(x) for x in warnings_top if x is not None)
    return sorted(set(flags))


def _apply_ports(
    step: StepSpec,
    *,
    step_output_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    ports: dict[str, Any] = {
        "artifact_path": str(step_output_path),
        "report_file": str(step_output_path),
    }
    for port_name, port_spec in step.ports.items():
        if isinstance(port_spec, str):
            if port_spec == "artifact_path":
                ports[port_name] = str(step_output_path)
            else:
                try:
                    ports[port_name] = extract_path(payload, port_spec)
                except Exception as exc:
                    raise BtvCliError(
                        f"Step '{step.id}' failed to resolve port '{port_name}' from '{port_spec}': {exc}",
                        exit_code=EXIT_RUNTIME_ERROR,
                    ) from exc
        else:
            ports[port_name] = port_spec
    return ports


def _build_cli_args(
    *,
    step: StepSpec,
    toi: str,
    inputs: dict[str, Any],
    output_path: Path,
    network_ok: bool,
) -> list[str]:
    if step.op not in _OP_TO_COMMAND:
        allowed = ", ".join(sorted(_OP_TO_COMMAND))
        raise BtvCliError(
            f"Unsupported op '{step.op}'. Allowed ops: {allowed}",
            exit_code=EXIT_RUNTIME_ERROR,
        )

    command = _OP_TO_COMMAND[step.op]
    args = [sys.executable, "-m", "tess_vetter.cli.enrich_cli", command]

    args.extend(["--toi", str(toi)])
    args.append("--network-ok" if network_ok else "--no-network")

    raw_flags = inputs.get("_flags")
    if raw_flags is not None and not isinstance(raw_flags, list):
        raise BtvCliError(
            f"Step '{step.id}' inputs._flags must be a list of CLI flags.",
            exit_code=EXIT_RUNTIME_ERROR,
        )

    raw_args = inputs.get("_args")
    if raw_args is not None and not isinstance(raw_args, list):
        raise BtvCliError(
            f"Step '{step.id}' inputs._args must be a list of CLI tokens.",
            exit_code=EXIT_RUNTIME_ERROR,
        )

    legacy_raw_args = inputs.get("args")
    if legacy_raw_args is not None and not isinstance(legacy_raw_args, list):
        raise BtvCliError(
            f"Step '{step.id}' inputs.args must be a list of CLI tokens.",
            exit_code=EXIT_RUNTIME_ERROR,
        )

    for key, value in inputs.items():
        if key in {"args", "_flags", "_args"}:
            continue
        flag = _flag_name(key)
        if value is None:
            continue
        if isinstance(value, dict) and any(
            k in value for k in {"_value", "_flag_true", "_flag_false"}
        ):
            bool_value = value.get("_value")
            true_flag = value.get("_flag_true")
            false_flag = value.get("_flag_false")
            if not isinstance(bool_value, bool):
                raise BtvCliError(
                    f"Step '{step.id}' input '{key}' must set boolean _value for paired flag form.",
                    exit_code=EXIT_RUNTIME_ERROR,
                )
            if true_flag is None and false_flag is None:
                raise BtvCliError(
                    f"Step '{step.id}' input '{key}' paired flag form must provide _flag_true and/or _flag_false.",
                    exit_code=EXIT_RUNTIME_ERROR,
                )
            if true_flag is not None and not isinstance(true_flag, str):
                raise BtvCliError(
                    f"Step '{step.id}' input '{key}' _flag_true must be a string.",
                    exit_code=EXIT_RUNTIME_ERROR,
                )
            if false_flag is not None and not isinstance(false_flag, str):
                raise BtvCliError(
                    f"Step '{step.id}' input '{key}' _flag_false must be a string.",
                    exit_code=EXIT_RUNTIME_ERROR,
                )
            selected = true_flag if bool_value else false_flag
            if selected:
                args.append(selected)
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])

    if raw_flags:
        args.extend(str(token) for token in raw_flags)

    if raw_args:
        args.extend(str(token) for token in raw_args)

    if legacy_raw_args:
        args.extend(str(token) for token in legacy_raw_args)

    if "--out" not in args and "-o" not in args:
        args.extend(["--out", str(output_path)])

    return args


def _run_step_command(
    *,
    step: StepSpec,
    toi: str,
    inputs: dict[str, Any],
    output_path: Path,
    stderr_path: Path,
    network_ok: bool,
) -> dict[str, Any]:
    args = _build_cli_args(step=step, toi=toi, inputs=inputs, output_path=output_path, network_ok=network_ok)
    proc = subprocess.run(args, capture_output=True, text=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text((proc.stderr or "") + ("\n" + proc.stdout if proc.stdout else ""), encoding="utf-8")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Command failed rc={proc.returncode}: {msg}")

    if not output_path.exists():
        raise RuntimeError(f"Step output file missing: {output_path}")
    return _load_json(output_path)


def _run_step_with_retries(
    *,
    step: StepSpec,
    toi: str,
    inputs: dict[str, Any],
    output_path: Path,
    stderr_path: Path,
    network_ok: bool,
    max_attempts: int,
    initial_backoff_seconds: float,
) -> tuple[dict[str, Any], int]:
    attempt = 0
    last_error: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            payload = _run_step_command(
                step=step,
                toi=toi,
                inputs=inputs,
                output_path=output_path,
                stderr_path=stderr_path,
                network_ok=network_ok,
            )
            return payload, attempt
        except Exception as exc:  # pragma: no cover - exercised through retry behavior tests
            last_error = exc
            if attempt >= max_attempts or not _is_retryable_error(str(exc)):
                break
            sleep_s = initial_backoff_seconds * (2 ** (attempt - 1))
            jitter_upper = min(0.25, max(0.0, float(sleep_s) * 0.1))
            jitter = random.uniform(0.0, jitter_upper) if jitter_upper > 0 else 0.0
            time.sleep(max(0.1, float(sleep_s) + float(jitter)))

    assert last_error is not None
    raise last_error


def _step_fingerprint(*, step: StepSpec, toi: str, inputs: dict[str, Any], network_ok: bool) -> str:
    return _json_hash(
        {
            "step_id": step.id,
            "op": step.op,
            "toi": toi,
            "network_ok": bool(network_ok),
            "inputs": inputs,
        }
    )


def _run_one_toi(
    *,
    composition: CompositionSpec,
    toi: str,
    out_dir: Path,
    network_ok: bool,
    continue_on_error: bool,
    resume: bool,
) -> dict[str, Any]:
    toi_dir = out_dir / toi
    steps_dir = toi_dir / "steps"
    logs_dir = toi_dir / "logs"
    checkpoints_dir = toi_dir / "checkpoints"

    step_outputs: dict[str, dict[str, Any]] = {}
    step_ports: dict[str, dict[str, Any]] = {}
    step_rows: list[dict[str, Any]] = []
    concern_flags: set[str] = set()

    run_status = "ok"
    for idx, step in enumerate(composition.steps, start=1):
        step_name = f"{idx:02d}_{step.id}"
        output_path = steps_dir / f"{step_name}.json"
        stderr_path = logs_dir / f"{step.id}.stderr.log"
        marker_path = checkpoints_dir / f"{step_name}.done.json"

        step_inputs_with_defaults = {
            k: v for k, v in composition.defaults.items() if k not in _EXECUTOR_DEFAULT_KEYS
        }
        step_inputs_with_defaults.update(step.inputs)
        resolved_inputs = resolve_value(
            step_inputs_with_defaults,
            step_outputs=step_outputs,
            step_ports=step_ports,
        )
        if not isinstance(resolved_inputs, dict):
            raise BtvCliError(
                f"Resolved inputs for step '{step.id}' must be an object.",
                exit_code=EXIT_RUNTIME_ERROR,
            )

        fingerprint = _step_fingerprint(step=step, toi=toi, inputs=resolved_inputs, network_ok=network_ok)

        payload: dict[str, Any]
        attempt = 0
        started_at = time.time()
        skipped = False
        if resume and marker_path.exists() and output_path.exists():
            marker = _load_json(marker_path)
            if marker.get("input_fingerprint") == fingerprint and marker.get("status") == "ok":
                payload = _load_json(output_path)
                attempt = int(marker.get("attempt") or 1)
                skipped = True
            else:
                payload = {}
        else:
            payload = {}

        if not skipped:
            try:
                payload, attempt = _run_step_with_retries(
                    step=step,
                    toi=toi,
                    inputs=resolved_inputs,
                    output_path=output_path,
                    stderr_path=stderr_path,
                    network_ok=network_ok,
                    max_attempts=int(composition.defaults.get("retry_max_attempts", 3)),
                    initial_backoff_seconds=float(composition.defaults.get("retry_initial_seconds", 1.0)),
                )
            except Exception as exc:
                row = {
                    "step_id": step.id,
                    "op": step.op,
                    "status": "failed",
                    "error": str(exc),
                    "step_output_path": str(output_path),
                }
                step_rows.append(row)
                run_status = "partial" if (continue_on_error or step.on_error == "continue") else "failed"
                if not (continue_on_error or step.on_error == "continue"):
                    break
                continue

        payload["_step_output_path"] = str(output_path)
        step_outputs[step.id] = payload
        step_ports[step.id] = _apply_ports(step, step_output_path=output_path, payload=payload)

        concern_flags.update(_extract_concern_flags(payload))

        finished_at = time.time()
        marker = {
            "toi": toi,
            "step_id": step.id,
            "step_index": idx,
            "status": "ok",
            "started_at": started_at,
            "finished_at": finished_at,
            "attempt": attempt,
            "step_output_path": str(output_path),
            "input_fingerprint": fingerprint,
        }
        _write_json(marker_path, marker)

        step_rows.append(
            {
                "step_id": step.id,
                "op": step.op,
                "status": "ok",
                "attempt": attempt,
                "skipped_resume": skipped,
                "step_output_path": str(output_path),
                "verdict": payload.get("verdict"),
            }
        )

    result_payload = {
        "schema_version": "pipeline.result.v1",
        "toi": toi,
        "profile_id": composition.id,
        "composition_digest": composition_digest(composition),
        "status": run_status,
        "steps": step_rows,
        "verdict": None,
        "artifacts": {"steps_dir": str(steps_dir)},
        "provenance": {
            "network_ok": bool(network_ok),
            "continue_on_error": bool(continue_on_error),
            "resume": bool(resume),
            "timestamp": time.time(),
        },
        "concern_flags": sorted(concern_flags),
    }

    out_result_path = toi_dir / "pipeline_result.json"
    _write_json(out_result_path, result_payload)
    result_payload["_path"] = str(out_result_path)

    return result_payload


def _extract_evidence_row(toi_result: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    toi = str(toi_result.get("toi"))

    def _load_step_payloads() -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[tuple[str, str], dict[str, Any]],
    ]:
        payloads_by_step_id: dict[str, dict[str, Any]] = {}
        payloads_by_op: dict[str, dict[str, Any]] = {}
        payloads_by_step_id_and_op: dict[tuple[str, str], dict[str, Any]] = {}
        for row in toi_result.get("steps", []):
            if row.get("status") != "ok":
                continue
            step_id = str(row.get("step_id") or "")
            op = str(row.get("op") or "")
            path = row.get("step_output_path")
            if path and Path(path).exists():
                payload = _load_json(Path(path))
                if isinstance(payload, dict):
                    if step_id and step_id not in payloads_by_step_id:
                        payloads_by_step_id[step_id] = payload
                    if op and op not in payloads_by_op:
                        payloads_by_op[op] = payload
                    if step_id and op and (step_id, op) not in payloads_by_step_id_and_op:
                        payloads_by_step_id_and_op[(step_id, op)] = payload
        return payloads_by_step_id, payloads_by_op, payloads_by_step_id_and_op

    def _is_finite_number(value: Any) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return number == number and number not in {float("inf"), float("-inf")}

    def _to_finite_float(value: Any) -> float | None:
        if not _is_finite_number(value):
            return None
        return float(value)

    def _is_transit_like_verdict(verdict: Any) -> bool:
        if verdict is None:
            return False
        label = str(verdict).strip().upper()
        if label == "":
            return False
        return "TRANSIT" in label and "NON_TRANSIT" not in label

    def _maybe_load_step(
        op: str,
        *,
        payloads_by_op: dict[str, dict[str, Any]],
        payloads_by_step_id: dict[str, dict[str, Any]],
        payloads_by_step_id_and_op: dict[tuple[str, str], dict[str, Any]],
        step_id: str | None = None,
        allow_op_fallback: bool = True,
    ) -> dict[str, Any] | None:
        if step_id is not None:
            payload = payloads_by_step_id_and_op.get((step_id, op))
            if payload is not None:
                return payload
            payload = payloads_by_step_id.get(step_id)
            if payload is not None and allow_op_fallback:
                return payload
            if not allow_op_fallback:
                return None
        return payloads_by_op.get(op)

    def _extract_verdict(payload: dict[str, Any] | None) -> Any:
        if not isinstance(payload, dict):
            return None
        verdict = payload.get("verdict")
        if verdict is not None:
            return verdict
        result = payload.get("result")
        if isinstance(result, dict):
            return result.get("verdict")
        return None

    def _extract_reliability_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        summary = payload.get("reliability_summary")
        if isinstance(summary, dict):
            return summary
        result = payload.get("result")
        if isinstance(result, dict):
            nested = result.get("reliability_summary")
            if isinstance(nested, dict):
                return nested
        return None

    def _extract_value_from_payload(payload: dict[str, Any] | None, key: str) -> Any:
        if not isinstance(payload, dict):
            return None
        direct = payload.get(key)
        if direct is not None:
            return direct
        result = payload.get("result")
        if isinstance(result, dict):
            nested = result.get(key)
            if nested is not None:
                return nested
            summary = result.get("summary")
            if isinstance(summary, dict):
                nested_summary = summary.get(key)
                if nested_summary is not None:
                    return nested_summary
        summary = payload.get("summary")
        if isinstance(summary, dict):
            nested = summary.get(key)
            if nested is not None:
                return nested
        return None

    payloads_by_step_id, payloads_by_op, payloads_by_step_id_and_op = _load_step_payloads()

    model_compete = _maybe_load_step(
        "model_compete",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    systematics = _maybe_load_step(
        "systematics_proxy",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    ephemeris = _maybe_load_step(
        "ephemeris_reliability",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    timing = _maybe_load_step(
        "timing",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    localize_host = _maybe_load_step(
        "localize_host",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    vet_payload = _maybe_load_step(
        "vet",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    report_payload = _maybe_load_step(
        "report",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    dilution = _maybe_load_step(
        "dilution",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    fpp = _maybe_load_step(
        "fpp_run",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    if fpp is None:
        fpp = _maybe_load_step(
            "fpp",
            payloads_by_op=payloads_by_op,
            payloads_by_step_id=payloads_by_step_id,
            payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        )
    resolve_neighbors = _maybe_load_step(
        "resolve_neighbors",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    model_compete_raw = _maybe_load_step(
        "model_compete",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="model_compete_raw",
    )
    model_compete_detrended = _maybe_load_step(
        "model_compete",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="model_compete_detrended",
    )
    fpp_raw_payload = _maybe_load_step(
        "fpp_run",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="fpp_raw",
        allow_op_fallback=False,
    )
    contrast_curve_summary = _maybe_load_step(
        "contrast_curve_summary",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    contrast_curves = _maybe_load_step(
        "contrast_curves",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    contrast_payload = contrast_curve_summary if contrast_curve_summary is not None else contrast_curves
    fpp_detrended_payload = _maybe_load_step(
        "fpp_run",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="fpp_detrended",
        allow_op_fallback=False,
    )

    localize_action_hint = None
    if isinstance(localize_host, dict):
        result = localize_host.get("result")
        if isinstance(result, dict):
            consensus = result.get("consensus")
            if isinstance(consensus, dict):
                localize_action_hint = consensus.get("action_hint")
    localize_reliability_summary = _extract_reliability_summary(localize_host)
    localize_reliability_status = (
        localize_reliability_summary.get("status") if isinstance(localize_reliability_summary, dict) else None
    )
    localize_reliability_action_hint = (
        localize_reliability_summary.get("action_hint")
        if isinstance(localize_reliability_summary, dict)
        else None
    )

    dilution_n_plausible = None
    if isinstance(dilution, dict):
        dilution_n_plausible = dilution.get("n_plausible_scenarios")
        if dilution_n_plausible is None:
            result = dilution.get("result")
            if isinstance(result, dict):
                dilution_n_plausible = result.get("n_plausible_scenarios")
    dilution_reliability_summary = _extract_reliability_summary(dilution)
    dilution_reliability_status = (
        dilution_reliability_summary.get("status") if isinstance(dilution_reliability_summary, dict) else None
    )
    dilution_reliability_action_hint = (
        dilution_reliability_summary.get("action_hint")
        if isinstance(dilution_reliability_summary, dict)
        else None
    )

    multiplicity_risk_payload: dict[str, Any] | None = None
    if isinstance(resolve_neighbors, dict):
        direct = resolve_neighbors.get("multiplicity_risk")
        if isinstance(direct, dict):
            multiplicity_risk_payload = direct
        else:
            prov = resolve_neighbors.get("provenance")
            if isinstance(prov, dict):
                nested = prov.get("multiplicity_risk")
                if isinstance(nested, dict):
                    multiplicity_risk_payload = nested
    multiplicity_risk_status = (
        multiplicity_risk_payload.get("status") if isinstance(multiplicity_risk_payload, dict) else None
    )
    multiplicity_risk_reasons = None
    if isinstance(multiplicity_risk_payload, dict):
        reasons = multiplicity_risk_payload.get("reasons")
        if isinstance(reasons, list):
            multiplicity_risk_reasons = [str(x) for x in reasons if x is not None]

    fpp_value = None
    if isinstance(fpp, dict):
        fpp_value = fpp.get("fpp")
        if fpp_value is None:
            result = fpp.get("result")
            if isinstance(result, dict):
                fpp_value = result.get("fpp")

    fpp_raw = None
    if isinstance(fpp_raw_payload, dict):
        fpp_raw = fpp_raw_payload.get("fpp")
        if fpp_raw is None:
            result = fpp_raw_payload.get("result")
            if isinstance(result, dict):
                fpp_raw = result.get("fpp")

    fpp_detrended = None
    if isinstance(fpp_detrended_payload, dict):
        fpp_detrended = fpp_detrended_payload.get("fpp")
        if fpp_detrended is None:
            result = fpp_detrended_payload.get("result")
            if isinstance(result, dict):
                fpp_detrended = result.get("fpp")

    fpp_raw_value = _to_finite_float(fpp_raw)
    fpp_detrended_value = _to_finite_float(fpp_detrended)
    fpp_delta_detrended_minus_raw: float | None = None
    if fpp_raw_value is not None and fpp_detrended_value is not None:
        fpp_delta_detrended_minus_raw = fpp_detrended_value - fpp_raw_value

    model_compete_raw_verdict = _extract_verdict(model_compete_raw)
    model_compete_detrended_verdict = _extract_verdict(model_compete_detrended)
    robustness_recommended_variant = None
    robustness_present = any(
        payload is not None
        for payload in (model_compete_raw, model_compete_detrended, fpp_raw_payload, fpp_detrended_payload)
    )
    if robustness_present:
        robustness_recommended_variant = "raw"
        if (
            fpp_raw_value is not None
            and fpp_detrended_value is not None
            and fpp_detrended_value < fpp_raw_value
            and _is_transit_like_verdict(model_compete_detrended_verdict)
        ):
            robustness_recommended_variant = "detrended"

    detrend_invariance_policy_verdict = "NOT_EVALUATED"
    detrend_invariance_policy_reason_code = "ROBUSTNESS_INPUTS_ABSENT"
    detrend_invariance_policy_observed_fpp_delta_abs = None
    detrend_invariance_policy_observed_model_verdict_changed = None
    if robustness_present:
        detrend_invariance_policy_verdict = "INSUFFICIENT_DATA"
        detrend_invariance_policy_reason_code = "MISSING_MODEL_VERDICTS_OR_FPP_VALUES"
        raw_verdict_norm = (
            str(model_compete_raw_verdict).strip().upper() if model_compete_raw_verdict is not None else None
        )
        detrended_verdict_norm = (
            str(model_compete_detrended_verdict).strip().upper()
            if model_compete_detrended_verdict is not None
            else None
        )
        if fpp_delta_detrended_minus_raw is not None:
            detrend_invariance_policy_observed_fpp_delta_abs = abs(fpp_delta_detrended_minus_raw)
        if raw_verdict_norm is not None and detrended_verdict_norm is not None:
            detrend_invariance_policy_observed_model_verdict_changed = raw_verdict_norm != detrended_verdict_norm
        if (
            detrend_invariance_policy_observed_model_verdict_changed is not None
            and detrend_invariance_policy_observed_fpp_delta_abs is not None
        ):
            if detrend_invariance_policy_observed_model_verdict_changed:
                detrend_invariance_policy_verdict = "NON_INVARIANT"
                detrend_invariance_policy_reason_code = "MODEL_VERDICT_CHANGED"
            elif detrend_invariance_policy_observed_fpp_delta_abs > _DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD:
                detrend_invariance_policy_verdict = "NON_INVARIANT"
                detrend_invariance_policy_reason_code = "FPP_DELTA_ABOVE_THRESHOLD"
            else:
                detrend_invariance_policy_verdict = "INVARIANT"
                detrend_invariance_policy_reason_code = "PASS"

    contrast_curve_availability = _extract_value_from_payload(contrast_payload, "availability")
    contrast_curve_n_observations = _extract_value_from_payload(contrast_payload, "n_observations")
    contrast_curve_filter = _extract_value_from_payload(contrast_payload, "filter")
    contrast_curve_quality = _extract_value_from_payload(contrast_payload, "quality")
    contrast_curve_depth0p5 = _extract_value_from_payload(contrast_payload, "depth0p5")
    contrast_curve_depth1p0 = _extract_value_from_payload(contrast_payload, "depth1p0")

    selected_curve = _extract_value_from_payload(contrast_payload, "selected_curve")
    if not isinstance(selected_curve, dict):
        selected_curve = {}
    contrast_curve_selected_id = selected_curve.get("id")
    contrast_curve_selected_source = selected_curve.get("source")
    contrast_curve_selected_filter = selected_curve.get("filter")
    contrast_curve_selected_quality = selected_curve.get("quality")
    contrast_curve_selected_depth0p5 = selected_curve.get("depth0p5")
    contrast_curve_selected_depth1p0 = selected_curve.get("depth1p0")
    contrast_curve_selected_metadata = selected_curve if selected_curve else None
    known_planet_status = _extract_value_from_payload(vet_payload, "known_planet_match_status")
    known_planet_payload = _extract_value_from_payload(vet_payload, "known_planet_match")
    if not isinstance(known_planet_payload, dict):
        known_planet_payload = {}
    matched_planet = known_planet_payload.get("matched_planet")
    if not isinstance(matched_planet, dict):
        matched_planet = {}
    known_planet_name = matched_planet.get("name")
    known_planet_period = matched_planet.get("period")

    stellar_contamination_risk_scalar = _extract_value_from_payload(
        report_payload,
        "stellar_contamination_risk_scalar",
    )
    if stellar_contamination_risk_scalar is None:
        stellar_contamination_risk_scalar = _extract_value_from_payload(
            vet_payload,
            "stellar_contamination_risk_scalar",
        )
    if stellar_contamination_risk_scalar is None:
        contamination_summary = _extract_value_from_payload(
            report_payload,
            "stellar_contamination_summary",
        )
        if not isinstance(contamination_summary, dict):
            contamination_summary = _extract_value_from_payload(
                vet_payload,
                "stellar_contamination_summary",
            )
        if isinstance(contamination_summary, dict):
            stellar_contamination_risk_scalar = contamination_summary.get("risk_scalar")

    concern_flags = {str(x) for x in (toi_result.get("concern_flags") or []) if x is not None}
    for payload in payloads_by_step_id.values():
        concern_flags.update(_extract_concern_flags(payload))
    for payload in payloads_by_op.values():
        concern_flags.update(_extract_concern_flags(payload))

    row = {
        "toi": toi,
        "model_compete_verdict": _extract_verdict(model_compete),
        "model_compete_raw_verdict": model_compete_raw_verdict,
        "model_compete_detrended_verdict": model_compete_detrended_verdict,
        "systematics_verdict": _extract_verdict(systematics),
        "ephemeris_verdict": _extract_verdict(ephemeris),
        "timing_verdict": _extract_verdict(timing),
        "localize_host_action_hint": localize_action_hint,
        "localize_host_reliability_status": localize_reliability_status,
        "localize_host_reliability_action_hint": localize_reliability_action_hint,
        "dilution_n_plausible_scenarios": dilution_n_plausible,
        "dilution_reliability_status": dilution_reliability_status,
        "dilution_reliability_action_hint": dilution_reliability_action_hint,
        "multiplicity_risk_status": multiplicity_risk_status,
        "multiplicity_risk_reasons": multiplicity_risk_reasons,
        "fpp": fpp_value,
        "fpp_raw": fpp_raw,
        "fpp_detrended": fpp_detrended,
        "fpp_delta_detrended_minus_raw": fpp_delta_detrended_minus_raw,
        "robustness_recommended_variant": robustness_recommended_variant,
        "detrend_invariance_policy_version": _DETREND_INVARIANCE_POLICY_VERSION,
        "detrend_invariance_policy_verdict": detrend_invariance_policy_verdict,
        "detrend_invariance_policy_reason_code": detrend_invariance_policy_reason_code,
        "detrend_invariance_policy_fpp_delta_abs_threshold": _DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD,
        "detrend_invariance_policy_observed_fpp_delta_abs": detrend_invariance_policy_observed_fpp_delta_abs,
        "detrend_invariance_policy_observed_model_verdict_changed": detrend_invariance_policy_observed_model_verdict_changed,
        "contrast_curve_availability": contrast_curve_availability,
        "contrast_curve_n_observations": contrast_curve_n_observations,
        "contrast_curve_filter": contrast_curve_filter,
        "contrast_curve_quality": contrast_curve_quality,
        "contrast_curve_depth0p5": contrast_curve_depth0p5,
        "contrast_curve_depth1p0": contrast_curve_depth1p0,
        "contrast_curve_selected_id": contrast_curve_selected_id,
        "contrast_curve_selected_source": contrast_curve_selected_source,
        "contrast_curve_selected_filter": contrast_curve_selected_filter,
        "contrast_curve_selected_quality": contrast_curve_selected_quality,
        "contrast_curve_selected_depth0p5": contrast_curve_selected_depth0p5,
        "contrast_curve_selected_depth1p0": contrast_curve_selected_depth1p0,
        "contrast_curve_selected_metadata": contrast_curve_selected_metadata,
        "known_planet_status": known_planet_status,
        "known_planet_name": known_planet_name,
        "known_planet_period": known_planet_period,
        "stellar_contamination_risk_scalar": stellar_contamination_risk_scalar,
        "concern_flags": sorted(concern_flags),
    }
    return row


def _write_evidence_table(*, out_dir: Path, toi_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_extract_evidence_row(item, out_dir=out_dir) for item in toi_results]

    json_path = out_dir / "evidence_table.json"
    _write_json(json_path, {"schema_version": "pipeline.evidence_table.v5", "rows": rows})

    csv_path = out_dir / "evidence_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "toi",
        "model_compete_verdict",
        "model_compete_raw_verdict",
        "model_compete_detrended_verdict",
        "systematics_verdict",
        "ephemeris_verdict",
        "timing_verdict",
        "localize_host_action_hint",
        "localize_host_reliability_status",
        "localize_host_reliability_action_hint",
        "dilution_n_plausible_scenarios",
        "dilution_reliability_status",
        "dilution_reliability_action_hint",
        "multiplicity_risk_status",
        "multiplicity_risk_reasons",
        "fpp",
        "fpp_raw",
        "fpp_detrended",
        "fpp_delta_detrended_minus_raw",
        "robustness_recommended_variant",
        "detrend_invariance_policy_version",
        "detrend_invariance_policy_verdict",
        "detrend_invariance_policy_reason_code",
        "detrend_invariance_policy_fpp_delta_abs_threshold",
        "detrend_invariance_policy_observed_fpp_delta_abs",
        "detrend_invariance_policy_observed_model_verdict_changed",
        "contrast_curve_availability",
        "contrast_curve_n_observations",
        "contrast_curve_filter",
        "contrast_curve_quality",
        "contrast_curve_depth0p5",
        "contrast_curve_depth1p0",
        "contrast_curve_selected_id",
        "contrast_curve_selected_source",
        "contrast_curve_selected_filter",
        "contrast_curve_selected_quality",
        "contrast_curve_selected_depth0p5",
        "contrast_curve_selected_depth1p0",
        "contrast_curve_selected_metadata",
        "known_planet_status",
        "known_planet_name",
        "known_planet_period",
        "stellar_contamination_risk_scalar",
        "concern_flags",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flags = sorted(str(x) for x in (row.get("concern_flags") or []) if x is not None)
            risk_reasons = row.get("multiplicity_risk_reasons")
            risk_reasons_text = (
                ";".join(str(x) for x in risk_reasons if x is not None)
                if isinstance(risk_reasons, list)
                else ""
            )
            writer.writerow(
                {
                    **row,
                    "concern_flags": ";".join(flags),
                    "multiplicity_risk_reasons": risk_reasons_text,
                }
            )

    return rows


def run_composition(
    *,
    composition: CompositionSpec,
    tois: list[str],
    out_dir: Path,
    network_ok: bool,
    continue_on_error: bool,
    max_workers: int,
    resume: bool,
) -> dict[str, Any]:
    if not tois:
        raise BtvCliError("At least one --toi is required.", exit_code=EXIT_RUNTIME_ERROR)

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    toi_results: list[dict[str, Any]] = []
    max_workers_eff = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=max_workers_eff) as pool:
        future_map = {
            pool.submit(
                _run_one_toi,
                composition=composition,
                toi=toi,
                out_dir=out_dir,
                network_ok=network_ok,
                continue_on_error=continue_on_error,
                resume=resume,
            ): toi
            for toi in tois
        }
        for fut in as_completed(future_map):
            toi = future_map[fut]
            try:
                toi_results.append(fut.result())
            except Exception as exc:  # pragma: no cover
                toi_results.append(
                    {
                        "toi": toi,
                        "status": "failed",
                        "steps": [],
                        "error": str(exc),
                        "concern_flags": [],
                    }
                )

    toi_results_sorted = sorted(toi_results, key=lambda row: str(row.get("toi") or ""))
    evidence_rows = _write_evidence_table(out_dir=out_dir, toi_results=toi_results_sorted)

    n_ok = sum(1 for row in toi_results_sorted if row.get("status") == "ok")
    n_partial = sum(1 for row in toi_results_sorted if row.get("status") == "partial")
    n_failed = sum(1 for row in toi_results_sorted if row.get("status") == "failed")

    manifest = {
        "schema_version": "pipeline.run_manifest.v1",
        "profile_id": composition.id,
        "composition_digest": composition_digest(composition),
        "tois": list(tois),
        "counts": {
            "n_tois": len(tois),
            "n_ok": n_ok,
            "n_partial": n_partial,
            "n_failed": n_failed,
        },
        "elapsed_seconds": time.time() - start,
        "options": {
            "network_ok": bool(network_ok),
            "continue_on_error": bool(continue_on_error),
            "max_workers": max_workers_eff,
            "resume": bool(resume),
        },
        "results": [
            {
                "toi": row.get("toi"),
                "status": row.get("status"),
                "result_path": row.get("_path"),
            }
            for row in toi_results_sorted
        ],
    }
    _write_json(out_dir / "run_manifest.json", manifest)

    return {
        "manifest": manifest,
        "toi_results": toi_results_sorted,
        "evidence_rows": evidence_rows,
    }


__all__ = ["run_composition"]
