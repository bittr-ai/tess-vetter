"""Executor for composable workflow pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bittr_tess_vetter.cli.common_cli import EXIT_RUNTIME_ERROR, BtvCliError
from bittr_tess_vetter.pipeline_composition.ref_resolver import extract_path, resolve_value
from bittr_tess_vetter.pipeline_composition.schema import CompositionSpec, StepSpec, composition_digest

_OP_TO_COMMAND = {
    "vet": "vet",
    "measure_sectors": "measure-sectors",
    "fit": "fit",
    "report": "report",
    "activity": "activity",
    "model_compete": "model-compete",
    "timing": "timing",
    "systematics_proxy": "systematics-proxy",
    "ephemeris_reliability": "ephemeris-reliability",
    "resolve_stellar": "resolve-stellar",
    "resolve_neighbors": "resolve-neighbors",
    "localize_host": "localize-host",
    "dilution": "dilution",
    "detrend_grid": "detrend-grid",
    "fpp": "fpp",
}

_RETRYABLE_TOKENS = ("429", "timeout", "timed out", "temporarily unavailable", "connection reset")


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
    args = [sys.executable, "-m", "bittr_tess_vetter.cli.enrich_cli", command]

    args.extend(["--toi", str(toi)])
    args.append("--network-ok" if network_ok else "--no-network")

    raw_args = inputs.get("args")
    if raw_args is not None and not isinstance(raw_args, list):
        raise BtvCliError(
            f"Step '{step.id}' inputs.args must be a list of CLI tokens.",
            exit_code=EXIT_RUNTIME_ERROR,
        )

    for key, value in inputs.items():
        if key in {"args"}:
            continue
        flag = _flag_name(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                args.append("--no-" + key.replace("_", "-"))
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])

    if raw_args:
        args.extend(str(token) for token in raw_args)

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
            time.sleep(max(0.1, float(sleep_s)))

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

        resolved_inputs = resolve_value(
            {**composition.defaults, **step.inputs},
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
    toi_dir = out_dir / toi

    def _maybe_load_step(op: str) -> dict[str, Any] | None:
        for row in toi_result.get("steps", []):
            if row.get("op") == op and row.get("status") == "ok":
                path = row.get("step_output_path")
                if path and Path(path).exists():
                    return _load_json(Path(path))
        return None

    model_compete = _maybe_load_step("model_compete")
    systematics = _maybe_load_step("systematics_proxy")
    ephemeris = _maybe_load_step("ephemeris_reliability")
    timing = _maybe_load_step("timing")
    localize_host = _maybe_load_step("localize_host")
    dilution = _maybe_load_step("dilution")
    fpp = _maybe_load_step("fpp")

    localize_action_hint = None
    if isinstance(localize_host, dict):
        result = localize_host.get("result")
        if isinstance(result, dict):
            consensus = result.get("consensus")
            if isinstance(consensus, dict):
                localize_action_hint = consensus.get("action_hint")

    dilution_n_plausible = None
    if isinstance(dilution, dict):
        dilution_n_plausible = dilution.get("n_plausible_scenarios")
        if dilution_n_plausible is None:
            result = dilution.get("result")
            if isinstance(result, dict):
                dilution_n_plausible = result.get("n_plausible_scenarios")

    fpp_value = None
    if isinstance(fpp, dict):
        fpp_value = fpp.get("fpp")

    row = {
        "toi": toi,
        "model_compete_verdict": model_compete.get("verdict") if isinstance(model_compete, dict) else None,
        "systematics_verdict": systematics.get("verdict") if isinstance(systematics, dict) else None,
        "ephemeris_verdict": ephemeris.get("verdict") if isinstance(ephemeris, dict) else None,
        "timing_verdict": timing.get("verdict") if isinstance(timing, dict) else None,
        "localize_host_action_hint": localize_action_hint,
        "dilution_n_plausible_scenarios": dilution_n_plausible,
        "fpp": fpp_value,
        "concern_flags": list(toi_result.get("concern_flags") or []),
    }
    return row


def _write_evidence_table(*, out_dir: Path, toi_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_extract_evidence_row(item, out_dir=out_dir) for item in toi_results]

    json_path = out_dir / "evidence_table.json"
    _write_json(json_path, {"schema_version": "pipeline.evidence_table.v1", "rows": rows})

    csv_path = out_dir / "evidence_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "toi",
        "model_compete_verdict",
        "systematics_verdict",
        "ephemeris_verdict",
        "timing_verdict",
        "localize_host_action_hint",
        "dilution_n_plausible_scenarios",
        "fpp",
        "concern_flags",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "concern_flags": ";".join(row.get("concern_flags", []))})

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
