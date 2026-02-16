from __future__ import annotations

from pathlib import Path

import pytest

from bittr_tess_vetter.pipeline_composition.executor import _build_cli_args, _run_step_with_retries
from bittr_tess_vetter.pipeline_composition.schema import StepSpec


def _step() -> StepSpec:
    return StepSpec(
        id="arg-step",
        op="vet",
        inputs={},
        ports={},
        outputs={},
        on_error="fail",
    )


def test_build_cli_args_omits_false_bool_and_supports_explicit_paired_bool_form(tmp_path: Path) -> None:
    args = _build_cli_args(
        step=_step(),
        toi="TOI-123.01",
        inputs={
            "include_lc_summary": False,
            "require_coordinates": True,
            "download_toggle": {
                "_value": False,
                "_flag_true": "--download",
                "_flag_false": "--no-download",
            },
        },
        output_path=tmp_path / "out.json",
        network_ok=False,
    )

    assert "--no-network" in args
    assert "--network-ok" not in args
    assert "--require-coordinates" in args
    assert "--include-lc-summary" not in args
    assert "--no-include-lc-summary" not in args
    assert "--no-download" in args
    assert "--download" not in args


def test_build_cli_args_supports_raw_flags_and_args_passthrough(tmp_path: Path) -> None:
    args = _build_cli_args(
        step=_step(),
        toi="TOI-123.01",
        inputs={
            "preset": "fast",
            "_flags": ["--foo", "--bar"],
            "_args": ["--x", "1", "--y", "2"],
        },
        output_path=tmp_path / "out.json",
        network_ok=True,
    )

    assert "--network-ok" in args
    assert "--no-network" not in args
    assert "--preset" in args
    assert "fast" in args
    assert "--foo" in args
    assert "--bar" in args
    assert "--x" in args
    assert "1" in args
    assert "--y" in args
    assert "2" in args


def test_build_cli_args_does_not_require_or_emit_retry_flags_from_executor_defaults(tmp_path: Path) -> None:
    args = _build_cli_args(
        step=_step(),
        toi="TOI-123.01",
        inputs={
            "preset": "fast",
        },
        output_path=tmp_path / "out.json",
        network_ok=True,
    )

    assert "--retry-max-attempts" not in args
    assert "--retry-initial-seconds" not in args


def test_run_step_with_retries_uses_backoff_jitter_for_retryable_errors(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def _fake_run_step_command(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("timed out while fetching")
        return {"ok": True}

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor._run_step_command",
        _fake_run_step_command,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.random.uniform",
        lambda a, b: 0.01,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.time.sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    payload, used_attempts = _run_step_with_retries(
        step=_step(),
        toi="TOI-123.01",
        inputs={},
        output_path=tmp_path / "out.json",
        stderr_path=tmp_path / "stderr.log",
        network_ok=True,
        max_attempts=3,
        initial_backoff_seconds=0.2,
    )

    assert payload == {"ok": True}
    assert used_attempts == 3
    assert sleep_calls == pytest.approx([0.21, 0.41])
