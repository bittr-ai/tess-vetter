from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tess_vetter.api.periodogram import run_periodogram
from tess_vetter.api.pipeline import describe_checks, list_checks
from tess_vetter.api.typed_wrappers import (
    DescribeChecksModel,
    ListChecksOutputModel,
    RunPeriodogramOutputModel,
    describe_checks_typed,
    list_checks_typed,
    run_periodogram_typed,
)
from tess_vetter.validation.registry import CheckRegistry, CheckRequirements, CheckTier


class _StubCheck:
    def __init__(self, check_id: str, name: str) -> None:
        self._id = check_id
        self._name = name

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return CheckTier.LC_ONLY

    @property
    def requirements(self) -> CheckRequirements:
        return CheckRequirements()

    @property
    def citations(self) -> list[str]:
        return ["Stub Citation"]


def test_list_checks_typed_matches_existing_output() -> None:
    registry = CheckRegistry()
    registry.register(_StubCheck("V01", "Odd-even"))

    typed = list_checks_typed(registry)
    raw = list_checks(registry)

    assert isinstance(typed, ListChecksOutputModel)
    assert [item.model_dump(mode="python") for item in typed.checks] == raw


def test_describe_checks_typed_matches_existing_output() -> None:
    registry = CheckRegistry()
    registry.register(_StubCheck("V01", "Odd-even"))

    typed = describe_checks_typed(registry)
    raw = describe_checks(registry)

    assert isinstance(typed, DescribeChecksModel)
    assert typed.description == raw


def test_run_periodogram_typed_matches_existing_output_for_ls() -> None:
    time = np.linspace(1500.0, 1527.0, 1200, dtype=np.float64)
    flux = 1.0 + 2e-4 * np.sin(2.0 * np.pi * time / 3.1)
    flux_err = np.full_like(time, 1e-4)

    typed = run_periodogram_typed(
        time=time.tolist(),
        flux=flux.tolist(),
        flux_err=flux_err.tolist(),
        method="ls",
        min_period=1.0,
        max_period=6.0,
        data_ref="lc:test:typed",
    )
    raw = run_periodogram(
        time=time,
        flux=flux,
        flux_err=flux_err,
        method="ls",
        min_period=1.0,
        max_period=6.0,
        data_ref="lc:test:typed",
    )

    assert isinstance(typed, RunPeriodogramOutputModel)
    assert typed.result == raw


def test_run_periodogram_typed_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        run_periodogram_typed(
            time=[1.0, 2.0, 3.0],
            flux=[1.0, 1.0, 1.0],
            method="ls",
            extra_field_not_allowed=True,
        )
