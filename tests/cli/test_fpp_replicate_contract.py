from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from tess_vetter.platform.network.timeout import NetworkTimeoutError
from tess_vetter.validation.triceratops_fpp import calculate_fpp_handler


class _MockProbs:
    def __init__(self, scenarios: list[str], probs: list[float]) -> None:
        self.columns = ["scenario", "prob"]
        self._scenarios = scenarios
        self._probs = probs

    def __getitem__(self, key: str) -> list[Any]:
        if key == "scenario":
            return self._scenarios
        if key == "prob":
            return self._probs
        raise KeyError(key)


@dataclass
class _MockTarget:
    trilegal_fname: str | None = "/tmp/mock_trilegal.csv"
    trilegal_url: str | None = None
    stars: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.stars = [{"Tmag": 10.0}, {"Tmag": 12.0}]
        self._calls = 0
        self.FPP = 0.02
        self.NFPP = 0.002
        self.probs: Any = _MockProbs(
            scenarios=["TP", "EB", "BEB", "NEB", "NTP"],
            probs=[0.99, 0.005, 0.003, 0.001, 0.001],
        )

    def calc_depths(self, tdepth: float) -> None:  # noqa: ARG002
        return

    def calc_probs(self, **kwargs: Any) -> None:  # noqa: ARG002
        self._calls += 1
        if self._calls == 1:
            # First replicate is degenerate; second is valid.
            self.FPP = float("nan")
            self.NFPP = 0.2
            self.probs = _MockProbs(
                scenarios=["TP", "EB", "BEB"],
                probs=[float("nan"), float("nan"), float("nan")],
            )
            return

        self.FPP = 0.02
        self.NFPP = 0.002
        self.probs = _MockProbs(
            scenarios=["TP", "EB", "BEB", "NEB", "NTP"],
            probs=[0.99, 0.005, 0.003, 0.001, 0.001],
        )


@dataclass
class _MockTargetTimeoutThenOk:
    trilegal_fname: str | None = "/tmp/mock_trilegal.csv"
    trilegal_url: str | None = None
    stars: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.stars = [{"Tmag": 10.0}, {"Tmag": 12.0}]
        self._calls = 0
        self.FPP = 0.02
        self.NFPP = 0.002
        self.probs = _MockProbs(
            scenarios=["TP", "EB", "BEB", "NEB", "NTP"],
            probs=[0.99, 0.005, 0.003, 0.001, 0.001],
        )

    def calc_depths(self, tdepth: float) -> None:  # noqa: ARG002
        return

    def calc_probs(self, **kwargs: Any) -> None:  # noqa: ARG002
        self._calls += 1
        if self._calls == 1:
            raise NetworkTimeoutError("replicate timed out", 1.0)
        self.FPP = 0.02
        self.NFPP = 0.002
        self.probs = _MockProbs(
            scenarios=["TP", "EB", "BEB", "NEB", "NTP"],
            probs=[0.99, 0.005, 0.003, 0.001, 0.001],
        )


def test_replicate_analysis_default_full_runs_and_compat_fields() -> None:
    cache = SimpleNamespace(cache_dir="/tmp/test_cache")

    time = np.linspace(0.0, 27.0, 1200, dtype=np.float64)
    lc = SimpleNamespace(
        time=time,
        flux=np.ones_like(time, dtype=np.float64),
        flux_err=np.full_like(time, 1e-3, dtype=np.float64),
        valid_mask=np.ones(time.shape, dtype=bool),
    )

    cache.keys = lambda: ["lc:12345:1:pdcsap"]
    cache.get = lambda key: lc if str(key).startswith("lc:12345") else None

    target = _MockTarget()
    fake_vendor_module = SimpleNamespace(triceratops=SimpleNamespace(target=lambda **_: target))
    with patch.dict(
        sys.modules,
        {"tess_vetter.ext.triceratops_plus_vendor.triceratops": fake_vendor_module},
    ), patch(
        "tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target",
        return_value=target,
    ), patch(
        "tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target",
        return_value=None,
    ):
        out = calculate_fpp_handler(
            cache=cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500.0,
            duration_hours=3.0,
            replicates=2,
            seed=77,
            max_points=8000,
            mc_draws=250000,
        )

    assert "error" not in out

    # Compatibility top-level fields remain.
    assert out["replicates"] == 2
    assert out["n_success"] == 1
    assert out["n_fail"] == 1
    assert out["replicate_success_rate"] == 0.5

    # First-class replicate payload exists with default full run details.
    replicate_analysis = out["replicate_analysis"]
    assert replicate_analysis["summary"]["requested_replicates"] == 2
    assert replicate_analysis["summary"]["attempted_replicates"] == 2
    assert len(replicate_analysis["runs"]) == 2
    assert "errors" in replicate_analysis

    statuses = {run["status"] for run in replicate_analysis["runs"]}
    assert "degenerate" in statuses
    assert "ok" in statuses

    for run in replicate_analysis["runs"]:
        assert "requested_config" in run
        assert "effective_config" in run
        assert isinstance(run.get("effective_config_hash"), str)
        assert len(run["effective_config_hash"]) == 64
        assert run["requested_config"]["max_points"] == 8000
        assert "max_points" in run["effective_config"]
        assert isinstance(run.get("fallback_trace"), list)
        assert len(run["fallback_trace"]) >= 1
        assert run["fallback_trace"][0]["attempt"] == 1


def test_aggregate_counts_include_timeout_errors_in_top_level_fields() -> None:
    cache = SimpleNamespace(cache_dir="/tmp/test_cache")

    time = np.linspace(0.0, 27.0, 1200, dtype=np.float64)
    lc = SimpleNamespace(
        time=time,
        flux=np.ones_like(time, dtype=np.float64),
        flux_err=np.full_like(time, 1e-3, dtype=np.float64),
        valid_mask=np.ones(time.shape, dtype=bool),
    )

    cache.keys = lambda: ["lc:12345:1:pdcsap"]
    cache.get = lambda key: lc if str(key).startswith("lc:12345") else None

    target = _MockTargetTimeoutThenOk()
    fake_vendor_module = SimpleNamespace(triceratops=SimpleNamespace(target=lambda **_: target))
    with patch.dict(
        sys.modules,
        {"tess_vetter.ext.triceratops_plus_vendor.triceratops": fake_vendor_module},
    ), patch(
        "tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target",
        return_value=target,
    ), patch(
        "tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target",
        return_value=None,
    ):
        out = calculate_fpp_handler(
            cache=cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500.0,
            duration_hours=3.0,
            replicates=2,
            seed=91,
        )

    assert "error" not in out
    assert out["replicates"] == 2
    assert out["n_success"] == 1
    assert out["n_fail"] == 1
    assert out["replicate_success_rate"] == 0.5
