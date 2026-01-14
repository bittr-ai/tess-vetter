"""CLI smoke tests for MLX-based command-line modules.

These tests verify that CLI modules can be imported and that --help invocations
produce expected output without errors. MLX functionality requires Apple Silicon
hardware, so actual execution tests are marked to skip unless MLX is available.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

# Check if MLX is available
MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None
TLS_AVAILABLE = importlib.util.find_spec("transitleastsquares") is not None

# List of CLI modules to test
CLI_MODULES = [
    "bittr_tess_vetter.cli.mlx_bls_search_cli",
    "bittr_tess_vetter.cli.mlx_bls_search_range_cli",
    "bittr_tess_vetter.cli.mlx_quick_vet_cli",
    "bittr_tess_vetter.cli.mlx_refine_candidates_cli",
    "bittr_tess_vetter.cli.mlx_tls_calibration_cli",
]


# =============================================================================
# Import Smoke Tests
# =============================================================================


class TestCLIImports:
    """Test that CLI modules can be imported successfully."""

    @pytest.mark.parametrize("module_name", CLI_MODULES)
    def test_cli_module_imports(self, module_name: str) -> None:
        """Verify CLI module can be imported without errors."""
        module: ModuleType = importlib.import_module(module_name)
        assert module is not None
        # Verify the module has a main function
        assert hasattr(module, "main"), f"{module_name} should have a main() function"


# =============================================================================
# Help Invocation Tests
# =============================================================================


class TestCLIHelp:
    """Test that CLI modules respond correctly to --help."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "bittr_tess_vetter.cli.mlx_bls_search_cli",
            "bittr_tess_vetter.cli.mlx_bls_search_range_cli",
            "bittr_tess_vetter.cli.mlx_quick_vet_cli",
            "bittr_tess_vetter.cli.mlx_refine_candidates_cli",
        ],
    )
    def test_cli_help_invocation(self, module_name: str) -> None:
        """Verify --help produces output and exits cleanly."""
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # argparse returns exit code 0 for --help
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        # Help output should contain usage information
        assert "usage:" in result.stdout.lower() or module_name in result.stdout

    @pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not installed")
    def test_tls_calibration_cli_help(self) -> None:
        """Verify mlx_tls_calibration_cli --help works when TLS is available."""
        result = subprocess.run(
            [sys.executable, "-m", "bittr_tess_vetter.cli.mlx_tls_calibration_cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "usage:" in result.stdout.lower() or "tls" in result.stdout.lower()


# =============================================================================
# Module Entry Point Tests
# =============================================================================


class TestCLIEntryPoints:
    """Test that CLI modules can be invoked as entry points."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "bittr_tess_vetter.cli.mlx_bls_search_cli",
            "bittr_tess_vetter.cli.mlx_bls_search_range_cli",
            "bittr_tess_vetter.cli.mlx_quick_vet_cli",
            "bittr_tess_vetter.cli.mlx_refine_candidates_cli",
        ],
    )
    def test_cli_module_runnable(self, module_name: str) -> None:
        """Verify module can be run with python -m and exits with expected behavior."""
        # Running without arguments should produce an error or help message
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # argparse typically exits with code 2 when required args are missing
        # Some modules may output JSON error instead
        assert result.returncode in (0, 2), (
            f"Unexpected exit: {result.returncode}, stderr: {result.stderr}"
        )

    @pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not installed")
    def test_tls_calibration_cli_runnable(self) -> None:
        """Verify TLS calibration CLI can be invoked."""
        result = subprocess.run(
            [sys.executable, "-m", "bittr_tess_vetter.cli.mlx_tls_calibration_cli"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 2), (
            f"Unexpected exit: {result.returncode}, stderr: {result.stderr}"
        )


# =============================================================================
# Module Content Verification
# =============================================================================


class TestCLIModuleContent:
    """Test that CLI modules have expected structure and attributes."""

    def test_mlx_bls_search_cli_has_result_dataclass(self) -> None:
        """mlx_bls_search_cli should define MlxBlsLikeResult."""
        from bittr_tess_vetter.cli import mlx_bls_search_cli

        assert hasattr(mlx_bls_search_cli, "MlxBlsLikeResult")
        assert hasattr(mlx_bls_search_cli, "MlxBlsLikeCandidate")

    def test_mlx_bls_search_range_cli_has_request_dataclass(self) -> None:
        """mlx_bls_search_range_cli should define RangeSearchRequest."""
        from bittr_tess_vetter.cli import mlx_bls_search_range_cli

        assert hasattr(mlx_bls_search_range_cli, "RangeSearchRequest")

    def test_mlx_refine_candidates_cli_has_candidate_dataclass(self) -> None:
        """mlx_refine_candidates_cli should define Candidate."""
        from bittr_tess_vetter.cli import mlx_refine_candidates_cli

        assert hasattr(mlx_refine_candidates_cli, "Candidate")


# =============================================================================
# MLX-Specific Tests (conditionally run)
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available (requires Apple Silicon)")
class TestMLXAvailability:
    """Tests that run only when MLX is available."""

    def test_mlx_import_helper_succeeds(self) -> None:
        """_try_import_mlx should succeed when MLX is available."""
        from bittr_tess_vetter.cli.mlx_bls_search_cli import _try_import_mlx

        mx = _try_import_mlx()
        assert mx is not None


@pytest.mark.skipif(MLX_AVAILABLE, reason="Test for when MLX is NOT available")
class TestMLXUnavailable:
    """Tests that run only when MLX is NOT available."""

    def test_mlx_import_helper_raises(self) -> None:
        """_try_import_mlx should raise ImportError when MLX unavailable."""
        from bittr_tess_vetter.cli.mlx_bls_search_cli import _try_import_mlx

        with pytest.raises(ImportError, match="MLX is not installed"):
            _try_import_mlx()
