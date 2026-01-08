"""Tests for external light curve file conversion in TRICERATOPS+ integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bittr_tess_vetter.api.fpp import ExternalLightCurve
from bittr_tess_vetter.validation.triceratops_fpp import _write_external_lc_files


class TestWriteExternalLcFiles:
    """Tests for _write_external_lc_files helper."""

    def test_single_lc_creates_file_with_correct_format(self) -> None:
        """Verify temp file has 3 columns, no header, correct values."""
        lc = ExternalLightCurve(
            time_from_midtransit_days=np.array([-0.1, 0.0, 0.1]),
            flux=np.array([1.0, 0.99, 1.0]),
            flux_err=np.array([0.001, 0.001, 0.001]),
            filter="r",
        )

        with tempfile.TemporaryDirectory() as td:
            paths, filters = _write_external_lc_files([lc], Path(td))

            assert len(paths) == 1
            assert filters == ["r"]
            assert Path(paths[0]).exists()

            # Verify file contents
            data = np.loadtxt(paths[0])
            assert data.shape == (3, 3)
            np.testing.assert_array_almost_equal(data[:, 0], [-0.1, 0.0, 0.1])
            np.testing.assert_array_almost_equal(data[:, 1], [1.0, 0.99, 1.0])
            np.testing.assert_array_almost_equal(data[:, 2], [0.001, 0.001, 0.001])

    def test_multiple_lcs_creates_multiple_files(self) -> None:
        """Multiple ExternalLightCurves create separate files."""
        lc_r = ExternalLightCurve(
            time_from_midtransit_days=np.array([-0.1, 0.0, 0.1]),
            flux=np.array([1.0, 0.99, 1.0]),
            flux_err=np.array([0.001, 0.001, 0.001]),
            filter="r",
        )
        lc_g = ExternalLightCurve(
            time_from_midtransit_days=np.array([-0.05, 0.0, 0.05]),
            flux=np.array([1.0, 0.985, 1.0]),
            flux_err=np.array([0.002, 0.002, 0.002]),
            filter="g",
        )

        with tempfile.TemporaryDirectory() as td:
            paths, filters = _write_external_lc_files([lc_r, lc_g], Path(td))

            assert len(paths) == 2
            assert filters == ["r", "g"]
            assert all(Path(p).exists() for p in paths)

    def test_empty_list_returns_empty(self) -> None:
        """Empty input returns empty lists."""
        with tempfile.TemporaryDirectory() as td:
            paths, filters = _write_external_lc_files([], Path(td))
            assert paths == []
            assert filters == []

    def test_array_length_mismatch_raises_valueerror(self) -> None:
        """Mismatched array lengths raise ValueError."""
        lc = ExternalLightCurve(
            time_from_midtransit_days=np.array([-0.1, 0.0]),  # 2 elements
            flux=np.array([1.0, 0.99, 1.0]),  # 3 elements - mismatch!
            flux_err=np.array([0.001, 0.001, 0.001]),
            filter="r",
        )

        with (
            tempfile.TemporaryDirectory() as td,
            pytest.raises(ValueError, match="length mismatch"),
        ):
            _write_external_lc_files([lc], Path(td))

    def test_non_finite_flux_raises_valueerror(self) -> None:
        """Non-finite flux values raise ValueError."""
        lc = ExternalLightCurve(
            time_from_midtransit_days=np.array([-0.1, 0.0, 0.1]),
            flux=np.array([1.0, np.nan, 1.0]),  # NaN!
            flux_err=np.array([0.001, 0.001, 0.001]),
            filter="r",
        )

        with tempfile.TemporaryDirectory() as td, pytest.raises(ValueError, match="non-finite"):
            _write_external_lc_files([lc], Path(td))

    def test_all_supported_filters(self) -> None:
        """All TRICERATOPS+ supported filters work."""
        for filt in ["g", "r", "i", "z", "J", "H", "K"]:
            lc = ExternalLightCurve(
                time_from_midtransit_days=np.array([0.0]),
                flux=np.array([1.0]),
                flux_err=np.array([0.001]),
                filter=filt,  # type: ignore[arg-type]
            )

            with tempfile.TemporaryDirectory() as td:
                paths, filters = _write_external_lc_files([lc], Path(td))
                assert filters == [filt]
                assert filt in paths[0]  # filter name in filename
