"""Unit tests for MASTClient with mocked lightkurve.

This module tests the MASTClient's conversion logic, error handling,
and data processing without making actual network calls to MAST.

Test coverage:
- search_lightcurve() returns SearchResult objects
- download_lightcurve() returns LightCurveData with correct dtypes
- get_target_info() returns Target with StellarParameters
- Quality mask filtering
- Flux normalization (median ~1.0)
- Error handling (LightCurveNotFoundError, TargetNotFoundError)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import astroquery early to initialize its logger before pytest captures warnings
# This avoids conflicts between astropy's logger and pytest's warning capture
try:
    import astroquery.mast  # noqa: F401
except ImportError:
    pass  # astroquery not installed - tests will skip

from bittr_tess_vetter.api.lightcurve import LightCurveData
from bittr_tess_vetter.api.target import StellarParameters, Target
from bittr_tess_vetter.io.mast_client import (
    DEFAULT_QUALITY_MASK,
    LightCurveNotFoundError,
    MASTClient,
    MASTClientError,
    SearchResult,
    TargetNotFoundError,
)

# -----------------------------------------------------------------------------
# Fixtures for mock objects
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_lightkurve():
    """Create a mocked lightkurve module."""
    mock_lk = MagicMock()
    return mock_lk


@pytest.fixture
def mock_search_result_row():
    """Create a mock search result row from lightkurve."""
    row = MagicMock()
    row.mission = ["TESS Sector 1"]
    row.sequence_number = 1
    row.author = "SPOC"
    row.exptime = 120.0
    row.distance = 0.5
    row.download = MagicMock()
    return row


@pytest.fixture
def mock_search_result(mock_search_result_row):
    """Create a mock SearchResult collection from lightkurve."""
    result = MagicMock()
    result.__len__ = MagicMock(return_value=1)
    result.__getitem__ = MagicMock(return_value=mock_search_result_row)
    return result


@pytest.fixture
def mock_lightcurve():
    """Create a mock lightkurve LightCurve object with realistic data."""
    lc = MagicMock()

    # Generate realistic light curve data (100 points, ~2 days)
    n_points = 100
    time_values = np.linspace(1500.0, 1502.0, n_points)
    flux_values = np.ones(n_points) * 1000.0  # Raw flux ~1000 electrons/s
    flux_values += np.random.normal(0, 1, n_points)  # Add small noise
    flux_err_values = np.ones(n_points) * 0.5
    quality_values = np.zeros(n_points, dtype=np.int32)

    # Add some bad quality points
    quality_values[10] = 1  # Attitude tweak
    quality_values[50] = 128  # Discontinuity

    # Mock time attribute
    time_mock = MagicMock()
    time_mock.value = time_values
    lc.time = time_mock

    # Mock flux attributes
    flux_mock = MagicMock()
    flux_mock.value = flux_values
    lc.flux = flux_mock
    lc.pdcsap_flux = flux_mock
    lc.sap_flux = flux_mock

    flux_err_mock = MagicMock()
    flux_err_mock.value = flux_err_values
    lc.flux_err = flux_err_mock
    lc.pdcsap_flux_err = flux_err_mock
    lc.sap_flux_err = flux_err_mock

    # Mock quality
    lc.quality = quality_values

    return lc


@pytest.fixture
def mock_tic_catalog_row():
    """Create a mock TIC catalog row from astroquery."""
    row = MagicMock()

    # Set up indexed access
    row.__getitem__ = lambda self, key: {
        "Teff": 5800.0,
        "logg": 4.4,
        "rad": 1.0,
        "mass": 1.0,
        "Tmag": 10.5,
        "contratio": 0.01,
        "lum": 1.0,
        "MH": 0.0,
        "ra": 120.5,
        "dec": -30.2,
        "pmRA": 10.5,
        "pmDEC": -5.2,
        "d": 100.0,
        "GAIA": 12345678901234,
        "TWOMASS": "12345678+9012345",
    }.get(key)

    return row


# -----------------------------------------------------------------------------
# Tests for search_lightcurve()
# -----------------------------------------------------------------------------


class TestSearchLightcurve:
    """Tests for MASTClient.search_lightcurve()."""

    def test_search_returns_search_results(self, mock_lightkurve, mock_search_result):
        """search_lightcurve() returns list of SearchResult objects."""
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            results = client.search_lightcurve(tic_id=261136679)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)

    def test_search_result_attributes(self, mock_lightkurve, mock_search_result):
        """SearchResult has correct attributes from mock data."""
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            results = client.search_lightcurve(tic_id=261136679)

            result = results[0]
            assert result.tic_id == 261136679
            assert result.sector == 1
            assert result.author == "SPOC"
            assert result.exptime == 120.0
            assert result.mission == "TESS"
            assert result.distance == 0.5

    def test_search_with_sector_filter(self, mock_lightkurve, mock_search_result):
        """search_lightcurve() passes sector parameter to lightkurve."""
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            client.search_lightcurve(tic_id=261136679, sector=5)

            mock_lightkurve.search_lightcurve.assert_called_once()
            call_kwargs = mock_lightkurve.search_lightcurve.call_args[1]
            assert call_kwargs.get("sector") == 5

    def test_search_empty_result(self, mock_lightkurve):
        """search_lightcurve() returns empty list when no results found."""
        mock_lightkurve.search_lightcurve.return_value = None

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            results = client.search_lightcurve(tic_id=261136679)

            assert results == []

    def test_search_multiple_sectors(self, mock_lightkurve):
        """search_lightcurve() returns multiple SearchResults sorted by sector."""
        # Create multiple mock rows
        rows = []
        for sector in [3, 1, 2]:  # Intentionally unsorted
            row = MagicMock()
            row.mission = [f"TESS Sector {sector}"]
            row.sequence_number = sector
            row.author = "SPOC"
            row.exptime = 120.0
            row.distance = 0.5
            rows.append(row)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=3)
        mock_result.__getitem__ = lambda self, i: rows[i]

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            results = client.search_lightcurve(tic_id=261136679)

            assert len(results) == 3
            # Should be sorted by sector
            assert [r.sector for r in results] == [1, 2, 3]

    def test_search_error_handling(self, mock_lightkurve):
        """search_lightcurve() raises MASTClientError on API failure."""
        mock_lightkurve.search_lightcurve.side_effect = Exception("Network error")

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            with pytest.raises(MASTClientError, match="Failed to search MAST"):
                client.search_lightcurve(tic_id=261136679)


# -----------------------------------------------------------------------------
# Tests for download_lightcurve()
# -----------------------------------------------------------------------------


class TestDownloadLightcurve:
    """Tests for MASTClient.download_lightcurve()."""

    def test_download_returns_lightcurve_data(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """download_lightcurve() returns LightCurveData object."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            assert isinstance(lc_data, LightCurveData)

    def test_download_correct_dtypes(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """LightCurveData has correct numpy dtypes."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            assert lc_data.time.dtype == np.float64
            assert lc_data.flux.dtype == np.float64
            assert lc_data.flux_err.dtype == np.float64
            assert lc_data.quality.dtype == np.int32
            assert lc_data.valid_mask.dtype == np.bool_

    def test_download_metadata(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """LightCurveData has correct metadata attributes."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            assert lc_data.tic_id == 261136679
            assert lc_data.sector == 1
            assert lc_data.cadence_seconds > 0

    def test_download_cadence_seconds_robust_to_gaps(
        self, mock_lightkurve, mock_search_result
    ) -> None:
        """cadence_seconds uses median dt (robust to occasional large gaps)."""
        lc = MagicMock()

        # Mostly 2-min cadence in days, with one huge gap injected.
        cadence_days = 120.0 / 86400.0
        time_values = 1500.0 + np.arange(1000, dtype=np.float64) * cadence_days
        time_values[500:] += 3.0  # 3-day gap

        flux_values = np.ones_like(time_values) * 1000.0
        flux_err_values = np.ones_like(time_values) * 0.5
        quality_values = np.zeros(len(time_values), dtype=np.int32)

        time_mock = MagicMock()
        time_mock.value = time_values
        lc.time = time_mock

        flux_mock = MagicMock()
        flux_mock.value = flux_values
        lc.flux = flux_mock
        lc.pdcsap_flux = flux_mock

        flux_err_mock = MagicMock()
        flux_err_mock.value = flux_err_values
        lc.flux_err = flux_err_mock
        lc.pdcsap_flux_err = flux_err_mock

        lc.quality = quality_values

        mock_search_result.__getitem__.return_value.download.return_value = lc
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(normalize=True)
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)
            assert abs(lc_data.cadence_seconds - 120.0) < 5.0

    def test_download_not_found_error(self, mock_lightkurve):
        """download_lightcurve() raises LightCurveNotFoundError when not found."""
        mock_lightkurve.search_lightcurve.return_value = None

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            with pytest.raises(LightCurveNotFoundError, match="No light curve found"):
                client.download_lightcurve(tic_id=261136679, sector=99)

    def test_download_invalid_flux_type(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """download_lightcurve() raises ValueError for invalid flux_type."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            with pytest.raises(ValueError, match="flux_type must be"):
                client.download_lightcurve(tic_id=261136679, sector=1, flux_type="invalid")

    def test_download_sap_flux(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """download_lightcurve() can use SAP flux type."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, flux_type="sap")

            assert isinstance(lc_data, LightCurveData)

    def test_download_error_handling(self, mock_lightkurve, mock_search_result):
        """download_lightcurve() raises MASTClientError on download failure."""
        mock_search_result.__getitem__.return_value.download.side_effect = Exception("Download failed")
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            with pytest.raises(MASTClientError, match="Failed to download"):
                client.download_lightcurve(tic_id=261136679, sector=1)

    def test_download_missing_flux_err_is_estimated(
        self, mock_lightkurve, mock_search_result
    ) -> None:
        """Missing flux_err is estimated and recorded in provenance."""
        # LightCurve-like object without any flux_err attributes (spec to make hasattr work).
        lc = MagicMock(spec=["time", "flux", "pdcsap_flux", "sap_flux", "quality"])

        n_points = 100
        time_values = np.linspace(1500.0, 1502.0, n_points)
        flux_values = np.ones(n_points, dtype=np.float64) * 1000.0
        quality_values = np.zeros(n_points, dtype=np.int32)

        time_mock = MagicMock()
        time_mock.value = time_values
        lc.time = time_mock

        flux_mock = MagicMock()
        flux_mock.value = flux_values
        lc.flux = flux_mock
        lc.pdcsap_flux = flux_mock
        lc.sap_flux = flux_mock

        lc.quality = quality_values

        mock_search_result.__getitem__.return_value.download.return_value = lc
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(normalize=True)
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)
            assert np.all(np.isfinite(lc_data.flux_err))
            assert lc_data.provenance is not None
            assert lc_data.provenance.flux_err_kind == "estimated_missing"


# -----------------------------------------------------------------------------
# Tests for quality mask filtering
# -----------------------------------------------------------------------------


class TestQualityMaskFiltering:
    """Tests for quality mask filtering in download_lightcurve()."""

    def test_default_quality_mask_filters_bad_points(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """Default quality mask filters points with bad quality flags."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()  # Uses DEFAULT_QUALITY_MASK
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            # Points with quality flags matching mask should be invalid
            # mock_lightcurve has quality[10] = 1 (attitude tweak) and quality[50] = 128 (discontinuity)
            # Both should be filtered by DEFAULT_QUALITY_MASK
            assert not lc_data.valid_mask[10]
            assert not lc_data.valid_mask[50]

    def test_custom_quality_mask(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """Custom quality mask can be passed to download_lightcurve()."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            # Use a custom mask that only filters attitude tweaks (bit 1)
            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, quality_mask=1)

            # Point with attitude tweak should be filtered
            assert not lc_data.valid_mask[10]
            # Point with discontinuity (128) should NOT be filtered with mask=1
            assert lc_data.valid_mask[50]

    def test_no_quality_mask(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """Quality mask of 0 allows all points."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(quality_mask=0)
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            # All points should be valid (assuming no NaNs)
            assert lc_data.valid_mask[10]
            assert lc_data.valid_mask[50]

    def test_quality_mask_with_nans(self, mock_lightkurve, mock_search_result):
        """NaN values are filtered regardless of quality mask."""
        lc = MagicMock()
        n_points = 50
        time_values = np.linspace(1500.0, 1501.0, n_points)
        flux_values = np.ones(n_points) * 1000.0
        flux_err_values = np.ones(n_points) * 0.5
        quality_values = np.zeros(n_points, dtype=np.int32)

        # Add NaN values
        flux_values[5] = np.nan
        time_values[25] = np.nan
        flux_err_values[35] = np.nan

        time_mock = MagicMock()
        time_mock.value = time_values
        lc.time = time_mock

        flux_mock = MagicMock()
        flux_mock.value = flux_values
        lc.flux = flux_mock
        lc.pdcsap_flux = flux_mock

        flux_err_mock = MagicMock()
        flux_err_mock.value = flux_err_values
        lc.flux_err = flux_err_mock
        lc.pdcsap_flux_err = flux_err_mock

        lc.quality = quality_values

        mock_search_result.__getitem__.return_value.download.return_value = lc
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(quality_mask=0)  # No quality filtering
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            # NaN points should be invalid
            assert not lc_data.valid_mask[5]
            assert not lc_data.valid_mask[25]
            assert not lc_data.valid_mask[35]


# -----------------------------------------------------------------------------
# Tests for flux normalization
# -----------------------------------------------------------------------------


class TestFluxNormalization:
    """Tests for flux normalization in download_lightcurve()."""

    def test_normalized_flux_median_near_one(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """Normalized flux has median close to 1.0."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(normalize=True)
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            # Median of valid flux values should be ~1.0
            median_flux = np.median(lc_data.flux[lc_data.valid_mask])
            assert abs(median_flux - 1.0) < 0.01

    def test_normalization_disabled(self, mock_lightkurve, mock_search_result, mock_lightcurve):
        """Flux is not normalized when normalize=False."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(normalize=False)
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            # Flux should be raw values (~1000)
            median_flux = np.median(lc_data.flux[lc_data.valid_mask])
            assert median_flux > 100  # Not normalized

    def test_flux_error_scaled_with_normalization(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """Flux errors are scaled proportionally during normalization."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient(normalize=True)
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1)

            # Error should be scaled by same factor as flux
            # Original flux ~1000, error ~0.5, so relative error ~0.0005
            # After normalization, flux ~1.0, error should be ~0.0005
            relative_error = np.median(lc_data.flux_err[lc_data.valid_mask]) / np.median(
                lc_data.flux[lc_data.valid_mask]
            )
            assert relative_error < 0.01  # Relative error preserved


# -----------------------------------------------------------------------------
# Tests for get_target_info()
# -----------------------------------------------------------------------------


class TestGetTargetInfo:
    """Tests for MASTClient.get_target_info()."""

    def test_get_target_info_returns_target(self, mock_lightkurve, mock_tic_catalog_row):
        """get_target_info() returns Target object."""
        mock_catalog_result = MagicMock()
        mock_catalog_result.__len__ = MagicMock(return_value=1)
        mock_catalog_result.__getitem__ = MagicMock(return_value=mock_tic_catalog_row)

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = mock_catalog_result

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                target = client.get_target_info(tic_id=261136679)

                assert isinstance(target, Target)
                assert target.tic_id == 261136679

    def test_get_target_info_stellar_parameters(self, mock_lightkurve, mock_tic_catalog_row):
        """Target has StellarParameters with correct values."""
        mock_catalog_result = MagicMock()
        mock_catalog_result.__len__ = MagicMock(return_value=1)
        mock_catalog_result.__getitem__ = MagicMock(return_value=mock_tic_catalog_row)

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = mock_catalog_result

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                target = client.get_target_info(tic_id=261136679)

                assert isinstance(target.stellar, StellarParameters)
                assert target.stellar.teff == 5800.0
                assert target.stellar.logg == 4.4
                assert target.stellar.radius == 1.0
                assert target.stellar.mass == 1.0
                assert target.stellar.tmag == 10.5

    def test_get_target_info_astrometric_data(self, mock_lightkurve, mock_tic_catalog_row):
        """Target has correct astrometric data."""
        mock_catalog_result = MagicMock()
        mock_catalog_result.__len__ = MagicMock(return_value=1)
        mock_catalog_result.__getitem__ = MagicMock(return_value=mock_tic_catalog_row)

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = mock_catalog_result

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                target = client.get_target_info(tic_id=261136679)

                assert target.ra == 120.5
                assert target.dec == -30.2
                assert target.pmra == 10.5
                assert target.pmdec == -5.2
                assert target.distance_pc == 100.0

    def test_get_target_info_cross_match_ids(self, mock_lightkurve, mock_tic_catalog_row):
        """Target has correct cross-match identifiers."""
        mock_catalog_result = MagicMock()
        mock_catalog_result.__len__ = MagicMock(return_value=1)
        mock_catalog_result.__getitem__ = MagicMock(return_value=mock_tic_catalog_row)

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = mock_catalog_result

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                target = client.get_target_info(tic_id=261136679)

                assert target.gaia_dr3_id == 12345678901234
                assert target.twomass_id == "12345678+9012345"

    def test_get_target_info_not_found(self, mock_lightkurve):
        """get_target_info() raises TargetNotFoundError when target not in TIC."""
        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = None

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                with pytest.raises(TargetNotFoundError, match="not found in catalog"):
                    client.get_target_info(tic_id=999999999)

    def test_get_target_info_empty_result(self, mock_lightkurve):
        """get_target_info() raises TargetNotFoundError for empty result."""
        mock_catalog_result = MagicMock()
        mock_catalog_result.__len__ = MagicMock(return_value=0)

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = mock_catalog_result

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                with pytest.raises(TargetNotFoundError, match="not found in catalog"):
                    client.get_target_info(tic_id=999999999)

    def test_get_target_info_missing_params(self, mock_lightkurve):
        """Target handles missing stellar parameters gracefully."""
        # Create a row with some missing parameters
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "Teff": 5800.0,
            "ra": 120.5,
            "dec": -30.2,
            # Missing: logg, rad, mass, etc.
        }.get(key)

        mock_catalog_result = MagicMock()
        mock_catalog_result.__len__ = MagicMock(return_value=1)
        mock_catalog_result.__getitem__ = MagicMock(return_value=row)

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            with patch("astroquery.mast.Catalogs") as mock_catalogs:
                mock_catalogs.query_criteria.return_value = mock_catalog_result

                client = MASTClient()
                client._lk = mock_lightkurve
                client._lk_imported = True

                target = client.get_target_info(tic_id=261136679)

                assert target.stellar.teff == 5800.0
                assert target.stellar.radius is None
                assert target.stellar.mass is None


# -----------------------------------------------------------------------------
# Tests for error handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in MASTClient."""

    def test_lightkurve_import_error(self):
        """MASTClient raises MASTClientError if lightkurve not installed."""
        client = MASTClient()
        client._lk_imported = False

        # Clear any cached module
        import sys

        original_modules = sys.modules.copy()

        # Temporarily remove lightkurve from modules
        if "lightkurve" in sys.modules:
            del sys.modules["lightkurve"]

        try:
            with patch.dict("sys.modules", {"lightkurve": None}):
                with patch("builtins.__import__", side_effect=ImportError("No module")):
                    with pytest.raises(MASTClientError, match="lightkurve is required"):
                        client._ensure_lightkurve()
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def test_mast_client_error_inheritance(self):
        """MASTClientError exceptions inherit properly."""
        assert issubclass(LightCurveNotFoundError, MASTClientError)
        assert issubclass(TargetNotFoundError, MASTClientError)

    @pytest.mark.skip(reason="Complex import mocking - astroquery import error is an edge case")
    def test_astroquery_import_error(self, mock_lightkurve):
        """get_target_info() raises MASTClientError if astroquery not available."""
        # This test is skipped because mocking import errors is complex and
        # conflicts with astropy's logger initialization.
        pass


# -----------------------------------------------------------------------------
# Tests for helper methods
# -----------------------------------------------------------------------------


class TestHelperMethods:
    """Tests for MASTClient helper methods."""

    def test_get_available_sectors(self, mock_lightkurve):
        """get_available_sectors() returns sorted list of sectors."""
        # Create multiple mock rows
        rows = []
        for sector in [5, 1, 10, 3]:
            row = MagicMock()
            row.mission = [f"TESS Sector {sector}"]
            row.sequence_number = sector
            row.author = "SPOC"
            row.exptime = 120.0
            row.distance = None
            rows.append(row)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=4)
        mock_result.__getitem__ = lambda self, i: rows[i]

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            sectors = client.get_available_sectors(tic_id=261136679)

            assert sectors == [1, 3, 5, 10]

    def test_get_float_with_masked_value(self):
        """_get_float() handles masked values correctly."""
        # Create a masked value
        masked_val = MagicMock()
        masked_val.mask = True

        row = MagicMock()
        row.__getitem__ = MagicMock(return_value=masked_val)

        result = MASTClient._get_float(row, "test_key")
        assert result is None

    def test_get_float_with_nan(self):
        """_get_float() returns None for NaN values."""
        row = MagicMock()
        row.__getitem__ = MagicMock(return_value=float("nan"))

        result = MASTClient._get_float(row, "test_key")
        assert result is None

    def test_get_float_with_inf(self):
        """_get_float() returns None for infinite values."""
        row = MagicMock()
        row.__getitem__ = MagicMock(return_value=float("inf"))

        result = MASTClient._get_float(row, "test_key")
        assert result is None

    def test_get_int_with_valid_value(self):
        """_get_int() returns integer for valid value."""
        row = MagicMock()
        row.__getitem__ = MagicMock(return_value=42)

        result = MASTClient._get_int(row, "test_key")
        assert result == 42

    def test_get_str_with_valid_value(self):
        """_get_str() returns string for valid value."""
        row = MagicMock()
        row.__getitem__ = MagicMock(return_value="test_value")

        result = MASTClient._get_str(row, "test_key")
        assert result == "test_value"


# -----------------------------------------------------------------------------
# Tests for client configuration
# -----------------------------------------------------------------------------


class TestClientConfiguration:
    """Tests for MASTClient configuration options."""

    def test_default_configuration(self):
        """MASTClient has correct default configuration."""
        client = MASTClient()

        assert client.quality_mask == DEFAULT_QUALITY_MASK
        assert client.author == "SPOC"
        assert client.normalize is True

    def test_custom_configuration(self):
        """MASTClient accepts custom configuration."""
        client = MASTClient(
            quality_mask=255,
            author="QLP",
            normalize=False,
        )

        assert client.quality_mask == 255
        assert client.author == "QLP"
        assert client.normalize is False

    def test_author_none_for_all_pipelines(self):
        """author=None searches all pipelines."""
        client = MASTClient(author=None)
        assert client.author is None


# -----------------------------------------------------------------------------
# Integration-style tests (still mocked)
# -----------------------------------------------------------------------------


class TestDownloadAllSectors:
    """Tests for download_all_sectors() method."""

    def test_download_all_sectors_success(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """download_all_sectors() downloads multiple sectors."""
        # Create search results for 3 sectors
        rows = []
        for sector in [1, 2, 3]:
            row = MagicMock()
            row.mission = [f"TESS Sector {sector}"]
            row.sequence_number = sector
            row.author = "SPOC"
            row.exptime = 120.0
            row.distance = None
            row.download = MagicMock(return_value=mock_lightcurve)
            rows.append(row)

        mock_multi_result = MagicMock()
        mock_multi_result.__len__ = MagicMock(return_value=3)
        mock_multi_result.__getitem__ = lambda self, i: rows[i]

        # First call returns multiple results, subsequent calls return single results
        call_count = [0]

        def search_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_multi_result
            return mock_search_result

        mock_lightkurve.search_lightcurve.side_effect = search_side_effect
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            light_curves = client.download_all_sectors(tic_id=261136679)

            assert len(light_curves) == 3
            assert all(isinstance(lc, LightCurveData) for lc in light_curves)

    def test_download_all_sectors_no_data(self, mock_lightkurve):
        """download_all_sectors() raises when no data available."""
        mock_lightkurve.search_lightcurve.return_value = None

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            with pytest.raises(LightCurveNotFoundError, match="No light curves available"):
                client.download_all_sectors(tic_id=261136679)

    def test_download_all_sectors_specific_sectors(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """download_all_sectors() can download specific sectors."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            light_curves = client.download_all_sectors(tic_id=261136679, sectors=[1, 5])

            assert len(light_curves) == 2


# -----------------------------------------------------------------------------
# Tests for exptime filtering
# -----------------------------------------------------------------------------


class TestExptimeFiltering:
    """Tests for exptime filtering in download_lightcurve()."""

    def test_download_with_exptime_filter(self, mock_lightkurve, mock_lightcurve):
        """download_lightcurve() filters by exptime when specified."""
        # Create search results with different exposure times
        rows = []
        for exptime in [20.0, 120.0]:
            row = MagicMock()
            row.mission = ["TESS Sector 1"]
            row.sequence_number = 1
            row.author = "SPOC"
            row.exptime = exptime
            row.distance = None
            row.download = MagicMock(return_value=mock_lightcurve)
            rows.append(row)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=2)
        mock_result.__getitem__ = lambda self, i: rows[i]

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            # Request 20-second cadence
            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, exptime=20)

            assert isinstance(lc_data, LightCurveData)
            assert rows[0].download.call_count == 1
            assert rows[1].download.call_count == 0

    def test_download_with_exptime_filter_120s(self, mock_lightkurve, mock_lightcurve):
        """download_lightcurve() can filter for 120s cadence."""
        # Create search results with different exposure times
        rows = []
        for exptime in [20.0, 120.0]:
            row = MagicMock()
            row.mission = ["TESS Sector 1"]
            row.sequence_number = 1
            row.author = "SPOC"
            row.exptime = exptime
            row.distance = None
            row.download = MagicMock(return_value=mock_lightcurve)
            rows.append(row)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=2)
        mock_result.__getitem__ = lambda self, i: rows[i]

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            # Request 120-second cadence
            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, exptime=120)

            assert isinstance(lc_data, LightCurveData)
            assert rows[0].download.call_count == 0
            assert rows[1].download.call_count == 1

    def test_download_exptime_none_prefers_120s(self, mock_lightkurve, mock_lightcurve):
        """When exptime is omitted, selection prefers the 120s cadence product."""
        rows = []
        for exptime in [20.0, 120.0]:
            row = MagicMock()
            row.mission = ["TESS Sector 1"]
            row.sequence_number = 1
            row.author = "SPOC"
            row.exptime = exptime
            row.distance = None
            row.download = MagicMock(return_value=mock_lightcurve)
            rows.append(row)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=2)
        mock_result.__getitem__ = lambda self, i: rows[i]

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, exptime=None)
            assert isinstance(lc_data, LightCurveData)
            assert rows[0].download.call_count == 0
            assert rows[1].download.call_count == 1

    def test_download_exptime_not_found(self, mock_lightkurve, mock_lightcurve):
        """download_lightcurve() raises error when exptime not available."""
        # Create search results with only 120s cadence
        row = MagicMock()
        row.mission = ["TESS Sector 1"]
        row.sequence_number = 1
        row.author = "SPOC"
        row.exptime = 120.0
        row.distance = None
        row.download = MagicMock(return_value=mock_lightcurve)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=1)
        mock_result.__getitem__ = lambda self, i: row

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            # Request 20-second cadence (not available)
            with pytest.raises(LightCurveNotFoundError, match="exptime=20"):
                client.download_lightcurve(tic_id=261136679, sector=1, exptime=20)

    def test_download_exptime_none_uses_first(
        self, mock_lightkurve, mock_search_result, mock_lightcurve
    ):
        """download_lightcurve() uses first result when exptime is None."""
        mock_search_result.__getitem__.return_value.download.return_value = mock_lightcurve
        mock_lightkurve.search_lightcurve.return_value = mock_search_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            # No exptime specified - should use first available
            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, exptime=None)

            assert isinstance(lc_data, LightCurveData)

    def test_download_exptime_with_tolerance(self, mock_lightkurve, mock_lightcurve):
        """download_lightcurve() matches exptime with 1s tolerance."""
        # Create search result with exptime slightly different from requested
        row = MagicMock()
        row.mission = ["TESS Sector 1"]
        row.sequence_number = 1
        row.author = "SPOC"
        row.exptime = 120.4  # Slightly off from 120
        row.distance = None
        row.download = MagicMock(return_value=mock_lightcurve)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=1)
        mock_result.__getitem__ = lambda self, i: row
        mock_result.download = MagicMock(return_value=mock_lightcurve)

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            # Request 120s - should match 120.4s within tolerance
            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, exptime=120)

            assert isinstance(lc_data, LightCurveData)

    def test_download_exptime_handles_astropy_quantity(self, mock_lightkurve, mock_lightcurve):
        """download_lightcurve() handles exptime as astropy Quantity."""
        # Create search result with exptime as astropy-like Quantity
        exptime_quantity = MagicMock()
        exptime_quantity.value = 120.0

        row = MagicMock()
        row.mission = ["TESS Sector 1"]
        row.sequence_number = 1
        row.author = "SPOC"
        row.exptime = exptime_quantity
        row.distance = None
        row.download = MagicMock(return_value=mock_lightcurve)

        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=1)
        mock_result.__getitem__ = lambda self, i: row
        mock_result.download = MagicMock(return_value=mock_lightcurve)

        mock_lightkurve.search_lightcurve.return_value = mock_result

        with patch.dict("sys.modules", {"lightkurve": mock_lightkurve}):
            client = MASTClient()
            client._lk = mock_lightkurve
            client._lk_imported = True

            lc_data = client.download_lightcurve(tic_id=261136679, sector=1, exptime=120)

            assert isinstance(lc_data, LightCurveData)
