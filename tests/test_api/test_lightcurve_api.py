from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from bittr_tess_vetter.api.lightcurve import LightCurveRef, make_data_ref
from bittr_tess_vetter.domain.lightcurve import LightCurveData


def test_make_data_ref_matches_expected_format() -> None:
    assert make_data_ref(141914082, 1) == "lc:141914082:1:pdcsap"
    assert make_data_ref(141914082, 1, flux_type="sap") == "lc:141914082:1:sap"


def test_lightcurve_ref_is_frozen_and_forbids_extra() -> None:
    n = 10
    data = LightCurveData(
        time=np.arange(n, dtype=np.float64),
        flux=np.ones(n, dtype=np.float64),
        flux_err=np.ones(n, dtype=np.float64),
        quality=np.zeros(n, dtype=np.int32),
        valid_mask=np.ones(n, dtype=np.bool_),
        tic_id=1,
        sector=2,
        cadence_seconds=120.0,
    )

    ref = LightCurveRef.from_data(data)
    try:
        ref.tic_id = 999  # type: ignore[misc]
        raise AssertionError("Expected ValidationError for frozen model assignment")
    except ValidationError:
        pass

    try:
        LightCurveRef(  # type: ignore[call-arg]
            data_ref="lc:1:2:pdcsap",
            tic_id=1,
            sector=2,
            n_points=n,
            n_valid=n,
            duration_days=float(data.duration_days),
            cadence_seconds=120.0,
            median_flux=float(data.median_flux),
            flux_std=float(data.flux_std),
            gap_fraction=float(data.gap_fraction),
            quality_flags_present=[0],
            extra_field="nope",
        )
        raise AssertionError("Expected ValidationError for extra fields")
    except ValidationError:
        pass


def test_lightcurve_ref_from_data_populates_expected_fields() -> None:
    n = 10
    data = LightCurveData(
        time=np.arange(n, dtype=np.float64),
        flux=np.ones(n, dtype=np.float64),
        flux_err=np.ones(n, dtype=np.float64),
        quality=np.zeros(n, dtype=np.int32),
        valid_mask=np.ones(n, dtype=np.bool_),
        tic_id=141914082,
        sector=1,
        cadence_seconds=120.0,
    )

    ref = LightCurveRef.from_data(data, flux_type="sap")
    assert ref.data_ref == "lc:141914082:1:sap"
    assert ref.tic_id == 141914082
    assert ref.sector == 1
    assert ref.n_points == n
    assert ref.n_valid == n
