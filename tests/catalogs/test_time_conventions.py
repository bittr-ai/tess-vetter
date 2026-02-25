from __future__ import annotations

import pytest

from tess_vetter.platform.catalogs.time_conventions import normalize_epoch_to_btjd


def test_normalize_epoch_to_btjd_with_bjd_reference() -> None:
    assert normalize_epoch_to_btjd(1.5, bjd_reference=2458000.0) == pytest.approx(1001.5)


def test_normalize_epoch_to_btjd_with_mjd_reference() -> None:
    # MJDREF=59000 corresponds to BJD reference 2459000.5
    assert normalize_epoch_to_btjd(1.0, mjd_reference=59000.0) == pytest.approx(2001.5)


def test_normalize_epoch_to_btjd_handles_common_offsets() -> None:
    assert normalize_epoch_to_btjd(2459001.5) == pytest.approx(2001.5)
    assert normalize_epoch_to_btjd(59001.0) == pytest.approx(2001.5)
    assert normalize_epoch_to_btjd(9001.5) == pytest.approx(2001.5)
