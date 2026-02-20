from __future__ import annotations

import pytest

from tess_vetter.errors import ErrorType, make_error
from tess_vetter.platform.network.timeout import network_timeout


def test_make_error_envelope() -> None:
    err = make_error(ErrorType.INVALID_DATA, "bad", tic_id=123)
    assert err.type == ErrorType.INVALID_DATA
    assert err.message == "bad"
    assert err.context["tic_id"] == 123


def test_network_timeout_rejects_non_positive_seconds_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tess_vetter.platform.network.timeout._is_timeout_supported", lambda: True
    )
    with pytest.raises(ValueError), network_timeout(0, operation="x"):
        pass
