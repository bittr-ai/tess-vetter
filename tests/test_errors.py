"""Tests for custom exceptions."""

from __future__ import annotations

import pytest

from bittr_tess_vetter.errors import MissingOptionalDependencyError


class TestMissingOptionalDependencyError:
    """Tests for MissingOptionalDependencyError exception."""

    def test_basic(self) -> None:
        exc = MissingOptionalDependencyError("tls")
        assert exc.extra == "tls"
        assert "tls" in str(exc)
        assert "pip install" in str(exc)

    def test_custom_hint(self) -> None:
        exc = MissingOptionalDependencyError("tls", install_hint="uv add bittr-tess-vetter[tls]")
        assert "uv add" in str(exc)

    def test_is_import_error(self) -> None:
        exc = MissingOptionalDependencyError("fit")
        assert isinstance(exc, ImportError)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(MissingOptionalDependencyError) as exc_info:
            raise MissingOptionalDependencyError("triceratops")
        assert exc_info.value.extra == "triceratops"
