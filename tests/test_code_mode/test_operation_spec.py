from __future__ import annotations

import pytest
from pydantic import ValidationError

from tess_vetter.code_mode.operation_spec import OperationSpec


def test_operation_spec_valid_defaults() -> None:
    spec = OperationSpec(
        id="code_mode.golden.vet_candidate",
        name="Vet Candidate",
    )

    assert spec.version == "1.0"
    assert spec.tier == "golden"
    assert spec.deprecated is False


def test_operation_spec_validates_id_pattern() -> None:
    with pytest.raises(ValidationError):
        OperationSpec(id="vet_candidate", name="Bad ID")


@pytest.mark.parametrize("version", ["1", "1.0.0", "v1.0", "01.2"])
def test_operation_spec_validates_version(version: str) -> None:
    with pytest.raises(ValidationError):
        OperationSpec(
            id="code_mode.primitive.fold",
            name="Fold",
            version=version,
        )


def test_operation_spec_requires_deprecated_for_replacement() -> None:
    with pytest.raises(ValidationError):
        OperationSpec(
            id="code_mode.primitive.fold",
            name="Fold",
            replaced_by="code_mode.primitive.median_detrend",
        )


def test_operation_spec_validates_replaced_by_pattern() -> None:
    with pytest.raises(ValidationError):
        OperationSpec(
            id="code_mode.primitive.fold",
            name="Fold",
            deprecated=True,
            replaced_by="bad.id",
        )


def test_operation_spec_sorts_tier_tags_deterministically() -> None:
    spec = OperationSpec(
        id="code_mode.primitive.fold",
        name="Fold",
        tier_tags=("zeta", "alpha", "alpha"),
    )

    assert spec.tier_tags == ("alpha", "zeta")
