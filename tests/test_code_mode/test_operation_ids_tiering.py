from __future__ import annotations

import pytest

from tess_vetter.code_mode.registries.operation_ids import (
    OPERATION_NAMESPACE,
    build_operation_id,
    is_valid_operation_id,
    parse_operation_id,
    validate_operation_id,
)
from tess_vetter.code_mode.registries.tiering import (
    DEFAULT_GOLDEN_PATH_SYMBOLS,
    ApiSymbol,
    tier_for_api_symbol,
)


def test_operation_id_roundtrip_is_deterministic() -> None:
    operation_id = build_operation_id(tier="primitive", name="fold")

    assert operation_id == f"{OPERATION_NAMESPACE}.primitive.fold"

    parsed = parse_operation_id(operation_id)
    assert parsed.namespace == OPERATION_NAMESPACE
    assert parsed.tier == "primitive"
    assert parsed.name == "fold"
    assert parsed.id == operation_id


def test_validate_operation_id_returns_original_string() -> None:
    operation_id = "code_mode.golden_path.vet_candidate"
    assert validate_operation_id(operation_id) == operation_id


@pytest.mark.parametrize(
    "operation_id",
    [
        "code_mode.golden_path.vet_candidate",
        "code_mode.primitive.median_detrend",
        "code_mode.internal.export_debug",
    ],
)
def test_is_valid_operation_id_true_for_canonical_ids(operation_id: str) -> None:
    assert is_valid_operation_id(operation_id) is True


@pytest.mark.parametrize(
    "operation_id",
    [
        "code_mode.golden.vet_candidate",  # unsupported tier token
        "code_mode.golden_path.VetCandidate",  # non-snake-case name
        "wrong_ns.golden_path.vet_candidate",  # wrong namespace
        "code_mode.golden_path",  # missing operation name
        "code_mode.golden_path.vet.candidate",  # too many segments
    ],
)
def test_is_valid_operation_id_false_for_invalid_ids(operation_id: str) -> None:
    assert is_valid_operation_id(operation_id) is False


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("tess_vetter.api.workflow.vet_candidate", "golden_path"),
        (ApiSymbol(module="tess_vetter.api.primitives", name="fold"), "primitive"),
        ("tess_vetter.api.exovetter.check_modshift", "internal"),
    ],
)
def test_tier_for_api_symbol_basics(symbol: str | ApiSymbol, expected: str) -> None:
    assert tier_for_api_symbol(symbol) == expected


def test_golden_path_override_set_stays_small_and_curated() -> None:
    # Guard against regressing into a monolithic hand-maintained mapping.
    assert "vet_candidate" in DEFAULT_GOLDEN_PATH_SYMBOLS
    assert len(DEFAULT_GOLDEN_PATH_SYMBOLS) <= 16
