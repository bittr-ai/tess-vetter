from __future__ import annotations

from bittr_tess_vetter.validation.triceratops_fpp import _normalize_triceratops_filter


def test_normalize_triceratops_filter_handles_common_exofop_labels() -> None:
    assert _normalize_triceratops_filter(None) == "Vis"
    assert _normalize_triceratops_filter("") == "Vis"

    assert _normalize_triceratops_filter("Kcont") == "K"
    assert _normalize_triceratops_filter("Brgamma") == "K"
    assert _normalize_triceratops_filter("Ks") == "K"
    assert _normalize_triceratops_filter("K'") == "K"
    assert _normalize_triceratops_filter("k-prime") == "K"

    assert _normalize_triceratops_filter("Hcont") == "H"
    assert _normalize_triceratops_filter("Jcont") == "J"

    assert _normalize_triceratops_filter("V") == "Vis"
    assert _normalize_triceratops_filter("Vband") == "Vis"
    assert _normalize_triceratops_filter("clear") == "Vis"


def test_normalize_triceratops_filter_leaves_kp_ambiguous() -> None:
    # Kp is ambiguous (Kepler band vs K-prime). We intentionally do not map it to K.
    assert _normalize_triceratops_filter("kp") == "Vis"

