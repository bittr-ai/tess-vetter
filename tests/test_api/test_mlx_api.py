from __future__ import annotations

import importlib.util


def test_api_mlx_module_importable() -> None:
    # Importing the facade must not require MLX to be installed.
    from tess_vetter.api import mlx  # noqa: F401


def test_api_root_mlx_exports_guarded() -> None:
    from tess_vetter import api

    expected_available = importlib.util.find_spec("mlx") is not None
    assert expected_available == api.MLX_AVAILABLE

    if expected_available:
        # When MLX is installed, these should be available from the root surface.
        assert api.score_fixed_period is not None  # type: ignore[attr-defined]
        assert api.smooth_box_template is not None  # type: ignore[attr-defined]
        assert api.integrated_gradients is not None  # type: ignore[attr-defined]
    else:
        # When MLX isn't installed, they should not be re-exported from the root.
        assert not hasattr(api, "score_fixed_period")
        assert not hasattr(api, "smooth_box_template")
        assert not hasattr(api, "integrated_gradients")
