from tess_vetter.api.pixel_prf import (
    PRFParams,
    build_prf_model,
    get_prf_model,
    prf_params_from_dict,
)


def test_pixel_prf_facade_imports() -> None:
    params = prf_params_from_dict(
        {"sigma_row": 1.5, "sigma_col": 1.5, "theta": 0.0, "amplitude": 1.0}
    )
    assert isinstance(params, PRFParams)
    assert callable(build_prf_model)
    assert callable(get_prf_model)
