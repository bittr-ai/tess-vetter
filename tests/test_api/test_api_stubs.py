from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import tess_vetter.api as api


def _stub_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "tess_vetter" / "api" / "__init__.pyi"


def _stub_tree() -> ast.Module:
    return ast.parse(_stub_path().read_text(encoding="utf-8"))


def _stub_function_signature(name: str) -> tuple[list[str], bool]:
    tree = _stub_tree()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            names: list[str] = [arg.arg for arg in node.args.posonlyargs]
            names.extend(arg.arg for arg in node.args.args)
            names.extend(arg.arg for arg in node.args.kwonlyargs)
            has_var_kw = node.args.kwarg is not None
            return names, has_var_kw
    raise AssertionError(f"Function {name!r} not found in {str(_stub_path())!r}")


def _runtime_export_function(name: str):
    export_map = api._get_export_map()
    target = export_map.get(name)
    if target is None:
        raise AssertionError(f"Runtime export {name!r} missing from export map")
    module_name, attr_name = target
    if module_name == "tess_vetter.api" and attr_name == name:
        mod = importlib.import_module(f"{module_name}.{name}")
        return mod
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def test_api_stub_file_exists() -> None:
    assert _stub_path().exists(), "API stub file must exist for typed package exports"


def test_key_stubbed_names_resolve_on_runtime_api() -> None:
    required_names = {
        "LightCurve",
        "Ephemeris",
        "Candidate",
        "CheckResult",
        "VettingBundleResult",
        "StellarParams",
        "TPFStamp",
        "VettingPipeline",
        "PipelineConfig",
        "CheckRegistry",
        "CheckTier",
        "CheckRequirements",
        "VettingCheck",
        "vet_candidate",
        "run_periodogram",
        "list_checks",
        "describe_checks",
        "generate_report",
        "per_sector_vet",
        "run_candidate_workflow",
        "run_check",
        "run_checks",
        "export_bundle",
        "hydrate_cache_from_dataset",
        "load_contrast_curve_exofop_tbl",
        "vet",
        "periodogram",
        "localize",
        "MLX_AVAILABLE",
        "MATPLOTLIB_AVAILABLE",
    }

    for name in required_names:
        assert hasattr(api, name), f"Stubbed API name {name!r} must resolve at runtime"


def test_stubbed_function_prefix_signatures_match_runtime() -> None:
    # Basic guardrail: keep the leading stub parameter names synchronized with
    # runtime call signatures while still allowing trailing kwargs for flexibility.
    function_names = [
        "vet_candidate",
        "vet_many",
        "run_periodogram",
        "list_checks",
        "describe_checks",
        "generate_report",
        "per_sector_vet",
        "run_candidate_workflow",
        "run_check",
        "run_checks",
        "export_bundle",
        "hydrate_cache_from_dataset",
        "load_contrast_curve_exofop_tbl",
    ]

    for name in function_names:
        stub_params, has_var_kw = _stub_function_signature(name)
        runtime_sig = inspect.signature(_runtime_export_function(name))
        runtime_params = [
            p.name
            for p in runtime_sig.parameters.values()
            if p.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]

        runtime_index = 0
        for stub_param in stub_params:
            while runtime_index < len(runtime_params) and runtime_params[runtime_index] != stub_param:
                runtime_index += 1
            assert runtime_index < len(runtime_params), (
                f"Stub parameter order mismatch for {name}: "
                f"could not find {stub_param!r} in runtime={runtime_params}"
            )
            runtime_index += 1

        if not has_var_kw:
            assert runtime_params == stub_params, (
                f"Stub must include all runtime parameters for {name} when no **kwargs is declared"
            )
