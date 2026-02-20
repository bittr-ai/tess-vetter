from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

from tess_vetter import __version__

__all__ = ["__version__"]

warnings.warn(
    "`bittr_tess_vetter` is deprecated; import `tess_vetter` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_LEGACY_PREFIX = "bittr_tess_vetter"
_CANONICAL_PREFIX = "tess_vetter"


def __getattr__(name: str):
    return getattr(importlib.import_module(_CANONICAL_PREFIX), name)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def create_module(self, spec):
        module = importlib.import_module(self.new_name)
        sys.modules[self.old_name] = module
        return module

    def exec_module(self, module):
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith(f"{_LEGACY_PREFIX}."):
            return None

        suffix = fullname[len(_LEGACY_PREFIX) :]
        canonical_name = f"{_CANONICAL_PREFIX}{suffix}"
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None

        is_package = canonical_spec.submodule_search_locations is not None
        alias_spec = importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, canonical_name),
            origin=canonical_spec.origin,
            is_package=is_package,
        )
        if alias_spec is None:
            return None
        if is_package:
            alias_spec.submodule_search_locations = list(
                canonical_spec.submodule_search_locations or []
            )
        return alias_spec


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
