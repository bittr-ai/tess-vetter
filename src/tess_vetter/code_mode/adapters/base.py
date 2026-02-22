"""Core adapter model for code-mode operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tess_vetter.code_mode.operation_spec import OperationSpec


@dataclass(frozen=True)
class OperationAdapter:
    """Runtime adapter binding operation metadata to a callable."""

    spec: OperationSpec
    fn: Callable[..., Any]

    @property
    def id(self) -> str:
        return self.spec.id

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


__all__ = ["OperationAdapter"]
