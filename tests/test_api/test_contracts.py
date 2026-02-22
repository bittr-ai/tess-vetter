"""Tests for API-layer callable signature contracts."""

from __future__ import annotations

from tess_vetter.api.contracts import callable_input_schema_from_signature


def test_callable_input_schema_from_signature_is_deterministic() -> None:
    class _Callable:
        def __call__(self, z: int, a: int = 1, *args: object, req: str = "x", **kwargs: object) -> None:
            _ = args
            _ = kwargs

    schema = callable_input_schema_from_signature(_Callable())

    assert schema == {
        "type": "object",
        "properties": {"a": {}, "req": {}, "z": {}},
        "additionalProperties": True,
        "required": ["z"],
    }


def test_callable_input_schema_from_signature_ignores_self_and_cls() -> None:
    class _Example:
        def instance(self, required: int, optional: int = 0) -> int:
            return required + optional

        @classmethod
        def cls_method(cls, token: str, enabled: bool = True) -> str:
            return token if enabled else ""

    instance_schema = callable_input_schema_from_signature(_Example.instance)
    classmethod_schema = callable_input_schema_from_signature(_Example.cls_method)

    assert instance_schema == {
        "type": "object",
        "properties": {"optional": {}, "required": {}},
        "additionalProperties": False,
        "required": ["required"],
    }
    assert classmethod_schema == {
        "type": "object",
        "properties": {"enabled": {}, "token": {}},
        "additionalProperties": False,
        "required": ["token"],
    }


def test_callable_input_schema_from_signature_fallback_for_non_callable() -> None:
    schema = callable_input_schema_from_signature("not callable")

    assert schema == {"type": "object", "properties": {}, "additionalProperties": True}


def test_callable_input_schema_from_signature_fallback_for_uninspectable_callable() -> None:
    schema = callable_input_schema_from_signature(type)

    assert schema == {"type": "object", "properties": {}, "additionalProperties": True}
