"""Canonical JSON serialization and hashing utilities.

This module provides deterministic JSON serialization that produces identical
byte-level output for semantically equivalent data structures. This is critical
for content-addressed storage, cache keys, and reproducibility.

Key guarantees:
- Sorted keys (recursive) for deterministic ordering
- UTF-8 encoding for consistent byte representation
- No whitespace for minimal size
- Floats rounded to 10 decimal places for numerical stability
- NaN/Inf rejection for JSON compliance
- Arrays preserve order (only dicts are sorted)
- Numpy arrays converted to lists
- Datetime objects converted to ISO8601 strings

Example:
    >>> from bittr_tess_vetter.utils.canonical import canonical_json, canonical_hash
    >>> data = {"b": 2, "a": 1, "c": [3, 2, 1]}
    >>> canonical_json(data)
    b'{"a":1,"b":2,"c":[3,2,1]}'
    >>> canonical_hash(data)
    '...'  # SHA-256 hex digest
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np  # Required dependency

# Float precision for canonical representation
FLOAT_DECIMAL_PLACES = 10


class CanonicalEncoder(json.JSONEncoder):
    """Custom JSON encoder for canonical serialization.

    Handles:
    - Floats: Rounded to FLOAT_DECIMAL_PLACES, rejects NaN/Inf
    - Numpy arrays: Converted to lists
    - Numpy scalars: Converted to Python types
    - Datetime objects: Converted to ISO8601 strings
    - Sets: Converted to sorted lists
    """

    def default(self, o: Any) -> Any:
        """Convert non-standard types to JSON-serializable types."""
        # Handle numpy arrays
        if isinstance(o, np.ndarray):
            return o.tolist()

        # Handle numpy scalar types
        if isinstance(o, (np.integer, np.floating)):
            return o.item()

        # Handle numpy bool
        if isinstance(o, np.bool_):
            return bool(o)

        # Handle datetime objects
        if isinstance(o, datetime):
            return o.isoformat()

        # Handle date objects
        if isinstance(o, date):
            return o.isoformat()

        # Handle sets by converting to sorted lists
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=str)

        # Let the base class raise TypeError for unknown types
        return super().default(o)

    def encode(self, o: Any) -> str:
        """Encode object to JSON string with canonical float handling."""
        # Pre-process the object to handle floats and nested structures
        processed = self._process_value(o)
        return super().encode(processed)

    def _process_value(self, obj: Any) -> Any:
        """Recursively process values for canonical representation."""
        if isinstance(obj, dict):
            # Sort keys and process values recursively
            return {k: self._process_value(v) for k, v in sorted(obj.items())}

        elif isinstance(obj, (list, tuple)):
            # Process list items but preserve order
            return [self._process_value(item) for item in obj]

        elif isinstance(obj, np.ndarray):
            # Convert numpy array to list and process
            return [self._process_value(item) for item in obj.tolist()]

        elif isinstance(obj, (float, np.floating)):
            return self._process_float(float(obj))

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()

        elif isinstance(obj, (set, frozenset)):
            return [self._process_value(item) for item in sorted(obj, key=str)]

        else:
            return obj

    def _process_float(self, value: float) -> float | int:
        """Process a float value for canonical representation.

        Args:
            value: Float value to process.

        Returns:
            Rounded float value, or int if the value is whole.

        Raises:
            ValueError: If value is NaN or Inf.
        """
        if math.isnan(value):
            raise ValueError("NaN values are not allowed in canonical JSON")
        if math.isinf(value):
            raise ValueError("Inf values are not allowed in canonical JSON")

        # Round to specified decimal places
        rounded = round(value, FLOAT_DECIMAL_PLACES)

        # If the rounded value is effectively an integer, return int
        # This ensures 1.0 becomes 1 for cleaner output
        if rounded == int(rounded) and abs(rounded) < 2**53:
            return int(rounded)

        return rounded


def canonical_json(obj: Any) -> bytes:
    """Convert object to canonical JSON bytes.

    Produces deterministic, byte-level identical output for semantically
    equivalent data structures. This is suitable for content-addressed
    storage, cache keys, and hash computation.

    Features:
    - UTF-8 encoding
    - Sorted keys (recursive)
    - No whitespace
    - Floats: round to 10 decimal places, then stringify
    - Reject NaN/Inf (raises ValueError)
    - Arrays preserve order
    - Handle numpy arrays (convert to list)
    - Handle datetime objects (ISO8601 string)

    Args:
        obj: Any JSON-serializable object, including:
            - dict, list, tuple
            - str, int, float, bool, None
            - numpy arrays and scalars
            - datetime and date objects
            - sets (converted to sorted lists)

    Returns:
        UTF-8 encoded bytes of the canonical JSON representation.

    Raises:
        ValueError: If the object contains NaN or Inf float values.
        TypeError: If the object contains types that cannot be serialized.

    Example:
        >>> canonical_json({"b": 2, "a": 1})
        b'{"a":1,"b":2}'
        >>> canonical_json([3, 2, 1])  # Lists preserve order
        b'[3,2,1]'
        >>> canonical_json({"x": 1.23456789012345})  # Float rounding
        b'{"x":1.2345678901}'
    """
    encoder = CanonicalEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    json_str = encoder.encode(obj)
    return json_str.encode("utf-8")


def canonical_hash(obj: Any) -> str:
    """Return full SHA-256 hex digest of canonical JSON.

    Computes a deterministic hash of the canonical JSON representation
    of the object. The hash is stable across runs, machines, and Python
    versions for semantically equivalent data.

    Args:
        obj: Any JSON-serializable object (see canonical_json for details).

    Returns:
        64-character lowercase hexadecimal SHA-256 hash.

    Raises:
        ValueError: If the object contains NaN or Inf float values.
        TypeError: If the object contains types that cannot be serialized.

    Example:
        >>> hash1 = canonical_hash({"b": 2, "a": 1})
        >>> hash2 = canonical_hash({"a": 1, "b": 2})
        >>> hash1 == hash2  # Same hash despite different key order
        True
    """
    json_bytes = canonical_json(obj)
    return hashlib.sha256(json_bytes).hexdigest()


def canonical_hash_prefix(obj: Any, length: int = 12) -> str:
    """Return truncated SHA-256 hex prefix.

    Computes the canonical hash and returns only the first `length`
    characters. Useful for short identifiers, logging, and display.

    Args:
        obj: Any JSON-serializable object (see canonical_json for details).
        length: Number of hex characters to return (default: 12).
            Must be between 1 and 64 inclusive.

    Returns:
        Truncated hexadecimal hash prefix.

    Raises:
        ValueError: If length is not between 1 and 64.
        ValueError: If the object contains NaN or Inf float values.
        TypeError: If the object contains types that cannot be serialized.

    Example:
        >>> canonical_hash_prefix({"a": 1}, length=8)
        'a1b2c3d4'  # Example output (actual hash depends on data)
    """
    if not isinstance(length, int) or length < 1 or length > 64:
        raise ValueError(f"length must be between 1 and 64, got {length}")

    full_hash = canonical_hash(obj)
    return full_hash[:length]


# =============================================================================
# Tests (run with: python -m pytest src/astro_arc/utils/canonical.py -v)
# =============================================================================

if __name__ == "__main__":
    import sys

    def run_tests() -> int:
        """Run all tests and return exit code."""
        failed = 0
        passed = 0

        def test(name: str, condition: bool, msg: str = "") -> None:
            nonlocal failed, passed
            if condition:
                print(f"  PASS: {name}")
                passed += 1
            else:
                print(f"  FAIL: {name} - {msg}")
                failed += 1

        print("\n=== canonical_json tests ===")

        # Test basic types
        test("null", canonical_json(None) == b"null")
        test("true", canonical_json(True) == b"true")
        test("false", canonical_json(False) == b"false")
        test("int", canonical_json(42) == b"42")
        test("negative int", canonical_json(-42) == b"-42")
        test("string", canonical_json("hello") == b'"hello"')
        test("unicode", canonical_json("caf\u00e9") == '"caf\u00e9"'.encode())

        # Test sorted keys
        test("sorted keys", canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}')
        test(
            "nested sorted keys",
            canonical_json({"z": {"b": 2, "a": 1}, "a": 1}) == b'{"a":1,"z":{"a":1,"b":2}}',
        )

        # Test arrays preserve order
        test("array order preserved", canonical_json([3, 2, 1]) == b"[3,2,1]")
        test("nested array order", canonical_json({"arr": [3, 2, 1]}) == b'{"arr":[3,2,1]}')

        # Test no whitespace
        test("no whitespace", b" " not in canonical_json({"a": [1, 2, 3]}))
        test("no newlines", b"\n" not in canonical_json({"a": [1, 2, 3]}))

        # Test float handling
        test("float rounding", canonical_json(1.23456789012345) == b"1.2345678901")
        test("float to int", canonical_json(1.0) == b"1")
        test("negative float", canonical_json(-1.5) == b"-1.5")
        test("small float", canonical_json(0.0000000001) == b"1e-10")
        test("very small float", canonical_json(0.00000000001) == b"0")

        # Test NaN/Inf rejection
        try:
            canonical_json(float("nan"))
            test("nan rejected", False, "Should have raised ValueError")
        except ValueError as e:
            test("nan rejected", "NaN" in str(e))

        try:
            canonical_json(float("inf"))
            test("inf rejected", False, "Should have raised ValueError")
        except ValueError as e:
            test("inf rejected", "Inf" in str(e))

        try:
            canonical_json(float("-inf"))
            test("negative inf rejected", False, "Should have raised ValueError")
        except ValueError as e:
            test("negative inf rejected", "Inf" in str(e))

        # Test nested NaN/Inf
        try:
            canonical_json({"x": [1, float("nan"), 3]})
            test("nested nan rejected", False, "Should have raised ValueError")
        except ValueError as e:
            test("nested nan rejected", "NaN" in str(e))

        # Test datetime handling
        dt = datetime(2024, 1, 15, 12, 30, 45)
        test("datetime", canonical_json(dt) == b'"2024-01-15T12:30:45"')
        test("date", canonical_json(date(2024, 1, 15)) == b'"2024-01-15"')

        # Test set handling (converted to sorted list)
        result = canonical_json({"a", "c", "b"})
        test("set to sorted list", result == b'["a","b","c"]')

        # Test tuple handling (converted to list)
        test("tuple", canonical_json((1, 2, 3)) == b"[1,2,3]")

        # Test numpy handling
        print("\n=== numpy tests ===")
        arr = np.array([1, 2, 3])
        test("numpy array", canonical_json(arr) == b"[1,2,3]")

        arr_float = np.array([1.5, 2.5])
        test("numpy float array", canonical_json(arr_float) == b"[1.5,2.5]")

        test("numpy int64", canonical_json(np.int64(42)) == b"42")
        test("numpy float64", canonical_json(np.float64(1.5)) == b"1.5")
        test("numpy bool", canonical_json(np.bool_(True)) == b"true")

        # Test numpy nan/inf rejection
        try:
            canonical_json(np.array([1.0, np.nan, 3.0]))
            test("numpy nan rejected", False, "Should have raised ValueError")
        except ValueError as e:
            test("numpy nan rejected", "NaN" in str(e))

        try:
            canonical_json(np.inf)
            test("numpy inf rejected", False, "Should have raised ValueError")
        except ValueError as e:
            test("numpy inf rejected", "Inf" in str(e))

        print("\n=== canonical_hash tests ===")

        # Test hash consistency
        hash1 = canonical_hash({"b": 2, "a": 1})
        hash2 = canonical_hash({"a": 1, "b": 2})
        test("hash consistency", hash1 == hash2)
        test("hash length", len(hash1) == 64)
        test("hash hex", all(c in "0123456789abcdef" for c in hash1))

        # Test different data produces different hash
        hash3 = canonical_hash({"a": 1, "b": 3})
        test("different data different hash", hash1 != hash3)

        print("\n=== canonical_hash_prefix tests ===")

        # Test prefix length
        prefix = canonical_hash_prefix({"a": 1}, length=12)
        test("prefix length 12", len(prefix) == 12)

        prefix8 = canonical_hash_prefix({"a": 1}, length=8)
        test("prefix length 8", len(prefix8) == 8)

        # Test prefix is start of full hash
        full = canonical_hash({"a": 1})
        test("prefix matches hash start", full.startswith(prefix))

        # Test invalid lengths
        try:
            canonical_hash_prefix({"a": 1}, length=0)
            test("length 0 rejected", False, "Should have raised ValueError")
        except ValueError:
            test("length 0 rejected", True)

        try:
            canonical_hash_prefix({"a": 1}, length=65)
            test("length 65 rejected", False, "Should have raised ValueError")
        except ValueError:
            test("length 65 rejected", True)

        try:
            canonical_hash_prefix({"a": 1}, length=-1)
            test("negative length rejected", False, "Should have raised ValueError")
        except ValueError:
            test("negative length rejected", True)

        print("\n=== Edge cases ===")

        # Empty structures
        test("empty dict", canonical_json({}) == b"{}")
        test("empty list", canonical_json([]) == b"[]")
        test("empty string", canonical_json("") == b'""')

        # Deeply nested structures
        deep = {"a": {"b": {"c": {"d": [1, 2, 3]}}}}
        test("deep nesting", canonical_json(deep) == b'{"a":{"b":{"c":{"d":[1,2,3]}}}}')

        # Mixed types
        mixed = {"int": 1, "float": 1.5, "str": "a", "list": [1], "dict": {"x": 1}}
        result = canonical_json(mixed)
        test("mixed types", b'"dict"' in result and b'"float"' in result)

        # Large integers
        test("large int", canonical_json(10**20) == b"100000000000000000000")

        # Unicode strings
        test("unicode emoji", canonical_json("\U0001f600") == '"\U0001f600"'.encode())

        # Summary
        print(f"\n=== Summary: {passed} passed, {failed} failed ===")
        return 0 if failed == 0 else 1

    sys.exit(run_tests())
