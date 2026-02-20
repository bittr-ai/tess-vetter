"""Verdict normalization utilities.

Provides robust normalization of verdict strings from various sources,
handling case variations, legacy aliases, and unknown values gracefully.
"""

from .contracts import Verdict

# Canonical verdict values
_CANONICAL_VERDICTS: frozenset[str] = frozenset(
    {"ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID", "NO_EVIDENCE"}
)

# Legacy aliases and common variants mapped to canonical values
_VERDICT_ALIASES: dict[str, Verdict] = {
    # Legacy aliases
    "UNAMBIGUOUS": "ON_TARGET",
    "CLEAR": "ON_TARGET",
    "CONFIRMED": "ON_TARGET",
    # Common variants for OFF_TARGET
    "OFFTARGET": "OFF_TARGET",
    "OFF-TARGET": "OFF_TARGET",
    "NOT_ON_TARGET": "OFF_TARGET",
    # Common variants for ON_TARGET
    "ONTARGET": "ON_TARGET",
    "ON-TARGET": "ON_TARGET",
    # Common variants for AMBIGUOUS
    "UNCERTAIN": "AMBIGUOUS",
    "UNCLEAR": "AMBIGUOUS",
    # Common variants for NO_EVIDENCE
    "NOEVIDENCE": "NO_EVIDENCE",
    "NO-EVIDENCE": "NO_EVIDENCE",
    "NONE": "NO_EVIDENCE",
    "UNKNOWN": "NO_EVIDENCE",
}


def normalize_verdict(
    value: str | None,
    *,
    track_unknown: list[str] | None = None,
) -> Verdict | None:
    """Normalize a verdict string to its canonical form.

    Args:
        value: The verdict string to normalize. Can be None or empty.
        track_unknown: Optional list to append unrecognized non-empty values to.
            Useful for debugging or logging unknown verdict strings encountered
            in production data.

    Returns:
        - None if value is None or empty string
        - Canonical Verdict if value matches a known verdict or alias (case insensitive)
        - "NO_EVIDENCE" if value is a non-empty unrecognized string

    Examples:
        >>> normalize_verdict(None)
        None
        >>> normalize_verdict("")
        None
        >>> normalize_verdict("ON_TARGET")
        'ON_TARGET'
        >>> normalize_verdict("on_target")
        'ON_TARGET'
        >>> normalize_verdict("UNAMBIGUOUS")
        'ON_TARGET'
        >>> normalize_verdict("garbage_string")
        'NO_EVIDENCE'
    """
    # Handle None and empty string
    if value is None or value == "":
        return None

    # Normalize to uppercase for comparison
    upper_value = value.upper()

    # Check if it's already a canonical verdict
    if upper_value in _CANONICAL_VERDICTS:
        # Type cast needed because upper_value is str, not Literal
        return upper_value  # type: ignore[return-value]

    # Check aliases
    if upper_value in _VERDICT_ALIASES:
        return _VERDICT_ALIASES[upper_value]

    # Unrecognized non-empty string: track if requested, return NO_EVIDENCE
    if track_unknown is not None:
        track_unknown.append(value)

    return "NO_EVIDENCE"


def is_valid_verdict(value: str | None) -> bool:
    """Check if a value is a valid canonical verdict.

    Args:
        value: The string to check.

    Returns:
        True if value is a valid canonical verdict (case insensitive), False otherwise.
    """
    if value is None:
        return False
    return value.upper() in _CANONICAL_VERDICTS


def get_verdict_aliases() -> dict[str, Verdict]:
    """Return a copy of the verdict aliases dictionary.

    Useful for inspection or extending the aliases in specialized contexts.
    """
    return dict(_VERDICT_ALIASES)
