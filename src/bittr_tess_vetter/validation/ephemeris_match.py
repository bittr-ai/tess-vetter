"""Ephemeris matching and indexing utilities (metrics-only).

This module provides a searchable index of known ephemerides (TOIs, EBs, artifacts)
with support for harmonic/subharmonic matching and optional sky-position filtering.

Design notes:
- This is intended to be reusable and open-safe: it contains no curated datasets.
- Host apps should provide curated ephemeris lists (JSON/CSV) separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

MatchClass = Literal["EPHEMERIS_MATCH_STRONG", "EPHEMERIS_MATCH_WEAK", "NONE"]


@dataclass
class EphemerisEntry:
    """An entry in the ephemeris index."""

    source_id: str
    source_type: str
    period: float
    t0: float
    duration_hours: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")


@dataclass
class EphemerisMatch:
    """A match result from the ephemeris index."""

    source_entry: EphemerisEntry
    harmonic_relation: str
    period_residual: float
    t0_residual: float
    match_score: float
    separation_arcsec: float | None = None


@dataclass
class EphemerisMatchResult:
    """Full match result for a candidate."""

    candidate_id: str
    period: float
    t0: float
    matches: list[EphemerisMatch]
    match_class: MatchClass
    best_match: EphemerisMatch | None
    provenance: dict[str, Any]


def wrap_t0(t0: float, period: float, reference_t0: float = 0.0) -> float:
    """Wrap t0 to [0, period) relative to a reference epoch."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return (t0 - reference_t0) % period


def compute_match_score(
    period_residual: float,
    t0_residual: float,
    *,
    period_weight: float = 1.0,
    t0_weight: float = 0.5,
) -> float:
    """Compute match score from residuals (higher is better, 0-1]."""
    import math

    sigma_period = 0.01
    sigma_t0 = 0.1

    period_term = math.exp(-0.5 * (period_residual / sigma_period) ** 2)
    t0_term = math.exp(-0.5 * (t0_residual / sigma_t0) ** 2)

    total_weight = period_weight + t0_weight
    return (period_weight * period_term + t0_weight * t0_term) / total_weight


def compute_harmonic_match(
    query_period: float,
    query_t0: float,
    entry_period: float,
    entry_t0: float,
    *,
    harmonics: list[int] | None = None,
    period_tolerance: float = 0.001,
    t0_tolerance: float = 0.05,
) -> tuple[str, float, float] | None:
    """Check for harmonic relationship between query and entry ephemerides.

    Returns:
        (harmonic_relation, period_residual, t0_residual) if a match is found,
        otherwise None. harmonic_relation is "query:entry" format (e.g. "2:1").
    """
    if harmonics is None:
        harmonics = [1, 2, 3]

    best_match: tuple[str, float, float] | None = None
    best_score = -1.0

    for k in harmonics:
        # k:1 (query_period = k * entry_period)
        scaled_entry_period = k * entry_period
        period_residual_k1 = abs(query_period - scaled_entry_period) / query_period

        if period_residual_k1 <= period_tolerance:
            query_phase = wrap_t0(query_t0, query_period, reference_t0=0.0)
            entry_phase = wrap_t0(entry_t0, query_period, reference_t0=0.0)

            phase_diff = abs(query_phase - entry_phase)
            t0_residual = min(phase_diff, query_period - phase_diff) / query_period

            if t0_residual <= t0_tolerance:
                score = compute_match_score(period_residual_k1, t0_residual)
                if score > best_score:
                    best_score = score
                    best_match = (f"{k}:1", period_residual_k1, t0_residual)

        # 1:k (query_period = entry_period / k)
        if k > 1:
            scaled_query_period = k * query_period
            period_residual_1k = abs(scaled_query_period - entry_period) / query_period

            if period_residual_1k <= period_tolerance:
                query_phase = wrap_t0(query_t0, entry_period, reference_t0=0.0)
                entry_phase = wrap_t0(entry_t0, entry_period, reference_t0=0.0)

                phase_diff = abs(query_phase - entry_phase)
                t0_residual = min(phase_diff, entry_period - phase_diff) / entry_period

                if t0_residual <= t0_tolerance:
                    score = compute_match_score(period_residual_1k, t0_residual)
                    if score > best_score:
                        best_score = score
                        best_match = (f"1:{k}", period_residual_1k, t0_residual)

    return best_match


def _compute_angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute angular separation between two sky positions in arcseconds."""
    import math

    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    delta_ra = ra2_rad - ra1_rad

    cos_dec1 = math.cos(dec1_rad)
    cos_dec2 = math.cos(dec2_rad)
    sin_dec1 = math.sin(dec1_rad)
    sin_dec2 = math.sin(dec2_rad)

    numerator = math.sqrt(
        (cos_dec2 * math.sin(delta_ra)) ** 2
        + (cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * math.cos(delta_ra)) ** 2
    )
    denominator = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * math.cos(delta_ra)

    separation_rad = math.atan2(numerator, denominator)
    return math.degrees(separation_rad) * 3600.0


class EphemerisIndex:
    """Searchable ephemeris index with harmonic support."""

    def __init__(self, entries: list[EphemerisEntry] | None = None) -> None:
        self._entries: list[EphemerisEntry] = list(entries) if entries else []
        self._period_sorted: list[int] = []
        self._periods: list[float] = []
        self._is_built: bool = False

    def add_entry(self, entry: EphemerisEntry) -> None:
        self._entries.append(entry)
        self._is_built = False

    def build_index(self) -> None:
        self._period_sorted = sorted(
            range(len(self._entries)), key=lambda i: self._entries[i].period
        )
        self._periods = [self._entries[i].period for i in self._period_sorted]
        self._is_built = True

    @property
    def entries(self) -> list[EphemerisEntry]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def query(
        self,
        period: float,
        t0: float,
        *,
        period_tolerance: float = 0.001,
        t0_tolerance: float = 0.05,
        harmonics: list[int] | None = None,
        max_results: int = 10,
        query_ra: float | None = None,
        query_dec: float | None = None,
    ) -> list[EphemerisMatch]:
        if harmonics is None:
            harmonics = [1, 2]

        if not self._is_built:
            self.build_index()

        matches: list[EphemerisMatch] = []

        for k in harmonics:
            target_period = period / k
            min_period = target_period * (1 - period_tolerance)
            max_period = target_period * (1 + period_tolerance)

            self._search_period_range(
                period,
                t0,
                min_period,
                max_period,
                period_tolerance,
                t0_tolerance,
                harmonics,
                matches,
                query_ra,
                query_dec,
            )

            if k > 1:
                target_period = period * k
                min_period = target_period * (1 - period_tolerance)
                max_period = target_period * (1 + period_tolerance)

                self._search_period_range(
                    period,
                    t0,
                    min_period,
                    max_period,
                    period_tolerance,
                    t0_tolerance,
                    harmonics,
                    matches,
                    query_ra,
                    query_dec,
                )

        seen: dict[str, EphemerisMatch] = {}
        for m in matches:
            source_id = m.source_entry.source_id
            if source_id not in seen or m.match_score > seen[source_id].match_score:
                seen[source_id] = m

        result = sorted(seen.values(), key=lambda m: m.match_score, reverse=True)
        return result[:max_results]

    def _search_period_range(
        self,
        query_period: float,
        query_t0: float,
        min_period: float,
        max_period: float,
        period_tolerance: float,
        t0_tolerance: float,
        harmonics: list[int],
        matches: list[EphemerisMatch],
        query_ra: float | None,
        query_dec: float | None,
    ) -> None:
        if not self._periods:
            return

        left_idx = bisect_left(self._periods, min_period)
        right_idx = bisect_right(self._periods, max_period)

        for sorted_idx in range(left_idx, right_idx):
            entry_idx = self._period_sorted[sorted_idx]
            entry = self._entries[entry_idx]

            match_result = compute_harmonic_match(
                query_period,
                query_t0,
                entry.period,
                entry.t0,
                harmonics=harmonics,
                period_tolerance=period_tolerance,
                t0_tolerance=t0_tolerance,
            )

            if match_result is None:
                continue

            harmonic_relation, period_residual, t0_residual = match_result
            score = compute_match_score(period_residual, t0_residual)

            separation_arcsec: float | None = None
            if (
                query_ra is not None
                and query_dec is not None
                and entry.ra_deg is not None
                and entry.dec_deg is not None
            ):
                separation_arcsec = _compute_angular_separation(
                    query_ra, query_dec, entry.ra_deg, entry.dec_deg
                )

            matches.append(
                EphemerisMatch(
                    source_entry=entry,
                    harmonic_relation=harmonic_relation,
                    period_residual=period_residual,
                    t0_residual=t0_residual,
                    match_score=score,
                    separation_arcsec=separation_arcsec,
                )
            )


def classify_matches(
    matches: list[EphemerisMatch],
    *,
    strong_threshold: float = 0.9,
    weak_threshold: float = 0.5,
) -> tuple[MatchClass, EphemerisMatch | None]:
    """Classify matches and return best match."""
    if not matches:
        return "NONE", None

    best_match = max(matches, key=lambda m: m.match_score)

    if best_match.match_score >= strong_threshold:
        return "EPHEMERIS_MATCH_STRONG", best_match
    if best_match.match_score >= weak_threshold:
        return "EPHEMERIS_MATCH_WEAK", best_match
    return "NONE", best_match


def run_ephemeris_matching(
    candidate_period: float,
    candidate_t0: float,
    index: EphemerisIndex,
    *,
    candidate_id: str = "",
    harmonics: list[int] | None = None,
    period_tolerance: float = 0.001,
    t0_tolerance: float = 0.05,
    strong_threshold: float = 0.9,
    weak_threshold: float = 0.5,
    candidate_ra: float | None = None,
    candidate_dec: float | None = None,
) -> EphemerisMatchResult:
    """Run full ephemeris matching for a candidate."""
    if harmonics is None:
        harmonics = [1, 2, 3]

    matches = index.query(
        candidate_period,
        candidate_t0,
        period_tolerance=period_tolerance,
        t0_tolerance=t0_tolerance,
        harmonics=harmonics,
        query_ra=candidate_ra,
        query_dec=candidate_dec,
    )

    match_class, best_match = classify_matches(
        matches,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
    )

    provenance = {
        "period_tolerance": period_tolerance,
        "t0_tolerance": t0_tolerance,
        "harmonics": harmonics,
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
        "index_size": len(index),
        "num_matches": len(matches),
    }

    return EphemerisMatchResult(
        candidate_id=candidate_id,
        period=candidate_period,
        t0=candidate_t0,
        matches=matches,
        match_class=match_class,
        best_match=best_match,
        provenance=provenance,
    )


def _entry_to_dict(entry: EphemerisEntry) -> dict[str, Any]:
    return asdict(entry)


def _entry_from_dict(d: dict[str, Any]) -> EphemerisEntry:
    return EphemerisEntry(
        source_id=d["source_id"],
        source_type=d["source_type"],
        period=d["period"],
        t0=d["t0"],
        duration_hours=d.get("duration_hours"),
        ra_deg=d.get("ra_deg"),
        dec_deg=d.get("dec_deg"),
        metadata=d.get("metadata", {}),
    )


def save_index(index: EphemerisIndex, path: Path) -> None:
    """Save ephemeris index to a JSON file."""
    data = {"version": "1.0", "entries": [_entry_to_dict(e) for e in index.entries]}

    content_str = json.dumps(data["entries"], sort_keys=True)
    content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    data["content_hash"] = content_hash

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_index(path: Path) -> EphemerisIndex:
    """Load ephemeris index from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    if "entries" not in data:
        raise ValueError(f"Invalid index file: missing 'entries' key in {path}")

    entries = [_entry_from_dict(d) for d in data["entries"]]
    index = EphemerisIndex(entries)
    index.build_index()
    return index


def build_index_from_csv(path: Path) -> EphemerisIndex:
    """Build ephemeris index from a CSV file."""
    required_columns = {"source_id", "source_type", "period", "t0"}
    entries: list[EphemerisEntry] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV file: {path}")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            source_id = row["source_id"]
            source_type = row["source_type"]
            period = float(row["period"])
            t0 = float(row["t0"])

            duration_hours: float | None = float(row["duration_hours"]) if row.get("duration_hours") else None
            ra_deg: float | None = float(row["ra_deg"]) if row.get("ra_deg") else None
            dec_deg: float | None = float(row["dec_deg"]) if row.get("dec_deg") else None

            known_columns = required_columns | {"duration_hours", "ra_deg", "dec_deg"}
            metadata = {k: v for k, v in row.items() if k not in known_columns and v}

            entries.append(
                EphemerisEntry(
                    source_id=source_id,
                    source_type=source_type,
                    period=period,
                    t0=t0,
                    duration_hours=duration_hours,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    metadata=metadata,
                )
            )

    index = EphemerisIndex(entries)
    index.build_index()
    return index


__all__ = [
    "EphemerisEntry",
    "EphemerisIndex",
    "EphemerisMatch",
    "EphemerisMatchResult",
    "MatchClass",
    "build_index_from_csv",
    "classify_matches",
    "compute_harmonic_match",
    "compute_match_score",
    "load_index",
    "run_ephemeris_matching",
    "save_index",
    "wrap_t0",
]

