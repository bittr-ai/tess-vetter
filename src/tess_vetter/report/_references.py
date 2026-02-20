"""Deterministic reference metadata for report summary payloads.

This module keeps report-layer citation plumbing independent of api.* wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceEntry:
    """Typed bibliography entry used in report summary references."""

    key: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    citation: str | None = None
    notes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "citation": self.citation,
            "notes": list(self.notes),
            "tags": list(self.tags),
        }


REFERENCE_REGISTRY: dict[str, ReferenceEntry] = {
    "AGOL_2005": ReferenceEntry(
        key="AGOL_2005",
        title="On detecting terrestrial planets with timing of giant planet transits",
        authors=("Agol", "Steffen", "Sari", "Clarkson"),
        year=2005,
        venue="MNRAS",
        citation="Agol et al. 2005, MNRAS 359, 567",
        tags=("timing", "ttv"),
    ),
    "COUGHLIN_2016": ReferenceEntry(
        key="COUGHLIN_2016",
        title="Planetary candidates observed by Kepler. VIII. A fully automated catalog",
        authors=("Coughlin", "Mullally", "Thompson"),
        year=2016,
        venue="ApJS",
        citation="Coughlin et al. 2016, ApJS 224, 12",
        tags=("vetting", "odd_even"),
    ),
    "COUGHLIN_LOPEZ_MORALES_2012": ReferenceEntry(
        key="COUGHLIN_LOPEZ_MORALES_2012",
        title="A uniform search for secondary eclipses of hot Jupiters",
        authors=("Coughlin", "Lopez-Morales"),
        year=2012,
        venue="AJ",
        citation="Coughlin & Lopez-Morales 2012, AJ 143, 39",
        tags=("secondary", "eclipse"),
    ),
    "FORD_2012": ReferenceEntry(
        key="FORD_2012",
        title="Transit Timing Observations from Kepler",
        authors=("Ford", "Holman", "Fabrycky"),
        year=2012,
        venue="ApJ",
        citation="Ford et al. 2012, ApJ 750, 113",
        tags=("timing", "ttv"),
    ),
    "FRESSIN_2013": ReferenceEntry(
        key="FRESSIN_2013",
        title="The false positive rate of Kepler and occurrence of planets",
        authors=("Fressin", "Torres", "Charbonneau"),
        year=2013,
        venue="ApJ",
        citation="Fressin et al. 2013, ApJ 766, 81",
        tags=("false_positive", "secondary"),
    ),
    "GUERRERO_2021": ReferenceEntry(
        key="GUERRERO_2021",
        title="The TESS Objects of Interest Catalog from the TESS Prime Mission",
        authors=("Guerrero", "Seager", "Huang"),
        year=2021,
        venue="ApJS",
        citation="Guerrero et al. 2021, ApJS 254, 39",
        tags=("tess", "catalog"),
    ),
    "HOLMAN_MURRAY_2005": ReferenceEntry(
        key="HOLMAN_MURRAY_2005",
        title="The use of transit timing to detect terrestrial-mass extrasolar planets",
        authors=("Holman", "Murray"),
        year=2005,
        venue="Science",
        citation="Holman & Murray 2005, Science 307, 1288",
        tags=("timing", "ttv"),
    ),
    "IVSHINA_WINN_2022": ReferenceEntry(
        key="IVSHINA_WINN_2022",
        title="Transit timing methods for TESS planet candidates",
        authors=("Ivshina", "Winn"),
        year=2022,
        venue="ApJS",
        citation="Ivshina & Winn 2022, ApJS 259, 62",
        tags=("timing", "tess"),
    ),
    "PONT_2006": ReferenceEntry(
        key="PONT_2006",
        title="Effect of red noise on planetary transit detection",
        authors=("Pont", "Zucker", "Queloz"),
        year=2006,
        venue="MNRAS",
        citation="Pont et al. 2006, MNRAS 373, 231",
        tags=("noise", "red_noise"),
    ),
    "SEAGER_MALLEN_ORNELAS_2003": ReferenceEntry(
        key="SEAGER_MALLEN_ORNELAS_2003",
        title="On the unique solution of planet and star parameters from transit light curves",
        authors=("Seager", "Mallen-Ornelas"),
        year=2003,
        venue="ApJ",
        citation="Seager & Mallen-Ornelas 2003, ApJ 585, 1038",
        tags=("transit", "duration"),
    ),
    "THOMPSON_2018": ReferenceEntry(
        key="THOMPSON_2018",
        title="Planetary candidates observed by Kepler. DR25 catalog",
        authors=("Thompson", "Coughlin", "Hoffman"),
        year=2018,
        venue="ApJS",
        citation="Thompson et al. 2018, ApJS 235, 38",
        tags=("vetting", "kepler"),
    ),
    "TWICKEN_2018": ReferenceEntry(
        key="TWICKEN_2018",
        title="Kepler Data Validation I: architecture and diagnostic tests",
        authors=("Twicken", "Jenkins", "Seader"),
        year=2018,
        venue="PASP",
        citation="Twicken et al. 2018, PASP 130, 064502",
        tags=("vetting", "diagnostics"),
    ),
}


CHECK_METHOD_REFS: dict[str, tuple[str, ...]] = {
    "V01": ("COUGHLIN_2016", "THOMPSON_2018", "PONT_2006"),
    "V02": ("COUGHLIN_LOPEZ_MORALES_2012", "THOMPSON_2018", "FRESSIN_2013"),
    "V03": ("SEAGER_MALLEN_ORNELAS_2003", "GUERRERO_2021"),
    "V04": ("THOMPSON_2018", "PONT_2006"),
    "V05": ("THOMPSON_2018", "TWICKEN_2018"),
    "V13": ("THOMPSON_2018",),
    "V15": ("THOMPSON_2018",),
}


SUMMARY_METHOD_REFS: dict[str, tuple[str, ...]] = {
    "odd_even_summary": ("COUGHLIN_2016", "THOMPSON_2018"),
    "noise_summary": ("PONT_2006", "THOMPSON_2018"),
    "variability_summary": ("IVSHINA_WINN_2022", "FORD_2012", "HOLMAN_MURRAY_2005", "AGOL_2005"),
    "alias_scalar_summary": ("THOMPSON_2018",),
    "timing_summary": ("IVSHINA_WINN_2022", "FORD_2012", "HOLMAN_MURRAY_2005", "AGOL_2005"),
    "secondary_scan_summary": ("COUGHLIN_LOPEZ_MORALES_2012", "FRESSIN_2013", "THOMPSON_2018"),
    "data_gap_summary": ("THOMPSON_2018",),
}


def refs_for_check(check_id: str) -> list[str]:
    """Return ordered method references for a check ID."""
    return list(CHECK_METHOD_REFS.get(check_id, ()))


def refs_for_summary_block(block_name: str) -> list[str]:
    """Return ordered method references for a summary block."""
    return list(SUMMARY_METHOD_REFS.get(block_name, ()))


def reference_entries(reference_ids: set[str]) -> list[dict[str, object]]:
    """Return deduped reference entries sorted by reference key."""
    entries: list[dict[str, object]] = []
    for ref_id in sorted(reference_ids):
        entry = REFERENCE_REGISTRY.get(ref_id)
        if entry is None:
            entries.append({
                "key": ref_id,
                "title": None,
                "authors": [],
                "year": None,
                "venue": None,
                "doi": None,
                "url": None,
                "citation": None,
                "notes": [],
                "tags": [],
            })
        else:
            entries.append(entry.to_json())
    return entries


__all__ = [
    "CHECK_METHOD_REFS",
    "REFERENCE_REGISTRY",
    "SUMMARY_METHOD_REFS",
    "reference_entries",
    "refs_for_check",
    "refs_for_summary_block",
]
