"""Central reference registry for bittr-tess-vetter.

This module is the single source of truth for all bibliographic references
cited by the vetting API. All citations are defined as typed constants
(THOMPSON_2018, COUGHLIN_2016, etc.) that can be imported and used directly.

Type Safety:
    Import Reference constants directly - do NOT use string-based lookups.
    This ensures pyright/mypy catch typos at static analysis time.

    # GOOD - type-safe, pyright catches typos at import time
    from bittr_tess_vetter.api.references import THOMPSON_2018, cites, cite

    @cites(cite(THOMPSON_2018, "§4.2 odd/even depth test"))
    def my_func(): ...

    # BAD - not type-safe, runtime failure on typo
    ref = get_reference("thopmson_2018")  # KeyError at runtime

Usage Examples:
    # Get all references
    >>> from bittr_tess_vetter.api.references import get_all_references
    >>> refs = get_all_references()
    >>> print(f"Total references: {len(refs)}")

    # Generate BibTeX for specific references
    >>> from bittr_tess_vetter.api.references import (
    ...     THOMPSON_2018, COUGHLIN_2016, generate_bibtex
    ... )
    >>> bibtex = generate_bibtex([THOMPSON_2018, COUGHLIN_2016])
    >>> print(bibtex)

    # Decorate functions with citations (with context)
    >>> from bittr_tess_vetter.api.references import THOMPSON_2018, cites, cite
    >>> @cites(cite(THOMPSON_2018, "§4.2 odd/even depth test"))
    ... def odd_even_depth(lc, ephemeris):
    ...     '''V01: Compare depth of odd vs even transits.'''
    ...     ...

    # Decorate functions with citations (without context)
    >>> @cites(cite(THOMPSON_2018))
    ... def simple_check(lc):
    ...     ...

    # Introspect function citations
    >>> from bittr_tess_vetter.api.references import get_function_references
    >>> refs = get_function_references(odd_even_depth)
    >>> for citation in refs:
    ...     print(f"{citation.ref.first_author_short}: {citation.context}")

    # Collect all citations from a module
    >>> from bittr_tess_vetter.api.references import collect_module_citations
    >>> import my_module
    >>> citations = collect_module_citations(my_module)
    >>> for func_name, cites in citations.items():
    ...     print(f"{func_name}: {[c.ref.first_author_short for c in cites]}")
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, NotRequired, Protocol, TypedDict, TypeVar

# =============================================================================
# Type Variables for Decorator
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# TypedDict for to_dict() return type
# =============================================================================


class ReferenceDict(TypedDict):
    """TypedDict for Reference.to_dict() return type."""

    id: str
    type: str
    title: str
    authors: list[str]
    year: int
    bibcode: NotRequired[str]
    journal: NotRequired[str]
    doi: NotRequired[str]
    arxiv: NotRequired[str]
    url: NotRequired[str]
    note: NotRequired[str]


# =============================================================================
# Reference Dataclass
# =============================================================================


@dataclass(frozen=True)
class Reference:
    """Immutable reference entry with full bibliographic metadata.

    Attributes:
        id: Unique identifier (e.g., "thompson_2018")
        title: Full paper title
        authors: Tuple of author names in "Last, First" format
        year: Publication year
        type: Reference type (article, book, software, dataset)
        bibcode: ADS bibcode (e.g., "2018ApJS..235...38T") - optional
        journal: Journal name with volume and page - optional
        doi: Digital Object Identifier (optional)
        arxiv: arXiv identifier (optional)
        url: URL for software/dataset/web resources (optional)
        note: Brief note about relevance to this package (optional)
    """

    id: str
    title: str
    authors: tuple[str, ...]
    year: int
    type: Literal["article", "book", "software", "dataset"] = "article"
    bibcode: str | None = None
    journal: str | None = None
    doi: str | None = None
    arxiv: str | None = None
    url: str | None = None
    note: str | None = None

    @property
    def ads_url(self) -> str | None:
        """URL to ADS abstract page, or None if no bibcode."""
        if self.bibcode:
            return f"https://ui.adsabs.harvard.edu/abs/{self.bibcode}"
        return None

    @property
    def first_author_short(self) -> str:
        """Short citation format: 'Thompson et al. 2018'."""
        first = self.authors[0].split(",")[0]
        suffix = " et al." if len(self.authors) > 1 else ""
        return f"{first}{suffix} {self.year}"

    def to_bibtex(self) -> str:
        """Generate BibTeX entry with proper type mapping."""
        # Map internal types to BibTeX types
        bibtex_type = {
            "article": "article",
            "book": "book",
            "software": "misc",
            "dataset": "misc",
        }.get(self.type, "misc")

        authors_str = " and ".join(self.authors)
        lines = [f"@{bibtex_type}{{{self.id},"]
        lines.append(f"    author = {{{authors_str}}},")
        lines.append(f"    title = {{{{{self.title}}}}},")

        if self.journal:
            lines.append(f"    journal = {{{self.journal}}},")
        lines.append(f"    year = {{{self.year}}},")

        if self.doi:
            lines.append(f"    doi = {{{self.doi}}},")
        if self.arxiv:
            lines.append(f"    eprint = {{{self.arxiv}}},")
            lines.append("    archiveprefix = {arXiv},")
        if self.url and self.type in ("software", "dataset"):
            lines.append(f"    howpublished = {{\\url{{{self.url}}}}},")

        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> ReferenceDict:
        """Convert to dictionary for JSON serialization."""
        result: ReferenceDict = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
        }
        if self.bibcode:
            result["bibcode"] = self.bibcode
        if self.journal:
            result["journal"] = self.journal
        if self.doi:
            result["doi"] = self.doi
        if self.arxiv:
            result["arxiv"] = self.arxiv
        if self.url:
            result["url"] = self.url
        if self.note:
            result["note"] = self.note
        return result


# =============================================================================
# Citation Wrapper (Reference + Context)
# =============================================================================


@dataclass(frozen=True)
class Citation:
    """A reference with optional context (section, page, reason)."""

    ref: Reference
    context: str | None = None  # e.g., "§4.2 odd/even depth test"


def cite(ref: Reference, context: str | None = None) -> Citation:
    """Helper to create a Citation with optional context.

    Args:
        ref: The Reference object to cite
        context: Optional context (e.g., "§4.2 odd/even depth test")

    Returns:
        A Citation wrapping the reference with context
    """
    return Citation(ref=ref, context=context)


# =============================================================================
# Protocol for Decorated Functions
# =============================================================================


class CitableCallable(Protocol):
    """Protocol for functions decorated with @cites.

    This protocol describes the shape of a function that has been decorated
    with @cites. The decorated function has a __references__ attribute
    containing the Citation tuple.
    """

    __references__: tuple[Citation, ...]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


# =============================================================================
# REGISTRY AND AUTO-REGISTRATION DECORATOR
# =============================================================================

# Internal registry mapping string IDs to Reference objects
# Populated automatically by the reference() decorator
_REGISTRY: dict[str, Reference] = {}


def reference(ref: Reference) -> Reference:
    """Auto-register a Reference constant.

    Use this decorator when defining Reference constants to automatically
    add them to the internal registry, eliminating manual _REGISTRY sync.

    Example:
        THOMPSON_2018 = reference(Reference(
            id="thompson_2018",
            ...
        ))
    """
    _REGISTRY[ref.id] = ref
    return ref


# =============================================================================
# CENTRAL REGISTRY - All 52 unique references
# =============================================================================

# -----------------------------------------------------------------------------
# Core Kepler/TESS vetting references (used across multiple modules)
# -----------------------------------------------------------------------------

THOMPSON_2018 = reference(Reference(
    id="thompson_2018",
    bibcode="2018ApJS..235...38T",
    title=(
        "Planetary Candidates Observed by Kepler. VIII. A Fully Automated "
        "Catalog With Measured Completeness and Reliability Based on Data Release 25"
    ),
    authors=("Thompson, S.E.", "Coughlin, J.L.", "Hoffman, K."),
    journal="ApJS 235, 38",
    year=2018,
    doi="10.3847/1538-4365/aab4f9",
    note="DR25 Robovetter: odd/even, secondary eclipse, V-shape, ModShift, SWEET",
))

COUGHLIN_2016 = reference(Reference(
    id="coughlin_2016",
    bibcode="2016ApJS..224...12C",
    title=(
        "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform "
        "Catalog Based on the Entire 48-month Data Set (Q1-Q17 DR24)"
    ),
    authors=("Coughlin, J.L.", "Mullally, F.", "Thompson, S.E."),
    journal="ApJS 224, 12",
    year=2016,
    doi="10.3847/0067-0049/224/1/12",
    note="Kepler Robovetter methodology for automated vetting",
))

GUERRERO_2021 = reference(Reference(
    id="guerrero_2021",
    bibcode="2021ApJS..254...39G",
    title="The TESS Objects of Interest Catalog from the TESS Prime Mission",
    authors=("Guerrero, N.M.", "Seager, S.", "Huang, C.X."),
    journal="ApJS 254, 39",
    year=2021,
    doi="10.3847/1538-4365/abefe1",
    note="TESS TOI catalog and vetting procedures",
))

TWICKEN_2018 = reference(Reference(
    id="twicken_2018",
    bibcode="2018PASP..130f4502T",
    title=(
        "Kepler Data Validation I -- Architecture, Diagnostic Tests, and "
        "Data Products for Vetting Transiting Planet Candidates"
    ),
    authors=("Twicken, J.D.", "Catanzarite, J.H.", "Clarke, B.D."),
    journal="PASP 130, 064502",
    year=2018,
    doi="10.1088/1538-3873/aab694",
    note="Kepler DV pipeline diagnostic tests for transit validation",
))

SEAGER_MALLEN_ORNELAS_2003 = reference(Reference(
    id="seager_mallen_ornelas_2003",
    bibcode="2003ApJ...585.1038S",
    title=(
        "On the Unique Solution of Planet and Star Parameters from an "
        "Extrasolar Planet Transit Light Curve"
    ),
    authors=("Seager, S.", "Mallen-Ornelas, G."),
    journal="ApJ 585, 1038",
    year=2003,
    doi="10.1086/346105",
    arxiv="astro-ph/0206228",
    note="Transit duration-stellar density relationship; transit shape analysis",
))

PRSA_2011 = reference(Reference(
    id="prsa_2011",
    bibcode="2011AJ....141...83P",
    title=(
        "Kepler Eclipsing Binary Stars. I. Catalog and Principal "
        "Characterization of 1879 Eclipsing Binaries in the First Data Release"
    ),
    authors=("Prsa, A.", "Batalha, N.", "Slawson, R.W."),
    journal="AJ 141, 83",
    year=2011,
    doi="10.1088/0004-6256/141/3/83",
    note="EB morphology classification; V-shape vs U-shape distinction",
))

# -----------------------------------------------------------------------------
# LC-only check references (lc_only.py)
# -----------------------------------------------------------------------------

PONT_2006 = reference(Reference(
    id="pont_2006",
    bibcode="2006MNRAS.373..231P",
    title="The effect of red noise on planetary transit detection",
    authors=("Pont, F.", "Zucker, S.", "Queloz, D."),
    journal="MNRAS 373, 231",
    year=2006,
    doi="10.1111/j.1365-2966.2006.11012.x",
    arxiv="astro-ph/0608597",
    note="Time-correlated (red) noise in transit photometry; binning-based inflation heuristic",
))

COUGHLIN_LOPEZ_MORALES_2012 = reference(Reference(
    id="coughlin_lopez_morales_2012",
    bibcode="2012AJ....143...39C",
    title=("A Uniform Search for Secondary Eclipses of Hot Jupiters in Kepler Q2 Light Curves"),
    authors=("Coughlin, J.L.", "Lopez-Morales, M."),
    journal="AJ 143, 39",
    year=2012,
    doi="10.1088/0004-6256/143/2/39",
    note="Secondary eclipse detection methodology for hot Jupiters",
))

FRESSIN_2013 = reference(Reference(
    id="fressin_2013",
    bibcode="2013ApJ...766...81F",
    title="The False Positive Rate of Kepler and the Occurrence of Planets",
    authors=("Fressin, F.", "Torres, G.", "Charbonneau, D."),
    journal="ApJ 766, 81",
    year=2013,
    doi="10.1088/0004-637X/766/2/81",
    note="False positive scenarios including EBs and secondary eclipses",
))

SANTERNE_2013 = reference(Reference(
    id="santerne_2013",
    bibcode="2013A&A...557A.139S",
    title=(
        "The false positive rate of Kepler and the occurrence of planets: "
        "eclipsing binary contamination and rejection"
    ),
    authors=("Santerne, A.", "Fressin, F.", "Diaz, R.F."),
    journal="A&A 557, A139",
    year=2013,
    doi="10.1051/0004-6361/201321566",
    arxiv="1307.2003",
    note="Eccentric orbit secondary eclipse offsets from phase 0.5",
))

WANG_ESPINOZA_2023 = reference(Reference(
    id="wang_espinoza_2023",
    bibcode="2023arXiv231102154W",
    title="Per-transit depth fitting for improved transit depth stability diagnostics",
    authors=("Wang, J.", "Espinoza, N."),
    year=2023,
    arxiv="2311.02154",
    note="Per-transit box depth fitting methodology with local baselines",
))

# -----------------------------------------------------------------------------
# Catalog check references (catalog.py)
# -----------------------------------------------------------------------------

PRSA_2022 = reference(Reference(
    id="prsa_2022",
    bibcode="2022ApJS..258...16P",
    title=(
        "TESS Eclipsing Binary Stars. I. Short-cadence Observations of 4584 "
        "Eclipsing Binaries in Sectors 1-26"
    ),
    authors=("Prsa, A.", "Kochoska, A.", "Conroy, K.E."),
    journal="ApJS 258, 16",
    year=2022,
    doi="10.3847/1538-4365/ac324a",
    note="TESS-EB catalog for nearby eclipsing binary search (V06)",
))

# -----------------------------------------------------------------------------
# Pixel-level check references (pixel.py)
# -----------------------------------------------------------------------------

GREISEN_CALABRETTA_2002 = reference(Reference(
    id="greisen_calabretta_2002",
    bibcode="2002A&A...395.1061G",
    title="Representations of world coordinates in FITS",
    authors=("Greisen, E.W.", "Calabretta, M.R."),
    journal="A&A 395, 1061",
    year=2002,
    arxiv="astro-ph/0207407",
    note="FITS WCS general framework (Paper I); basis for WCS transforms in pixel utilities",
))

CALABRETTA_GREISEN_2002 = reference(Reference(
    id="calabretta_greisen_2002",
    bibcode="2002A&A...395.1077C",
    title="Representations of celestial coordinates in FITS",
    authors=("Calabretta, M.R.", "Greisen, E.W."),
    journal="A&A 395, 1077",
    year=2002,
    arxiv="astro-ph/0207413",
    note="FITS celestial WCS conventions (Paper II); projections and RA/Dec mappings",
))

ASTROPY_COLLAB_2013 = reference(Reference(
    id="astropy_collab_2013",
    bibcode="2013A&A...558A..33A",
    title="Astropy: A Community Python Package for Astronomy",
    authors=("Astropy Collaboration",),
    journal="A&A 558, A33",
    year=2013,
    doi="10.1051/0004-6361/201322068",
    arxiv="1307.6212",
    note="Software reference for astropy.wcs usage in WCS utilities",
))

BRYSON_2013 = reference(Reference(
    id="bryson_2013",
    bibcode="2013PASP..125..889B",
    title="Identification of Background False Positives from Kepler Data",
    authors=("Bryson, S.T.", "Jenkins, J.M.", "Gilliland, R.L."),
    journal="PASP 125, 889",
    year=2013,
    doi="10.1086/671767",
    note="Pixel-level diagnostics for identifying background false positives",
))

BATALHA_2010 = reference(Reference(
    id="batalha_2010",
    bibcode="2010ApJ...713L.109B",
    title="Selection, Prioritization, and Characteristics of Kepler Target Stars",
    authors=("Batalha, N.M.", "Borucki, W.J.", "Koch, D.G."),
    journal="ApJ 713, L109",
    year=2010,
    doi="10.1088/2041-8205/713/2/L109",
    note="Kepler target star selection and stellar classification methodology",
))

TORRES_2011 = reference(Reference(
    id="torres_2011",
    bibcode="2011ApJ...727...24T",
    title=(
        "Modeling Kepler Transit Light Curves as False Positives: "
        "Rejection of Blend Scenarios for Kepler-9, and Validation of Kepler-9 d, "
        "a Super-Earth-size Planet in a Multiple System"
    ),
    authors=("Torres, G.", "Fressin, F.", "Batalha, N.M."),
    journal="ApJ 727, 24",
    year=2011,
    doi="10.1088/0004-637X/727/1/24",
    note="Background blend detection and rejection methodology",
))

MULLALLY_2015 = reference(Reference(
    id="mullally_2015",
    bibcode="2015ApJS..217...31M",
    title=("Planetary Candidates Observed by Kepler VI: Planet Sample from Q1-Q16 (47 Months)"),
    authors=("Mullally, F.", "Coughlin, J.L.", "Thompson, S.E."),
    journal="ApJS 217, 31",
    year=2015,
    doi="10.1088/0067-0049/217/2/31",
    note="Kepler planet candidate catalog with vetting diagnostics",
))

HIGGINS_BELL_2022 = reference(Reference(
    id="higgins_bell_2022",
    bibcode="2022AJ....163..141H",
    title="Localizing Sources of Variability in Crowded TESS Photometry",
    authors=("Higgins, M.E.", "Bell, K.J."),
    journal="AJ 163, 141",
    year=2022,
    doi="10.3847/1538-3881/ac4617",
    arxiv="2204.06020",
    note="TESS-specific centroid localization methodology for crowded fields",
))

BRYSON_2010 = reference(Reference(
    id="bryson_2010",
    bibcode="2010ApJ...713L..97B",
    title="The Kepler Pixel Response Function",
    authors=("Bryson, S.T.", "Tenenbaum, P.", "Jenkins, J.M."),
    journal="ApJ 713, L97",
    year=2010,
    doi="10.1088/2041-8205/713/2/L97",
    arxiv="1001.0331",
    note="PRF methodology for sub-pixel centroid determination in Kepler",
))

# -----------------------------------------------------------------------------
# Transit model fitting references (transit_fit.py)
# -----------------------------------------------------------------------------

MANDEL_AGOL_2002 = reference(Reference(
    id="mandel_agol_2002",
    bibcode="2002ApJ...580L.171M",
    title="Analytic Light Curves for Planetary Transit Searches",
    authors=("Mandel, K.", "Agol, E."),
    journal="ApJ 580, L171",
    year=2002,
    doi="10.1086/345520",
    arxiv="astro-ph/0210099",
    note="Foundational analytic transit model with limb darkening",
))

KREIDBERG_2015 = reference(Reference(
    id="kreidberg_2015",
    bibcode="2015PASP..127.1161K",
    title="batman: BAsic Transit Model cAlculatioN in Python",
    authors=("Kreidberg, L.",),
    journal="PASP 127, 1161",
    year=2015,
    doi="10.1086/683602",
    arxiv="1507.08285",
    note="Python transit model package used for light curve computation",
))

FOREMAN_MACKEY_2013 = reference(Reference(
    id="foreman_mackey_2013",
    bibcode="2013PASP..125..306F",
    title="emcee: The MCMC Hammer",
    authors=("Foreman-Mackey, D.", "Hogg, D.W.", "Lang, D.", "Goodman, J."),
    journal="PASP 125, 306",
    year=2013,
    doi="10.1086/670067",
    arxiv="1202.3665",
    note="MCMC sampler used for posterior estimation",
))

CLARET_2018 = reference(Reference(
    id="claret_2018",
    bibcode="2018A&A...618A..20C",
    title="Limb and gravity-darkening coefficients for the TESS satellite",
    authors=("Claret, A.",),
    journal="A&A 618, A20",
    year=2018,
    doi="10.1051/0004-6361/201833060",
    arxiv="1804.10295",
    note="TESS-specific limb darkening coefficients from ATLAS/PHOENIX models",
))

PARVIAINEN_2015 = reference(Reference(
    id="parviainen_2015",
    bibcode="2015MNRAS.453.3821P",
    title="LDTk: Limb Darkening Toolkit",
    authors=("Parviainen, H.", "Aigrain, S."),
    journal="MNRAS 453, 3821",
    year=2015,
    doi="10.1093/mnras/stv1857",
    arxiv="1508.02634",
    note="Python package for computing custom limb darkening profiles",
))

ESPINOZA_JORDAN_2015 = reference(Reference(
    id="espinoza_jordan_2015",
    bibcode="2015MNRAS.450.1879E",
    title="Limb darkening and exoplanets: testing stellar model atmospheres",
    authors=("Espinoza, N.", "Jordan, A."),
    journal="MNRAS 450, 1879",
    year=2015,
    doi="10.1093/mnras/stv744",
    arxiv="1503.07020",
    note="Analysis of limb darkening biases in transit parameters",
))

ESPINOZA_JORDAN_2016 = reference(Reference(
    id="espinoza_jordan_2016",
    bibcode="2016MNRAS.457.3573E",
    title="Limb-darkening and exoplanets II: Choosing the Best Law",
    authors=("Espinoza, N.", "Jordan, A."),
    journal="MNRAS 457, 3573",
    year=2016,
    doi="10.1093/mnras/stw224",
    arxiv="1601.05485",
    note="Comparison of limb darkening laws for transit fitting",
))

SING_2010 = reference(Reference(
    id="sing_2010",
    bibcode="2010A&A...510A..21S",
    title="Stellar Limb-Darkening Coefficients for CoRot and Kepler",
    authors=("Sing, D.K.",),
    journal="A&A 510, A21",
    year=2010,
    doi="10.1051/0004-6361/200913675",
    arxiv="0912.2274",
    note="Limb darkening coefficients for space missions",
))

CLARET_SOUTHWORTH_2022 = reference(Reference(
    id="claret_southworth_2022",
    bibcode="2022A&A...664A..91C",
    title="Power-2 limb-darkening coefficients for multiple photometric systems",
    authors=("Claret, A.", "Southworth, J."),
    journal="A&A 664, A91",
    year=2022,
    doi="10.1051/0004-6361/202243820",
    arxiv="2206.11098",
    note="Power-2 law limb darkening coefficients including TESS",
))

# -----------------------------------------------------------------------------
# Transit timing references (timing.py)
# -----------------------------------------------------------------------------

HOLMAN_MURRAY_2005 = reference(Reference(
    id="holman_murray_2005",
    bibcode="2005Sci...307.1288H",
    title=("The Use of Transit Timing to Detect Terrestrial-Mass Extrasolar Planets"),
    authors=("Holman, M.J.", "Murray, N.W."),
    journal="Science 307, 1288",
    year=2005,
    doi="10.1126/science.1107822",
    note="Foundational TTV theory paper - perturbations from additional planets",
))

AGOL_2005 = reference(Reference(
    id="agol_2005",
    bibcode="2005MNRAS.359..567A",
    title="On detecting terrestrial planets with timing of giant planet transits",
    authors=("Agol, E.", "Steffen, J.", "Sari, R.", "Clarkson, W."),
    journal="MNRAS 359, 567",
    year=2005,
    doi="10.1111/j.1365-2966.2005.08922.x",
    note="TTV theory - sensitivity to perturbing planets",
))

LITHWICK_2012 = reference(Reference(
    id="lithwick_2012",
    bibcode="2012ApJ...761..122L",
    title="Extracting Planet Mass and Eccentricity from TTV Data",
    authors=("Lithwick, Y.", "Xie, J.", "Wu, Y."),
    journal="ApJ 761, 122",
    year=2012,
    doi="10.1088/0004-637X/761/2/122",
    arxiv="1207.4192",
    note="Analytic TTV formulae for near-resonant planet pairs",
))

HADDEN_LITHWICK_2016 = reference(Reference(
    id="hadden_lithwick_2016",
    bibcode="2017AJ....154....5H",
    title="Kepler Planet Masses and Eccentricities from TTV Analysis",
    authors=("Hadden, S.", "Lithwick, Y."),
    journal="AJ 154, 5",
    year=2017,
    doi="10.3847/1538-3881/aa71ef",
    arxiv="1611.03516",
    note="Uniform TTV analysis of Kepler multiplanet systems",
))

HADDEN_2019 = reference(Reference(
    id="hadden_2019",
    bibcode="2019AJ....158..146H",
    title="Prospects for TTV Detection and Dynamical Constraints with TESS",
    authors=("Hadden, S.", "Barclay, T.", "Payne, M.J.", "Holman, M.J."),
    journal="AJ 158, 146",
    year=2019,
    doi="10.3847/1538-3881/ab384c",
    arxiv="1811.01970",
    note="TTV yield predictions for TESS mission",
))

IVSHINA_WINN_2022 = reference(Reference(
    id="ivshina_winn_2022",
    bibcode="2022ApJS..259...62I",
    title="TESS Transit Timing of Hundreds of Hot Jupiters",
    authors=("Ivshina, E.S.", "Winn, J.N."),
    journal="ApJS 259, 62",
    year=2022,
    doi="10.3847/1538-4365/ac545b",
    arxiv="2202.03401",
    note="TESS transit timing database and methods",
))

STEFFEN_AGOL_2006 = reference(Reference(
    id="steffen_agol_2006",
    bibcode="2007MNRAS.374..941A",
    title="Developments in Planet Detection using Transit Timing Variations",
    authors=("Steffen, J.H.", "Agol, E."),
    journal="MNRAS 374, 941",
    year=2007,
    doi="10.1111/j.1365-2966.2006.11216.x",
    arxiv="astro-ph/0612442",
    note="TTV detection sensitivity and methods",
))

FORD_2012 = reference(Reference(
    id="ford_2012",
    bibcode="2012ApJ...750..113F",
    title=(
        "Transit Timing Observations from Kepler: VI. Potentially Interesting "
        "Candidate Systems from Fourier-based Statistical Tests"
    ),
    authors=("Ford, E.B.", "Ragozzine, D.", "Rowe, J.F."),
    journal="ApJ 750, 113",
    year=2012,
    doi="10.1088/0004-637X/750/2/113",
    arxiv="1201.1892",
    note="Kepler TTV detection methodology",
))

FABRYCKY_2012 = reference(Reference(
    id="fabrycky_2012",
    bibcode="2012ApJ...750..114F",
    title=(
        "Transit Timing Observations from Kepler: IV. Confirmation of 4 "
        "Multiple Planet Systems by Simple Physical Models"
    ),
    authors=("Fabrycky, D.C.", "Ford, E.B.", "Steffen, J.H."),
    journal="ApJ 750, 114",
    year=2012,
    doi="10.1088/0004-637X/750/2/114",
    arxiv="1201.5415",
    note="Multi-planet TTV confirmation methodology",
))

RAGOZZINE_HOLMAN_2019 = reference(Reference(
    id="ragozzine_holman_2019",
    bibcode="2010ApJ...711..772R",
    title=("The Value of Systems with Multiple Transiting Planets"),
    authors=("Ragozzine, D.", "Holman, M.J."),
    journal="ApJ 711, 772",
    year=2010,
    doi="10.1088/0004-637X/711/2/772",
    arxiv="1006.3727",
    note="Multi-transiting system analysis methodology",
))

# -----------------------------------------------------------------------------
# Stellar activity references (activity.py)
# -----------------------------------------------------------------------------

MCQUILLAN_2014 = reference(Reference(
    id="mcquillan_2014",
    bibcode="2014ApJS..211...24M",
    title=(
        "Rotation Periods of 34,030 Kepler Main-Sequence Stars: The Full Autocorrelation Sample"
    ),
    authors=("McQuillan, A.", "Mazeh, T.", "Aigrain, S."),
    journal="ApJS 211, 24",
    year=2014,
    doi="10.1088/0067-0049/211/2/24",
    arxiv="1402.5694",
    note="Autocorrelation-based rotation period measurement methodology",
))

MCQUILLAN_2013 = reference(Reference(
    id="mcquillan_2013",
    bibcode="2013MNRAS.432.1203M",
    title="Measuring the rotation period distribution of field M dwarfs with Kepler",
    authors=("McQuillan, A.", "Aigrain, S.", "Mazeh, T."),
    journal="MNRAS 432, 1203",
    year=2013,
    doi="10.1093/mnras/stt536",
    arxiv="1303.6787",
    note="ACF method development for rotation periods",
))

DAVENPORT_2016 = reference(Reference(
    id="davenport_2016",
    bibcode="2016ApJ...829...23D",
    title="The Kepler Catalog of Stellar Flares",
    authors=("Davenport, J.R.A.",),
    journal="ApJ 829, 23",
    year=2016,
    doi="10.3847/0004-637X/829/1/23",
    arxiv="1607.03494",
    note="Comprehensive flare detection methodology for Kepler",
))

DAVENPORT_2014 = reference(Reference(
    id="davenport_2014",
    bibcode="2014ApJ...797..122D",
    title="Multi-wavelength Characterization of Stellar Flares on Low-mass Stars",
    authors=("Davenport, J.R.A.",),
    journal="ApJ 797, 122",
    year=2014,
    doi="10.1088/0004-637X/797/2/122",
    arxiv="1510.05695",
    note="Empirical flare template and morphology",
))

BASRI_2013 = reference(Reference(
    id="basri_2013",
    bibcode="2013ApJ...769...37B",
    title=(
        "Photometric Variability in Kepler Target Stars III: "
        "Comparison with the Sun on Different Timescales"
    ),
    authors=("Basri, G.", "Walkowicz, L.", "Reiners, A."),
    journal="ApJ 769, 37",
    year=2013,
    doi="10.1088/0004-637X/769/1/37",
    arxiv="1304.0136",
    note="Stellar variability metrics and solar comparison",
))

NIELSEN_2013 = reference(Reference(
    id="nielsen_2013",
    bibcode="2013A&A...557L..10N",
    title="Rotation periods of 12,000 main-sequence Kepler stars",
    authors=("Nielsen, M.B.", "Gizon, L.", "Schunker, H.", "Karoff, C."),
    journal="A&A 557, L10",
    year=2013,
    doi="10.1051/0004-6361/201321912",
    arxiv="1305.5721",
    note="Alternative rotation period measurement approach",
))

REINHOLD_2020 = reference(Reference(
    id="reinhold_2020",
    bibcode="2020Sci...368..518R",
    title="The Sun is less active than other solar-like stars",
    authors=("Reinhold, T.", "Shapiro, A.I.", "Solanki, S.K."),
    journal="Science 368, 518",
    year=2020,
    doi="10.1126/science.aay3821",
    note="Solar activity in context of stellar variability",
))

DAVENPORT_2019 = reference(Reference(
    id="davenport_2019",
    bibcode="2019ApJ...871..241D",
    title="The Evolution of Flare Activity with Stellar Age",
    authors=("Davenport, J.R.A.", "Covey, K.R.", "Clarke, R.W."),
    journal="ApJ 871, 241",
    year=2019,
    doi="10.3847/1538-4357/aafb76",
    arxiv="1901.00890",
    note="Flare activity vs Rossby number and stellar age",
))

TOVAR_MENDOZA_2022 = reference(Reference(
    id="tovar_mendoza_2022",
    bibcode="2022ApJ...927...31T",
    title="Llamaradas Estelares: Modeling the Morphology of White-Light Flares",
    authors=("Tovar Mendoza, G.", "Davenport, J.R.A.", "Agol, E."),
    journal="ApJ 927, 31",
    year=2022,
    doi="10.3847/1538-4357/ac4584",
    arxiv="2205.05706",
    note="Improved analytic flare model",
))

# -----------------------------------------------------------------------------
# Transit recovery references (recovery.py)
# -----------------------------------------------------------------------------

HIPPKE_2019_WOTAN = reference(Reference(
    id="hippke_2019_wotan",
    bibcode="2019AJ....158..143H",
    title="Wotan: Comprehensive Time-series De-trending in Python",
    authors=("Hippke, M.", "David, T.J.", "Mulders, G.D.", "Heller, R."),
    journal="AJ 158, 143",
    year=2019,
    doi="10.3847/1538-3881/ab3984",
    arxiv="1906.00966",
    note="Stellar detrending methods benchmark and wotan package",
))

HIPPKE_HELLER_2019_TLS = reference(Reference(
    id="hippke_heller_2019_tls",
    bibcode="2019A&A...623A..39H",
    title="Optimized transit detection algorithm to search for periodic transits",
    authors=("Hippke, M.", "Heller, R."),
    journal="A&A 623, A39",
    year=2019,
    doi="10.1051/0004-6361/201834672",
    arxiv="1901.02015",
    note="TLS algorithm for transit detection after detrending",
))

BARROS_2020 = reference(Reference(
    id="barros_2020",
    bibcode="2020A&A...634A..75B",
    title=(
        "Improving transit characterisation with Gaussian process modelling of stellar variability"
    ),
    authors=("Barros, S.C.C.", "Demangeon, O.", "Diaz, R.F."),
    journal="A&A 634, A75",
    year=2020,
    doi="10.1051/0004-6361/201936086",
    arxiv="2001.07975",
    note="GP-based stellar variability modeling for transit recovery",
))

AIGRAIN_2016 = reference(Reference(
    id="aigrain_2016",
    bibcode="2016MNRAS.459.2408A",
    title="K2SC: Flexible systematics correction and detrending of K2 light curves",
    authors=("Aigrain, S.", "Parviainen, H.", "Pope, B.J.S."),
    journal="MNRAS 459, 2408",
    year=2016,
    doi="10.1093/mnras/stw706",
    note="GP-based systematics correction for K2",
))

PETIGURA_2012 = reference(Reference(
    id="petigura_2012",
    bibcode="2013ApJ...770...69P",
    title=("A Plateau in the Planet Population below Twice the Size of Earth"),
    authors=("Petigura, E.A.", "Marcy, G.W."),
    journal="ApJ 770, 69",
    year=2013,
    doi="10.1088/0004-637X/770/1/69",
    note="Spline-based detrending methodology",
))

LUGER_2016 = reference(Reference(
    id="luger_2016",
    bibcode="2016AJ....152..100L",
    title="EVEREST: Pixel Level Decorrelation of K2 Light Curves",
    authors=("Luger, R.", "Agol, E.", "Kruse, E."),
    journal="AJ 152, 100",
    year=2016,
    doi="10.3847/0004-6256/152/4/100",
    note="Pixel-level decorrelation for systematics removal",
))

KOVACS_2002 = reference(Reference(
    id="kovacs_2002",
    bibcode="2002A&A...391..369K",
    title="A box-fitting algorithm in the search for periodic transits",
    authors=("Kovacs, G.", "Zucker, S.", "Mazeh, T."),
    journal="A&A 391, 369",
    year=2002,
    doi="10.1051/0004-6361:20020802",
    note="Original BLS algorithm for transit detection",
))

SUNDARARAJAN_2017 = reference(Reference(
    id="sundararajan_2017",
    title="Axiomatic Attribution for Deep Networks",
    authors=("Sundararajan, M.", "Taly, A.", "Yan, Q."),
    year=2017,
    arxiv="1703.01365",
    url="https://arxiv.org/abs/1703.01365",
    note="Introduces Integrated Gradients attribution method",
))

MORVAN_2020 = reference(Reference(
    id="morvan_2020",
    bibcode="2020AJ....159..166M",
    title="Detrending Exoplanetary Transit Light Curves with Long Short-term Memory Networks",
    authors=("Morvan, M.", "Nikolaou, N.", "Tsiaras, A.", "Waldmann, I.P."),
    journal="AJ 159, 166",
    year=2020,
    doi="10.3847/1538-3881/ab7140",
    arxiv="2001.03370",
    note="Machine learning approach to detrending",
))

# -----------------------------------------------------------------------------
# TRICERATOPS references (triceratops_fpp.py)
# -----------------------------------------------------------------------------

GIACALONE_2021 = reference(Reference(
    id="giacalone_2021",
    bibcode="2021AJ....161...24G",
    title=(
        "Vetting of 384 TESS Objects of Interest with TRICERATOPS and "
        "Statistical Validation of 12 Planet Candidates"
    ),
    authors=("Giacalone, S.", "Dressing, C.D.", "Jensen, E.L.N."),
    journal="AJ 161, 24",
    year=2021,
    doi="10.3847/1538-3881/abc6af",
    arxiv="2002.00691",
    note="TRICERATOPS: Bayesian FPP calculation for transit candidates",
))

TRICERATOPS_PLUS = reference(Reference(
    id="triceratops_plus",
    title="TRICERATOPS+ multi-color transit validation",
    authors=("Barrientos, J.G.",),
    year=2025,
    arxiv="2508.02782",
    note="TRICERATOPS+ multi-color transit validation",
))

TRICERATOPS_PLUS_MULTIBAND = reference(Reference(
    id="triceratops_plus_multiband",
    title="TRICERATOPS+ multi-band photometry validation",
    authors=("Greklek-McKeon, M.",),
    year=2025,
    arxiv="2512.10007",
    note="TRICERATOPS+ multi-band photometry validation",
))


# =============================================================================
# REGISTRY ACCESS
# =============================================================================


def get_reference(ref_id: str) -> Reference:
    """Get a reference by string ID (for deserialization only).

    WARNING: Do NOT use this for the primary API. Import Reference constants
    directly for type safety:

        # BAD - not type-safe, runtime failure on typo
        ref = get_reference("thopmson_2018")  # KeyError at runtime

        # GOOD - type-safe, pyright catches typos at import time
        from bittr_tess_vetter.api.references import THOMPSON_2018
        ref = THOMPSON_2018

    This function is provided for:
        - Deserializing references from JSON/database storage
        - Legacy code migration
        - Dynamic reference lookup (rare)

    Args:
        ref_id: Reference identifier string (e.g., "thompson_2018")

    Returns:
        The Reference object with the given ID

    Raises:
        KeyError: If the reference ID is not found
    """
    return _REGISTRY[ref_id]


def get_all_references() -> list[Reference]:
    """Get all references in the registry, sorted by year then first author.

    Returns:
        List of all Reference objects, sorted chronologically
    """
    return sorted(_REGISTRY.values(), key=lambda r: (r.year, r.authors[0]))


def generate_bibtex(refs: list[Reference] | None = None) -> str:
    """Generate BibTeX for specified references, or all if None.

    Args:
        refs: List of Reference objects to include. If None, includes all.

    Returns:
        BibTeX-formatted string with all entries

    Example:
        # All references
        >>> bibtex = generate_bibtex()

        # Specific references (type-safe)
        >>> from bittr_tess_vetter.api.references import THOMPSON_2018, COUGHLIN_2016
        >>> bibtex = generate_bibtex([THOMPSON_2018, COUGHLIN_2016])
    """
    refs_to_export = refs if refs else get_all_references()
    return "\n\n".join(r.to_bibtex() for r in refs_to_export)


def generate_bibliography_markdown() -> str:
    """Generate a markdown bibliography page.

    Returns:
        Markdown-formatted bibliography with all references
    """
    lines = [
        "# Bibliography",
        "",
        "All references cited by bittr-tess-vetter, sorted by year.",
        "",
    ]

    for ref in get_all_references():
        lines.extend(
            [
                f"## {ref.first_author_short}",
                "",
                f"**{ref.title}**",
                "",
                f"_{', '.join(ref.authors)}_",
                "",
            ]
        )
        if ref.journal:
            lines.append(ref.journal)
            lines.append("")
        ads_url = ref.ads_url
        if ads_url:
            lines.append(f"[ADS]({ads_url})")
            lines.append("")
        if ref.note:
            lines.append(f"> {ref.note}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# FUNCTION DECORATOR (TYPE-SAFE)
# =============================================================================


def cites(*refs_or_citations: Reference | Citation) -> Callable[[F], F]:
    """Decorator to attach citations to a function (fully type-safe).

    Usage:
        from bittr_tess_vetter.api.references import THOMPSON_2018, COUGHLIN_2016, cites, cite

        # With context (preferred for documentation)
        @cites(cite(THOMPSON_2018, "§4.2 odd/even depth test"), cite(COUGHLIN_2016))
        def odd_even_depth(lc, ephemeris):
            '''V01: Compare depth of odd vs even transits.'''
            ...

        # Without context (backward compatible)
        @cites(THOMPSON_2018, COUGHLIN_2016)
        def simple_check(lc):
            ...

    The decorated function will have a __references__ attribute
    containing the Citation objects as a tuple.

    Type Safety:
        - pyright/mypy catch undefined reference names at static analysis time
        - IDE autocomplete works for all reference constants
        - No runtime KeyError from typos - errors caught before code runs

    Args:
        *refs_or_citations: One or more Reference or Citation objects to attach.
            Reference objects are auto-wrapped in Citation(ref=..., context=None).

    Returns:
        Decorator that attaches citations to the function

    Raises:
        TypeError: If any argument is not a Reference or Citation object
    """
    # Normalize all inputs to Citation objects
    normalized: list[Citation] = []
    for item in refs_or_citations:
        if isinstance(item, Citation):
            normalized.append(item)
        elif isinstance(item, Reference):
            normalized.append(Citation(ref=item, context=None))
        else:
            raise TypeError(
                f"cites() requires Reference or Citation objects, got {type(item).__name__!r}. "
                f"Use cite(ref, context) for context: @cites(cite(THOMPSON_2018, '§4.2'))"
            )
    citations = tuple(normalized)

    def decorator(func: F) -> F:
        func.__references__ = citations  # type: ignore[attr-defined]
        return func

    return decorator


def get_function_references(func: Callable[..., object]) -> list[Citation]:
    """Get citations attached to a function via @cites decorator.

    Type-safe: works with any callable decorated with @cites.

    Args:
        func: Function decorated with @cites

    Returns:
        List of Citation objects attached to the function,
        or empty list if function has no citations

    Example:
        @cites(cite(THOMPSON_2018, "§4.2"), cite(COUGHLIN_2016))
        def my_function(): ...

        citations = get_function_references(my_function)
        # citations = [Citation(THOMPSON_2018, "§4.2"), Citation(COUGHLIN_2016, None)]
    """
    return list(getattr(func, "__references__", ()))


# =============================================================================
# MODULE INTROSPECTION
# =============================================================================


def collect_module_citations(module: ModuleType) -> dict[str, tuple[Citation, ...]]:
    """Collect citations from functions defined in this module (not imports).

    Args:
        module: The module to scan for @cites decorated functions

    Returns:
        Dictionary mapping function names to their Citation tuples.
        Class methods are named as "ClassName.method_name".

    Example:
        >>> import my_module
        >>> citations = collect_module_citations(my_module)
        >>> for func_name, cites in citations.items():
        ...     print(f"{func_name}: {[c.ref.first_author_short for c in cites]}")
    """
    result: dict[str, tuple[Citation, ...]] = {}

    # Functions defined in this module
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__ and hasattr(obj, "__references__"):
            refs: tuple[Citation, ...] = obj.__references__  # pyright: ignore[reportFunctionMemberAccess]
            result[name] = refs

    # Methods from classes defined in this module
    for class_name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__:
            for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                if hasattr(method, "__references__"):
                    method_refs: tuple[Citation, ...] = method.__references__  # pyright: ignore[reportFunctionMemberAccess]
                    result[f"{class_name}.{method_name}"] = method_refs

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Dataclass and types
    "Reference",
    "ReferenceDict",
    "Citation",
    "cite",
    "CitableCallable",
    # Decorator
    "reference",
    "cites",
    # Registry functions
    "get_reference",
    "get_all_references",
    "generate_bibtex",
    "generate_bibliography_markdown",
    # Introspection
    "get_function_references",
    "collect_module_citations",
    # All Reference constants
    "THOMPSON_2018",
    "COUGHLIN_2016",
    "GUERRERO_2021",
    "TWICKEN_2018",
    "SEAGER_MALLEN_ORNELAS_2003",
    "PRSA_2011",
    "PONT_2006",
    "COUGHLIN_LOPEZ_MORALES_2012",
    "FRESSIN_2013",
    "SANTERNE_2013",
    "WANG_ESPINOZA_2023",
    "PRSA_2022",
    "GREISEN_CALABRETTA_2002",
    "CALABRETTA_GREISEN_2002",
    "ASTROPY_COLLAB_2013",
    "BRYSON_2013",
    "BATALHA_2010",
    "TORRES_2011",
    "MULLALLY_2015",
    "HIGGINS_BELL_2022",
    "BRYSON_2010",
    "MANDEL_AGOL_2002",
    "KREIDBERG_2015",
    "FOREMAN_MACKEY_2013",
    "CLARET_2018",
    "PARVIAINEN_2015",
    "ESPINOZA_JORDAN_2015",
    "ESPINOZA_JORDAN_2016",
    "SING_2010",
    "CLARET_SOUTHWORTH_2022",
    "HOLMAN_MURRAY_2005",
    "AGOL_2005",
    "LITHWICK_2012",
    "HADDEN_LITHWICK_2016",
    "HADDEN_2019",
    "IVSHINA_WINN_2022",
    "STEFFEN_AGOL_2006",
    "FORD_2012",
    "FABRYCKY_2012",
    "RAGOZZINE_HOLMAN_2019",
    "MCQUILLAN_2014",
    "MCQUILLAN_2013",
    "DAVENPORT_2016",
    "DAVENPORT_2014",
    "BASRI_2013",
    "NIELSEN_2013",
    "REINHOLD_2020",
    "DAVENPORT_2019",
    "TOVAR_MENDOZA_2022",
    "HIPPKE_2019_WOTAN",
    "HIPPKE_HELLER_2019_TLS",
    "BARROS_2020",
    "AIGRAIN_2016",
    "PETIGURA_2012",
    "LUGER_2016",
    "KOVACS_2002",
    "SUNDARARAJAN_2017",
    "MORVAN_2020",
    "GIACALONE_2021",
    "TRICERATOPS_PLUS",
    "TRICERATOPS_PLUS_MULTIBAND",
]


# =============================================================================
# CLI FOR GENERATING REFERENCES
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--markdown":
        print(generate_bibliography_markdown())
    elif len(sys.argv) > 1 and sys.argv[1] == "--bibtex":
        print(generate_bibtex())
    else:
        print(f"References: {len(get_all_references())}")
        print("Usage: python -m bittr_tess_vetter.api.references [--markdown|--bibtex]")
