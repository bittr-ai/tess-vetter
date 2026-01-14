# Open-Source Best Practices for Scientific Astronomy Packages

**Document:** Architecture Review 2 - Research Report 08
**Date:** 2026-01-14
**Focus:** Preparing bittr-tess-vetter for the astronomy open-source community

---

## Executive Summary

This report synthesizes best practices from PyOpenSci, the Astropy Project, JOSS, ASCL, and community research on open-sourcing scientific software. The goal is to provide a concrete roadmap for transitioning `bittr-tess-vetter` from an internal domain library to a community-ready, citable, and potentially Astropy-affiliated package.

---

## 1. PyOpenSci Guidelines

### Key Requirements

PyOpenSci's [Python Package Guide](https://www.pyopensci.org/python-package-guide/index.html) emphasizes five essential package components:

1. **User Documentation** - Clear tutorials, API reference, installation guide
2. **Code/API Implementation** - Clean, well-structured code
3. **Test Suite** - Comprehensive automated tests
4. **Contributor Documentation** - CONTRIBUTING.md, Code of Conduct
5. **Project Metadata** - Complete pyproject.toml, LICENSE, README

### Modern Packaging (2025)

PyOpenSci now recommends:
- **Hatch** as the build backend (already in use via hatchling)
- **UV** for fast dependency resolution (already in use)
- **Trusted Publishing** for PyPI security
- Modern pyproject.toml-only configuration

### Peer Review Process

PyOpenSci conducts open peer review. Packages accepted through their process:
- Gain credibility and discoverability
- Can be fast-tracked through JOSS
- Are listed in the [pyOpenSci ecosystem](https://www.pyopensci.org/python-packages.html)

**Current Status:** The package uses Hatch/UV but lacks formal documentation structure for pyOpenSci submission.

---

## 2. Astropy Affiliated Package Requirements

### Submission Path

Astropy affiliated packages are now submitted through pyOpenSci with [Astropy-specific review criteria](https://github.com/astropy/astropy-project/blob/main/affiliated/affiliated_package_review_guidelines.md).

### Astropy-Specific Criteria

| Criterion | Requirement | Current Status |
|-----------|-------------|----------------|
| **Astronomy Relevance** | Must be useful for astronomers beyond one project | TESS vetting is widely applicable |
| **Astropy Integration** | Use astropy units, coordinates, time where applicable | Uses astropy for FITS, coordinates |
| **No Namespace Collision** | Cannot use `astropy.*` namespace | Clean namespace |
| **PyPI Registration** | Must be on PyPI with proper metadata | Not yet published |
| **Documentation** | Sphinx docs with examples | None currently |

### Integration Expectations

Packages should use:
- `astropy.units` for physical quantities
- `astropy.coordinates` for celestial positions
- `astropy.time` for time handling (BTJD, etc.)
- `astropy.io.fits` for FITS I/O

**Gap:** Current code uses astropy internally but doesn't expose unit-aware APIs to users.

### Package Template

The [Astropy package template](https://github.com/astropy/package-template) and [OpenAstronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide) provide scaffolding for:
- Sphinx documentation with astropy theme
- GitHub Actions CI/CD
- RTD integration
- Changelog management

---

## 3. CITATION.cff Best Practices

### What It Is

[CITATION.cff](https://citation-file-format.github.io/) is a machine-readable file that tells users how to cite your software. GitHub, Zenodo, Zotero, and Software Heritage all parse it automatically.

### Essential Fields

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
type: software
title: "bittr-tess-vetter"
version: "0.1.0"
date-released: "2026-XX-XX"
license: MIT
repository-code: "https://github.com/xxx/bittr-tess-vetter"
authors:
  - family-names: "Author"
    given-names: "Name"
    orcid: "https://orcid.org/0000-0000-0000-0000"
keywords:
  - TESS
  - exoplanet
  - transit vetting
  - astronomy
preferred-citation:  # Optional: link to JOSS paper
  type: article
  ...
```

### Tools

- **cffinit**: [Web tool](https://citation-file-format.github.io/cff-initializer-javascript/) to create valid CITATION.cff
- **cffconvert**: Python CLI to validate and convert
- Zenodo auto-ingests CITATION.cff for DOI minting

### Integration Benefits

| Platform | Benefit |
|----------|---------|
| GitHub | "Cite this repository" button on landing page |
| Zenodo | Auto-populates metadata for releases |
| Zotero | Browser plugin imports correct citation |
| ADS | Indexed if registered with ASCL |
| Software Heritage | Permanent archive with citation support |

**Action:** Create CITATION.cff before first public release.

---

## 4. Community Standards: ASCL and JOSS

### ASCL (Astrophysics Source Code Library)

[ASCL](https://ascl.net/) is a registry indexed by ADS and Web of Science.

**Submission Requirements:**
- Code used in peer-reviewed astronomy publication
- Source available (GitHub, etc.)
- Description and author information
- Link to related publication

**Benefits:**
- ADS-indexed with unique `ascl.XXXX.XXXX` identifier
- Citable even without a dedicated paper
- Permanent record of software existence

**Process:** Submit via [ASCL form](https://ascl.net/submissions) or codemeta.json auto-import.

### JOSS (Journal of Open Source Software)

[JOSS](https://joss.theoj.org/) publishes short papers describing research software.

**Review Criteria ([full checklist](https://joss.readthedocs.io/en/latest/review_criteria.html)):**

| Category | Requirement |
|----------|-------------|
| **License** | OSI-approved license file in repo |
| **Development History** | 6+ months public history preferred |
| **Documentation** | README, installation, usage examples, API docs |
| **Tests** | Automated test suite with CI |
| **Functionality** | Reviewers verify core features work |
| **Code Size** | >1000 LOC expected, >300 LOC minimum |
| **Paper** | 250-1000 words with Statement of Need |

**AAS Partnership:**
- JOSS partners with AAS Journals
- Software reviewed by JOSS gets AAS endorsement badge
- Parallel review possible during AAS paper submission

**Scope for bittr-tess-vetter:**
- Clearly research-focused (TESS exoplanet vetting)
- Significant codebase (well over 1000 LOC)
- Novel contribution (integrated vetting pipeline)

---

## 5. Governance and Contribution Patterns for Small Packages

### Minimal Viable Governance

For small astronomy packages (1-3 maintainers), the recommended approach:

1. **CONTRIBUTING.md**: Clear contribution workflow (fork, branch, PR)
2. **CODE_OF_CONDUCT.md**: Adopt Astropy or Contributor Covenant
3. **Issue Templates**: Bug report, feature request
4. **PR Template**: Checklist for contributors
5. **CODEOWNERS**: Specify who reviews what

### Maintainer Sustainability

Common patterns that work:

| Pattern | Description |
|---------|-------------|
| **Bus Factor > 1** | Aim for 2+ people with merge rights |
| **Clear Scope** | Define what's in/out of scope explicitly |
| **Version Deprecation** | Document support timeline (e.g., latest 2 minor versions) |
| **Office Hours** | Optional: periodic times for community questions |
| **Transparent Archival** | If abandoning, archive repo rather than delete |

### Astropy Coordinated vs Affiliated

| Type | Governance | Control |
|------|------------|---------|
| **Affiliated** | Maintained by original authors | Authors retain full control |
| **Coordinated** | Maintained by Astropy Project | Astropy has admin control |

For `bittr-tess-vetter`, **affiliated** status is appropriate given bittr.ai ownership.

---

## 6. Common Pitfalls When Open-Sourcing Research Code

### The "Academics Doing Open Source Wrong" Problem

Common issues ([research](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009481)):

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Hardcoded Paths** | Paths to researcher's machine | Use relative paths, config files |
| **Missing Dependencies** | requirements.txt incomplete | Use lock files (uv.lock) |
| **No Pinned Versions** | "works on my machine" | Pin in pyproject.toml |
| **Missing Files** | Forgot to commit essential files | Pre-release checklist |
| **No Installation Docs** | Assumes user knows setup | Explicit installation guide |
| **No Examples** | API exists but no usage shown | Quickstart + tutorials |

### Code Quality Issues

| Issue | Error Rate | Mitigation |
|-------|------------|------------|
| Manual data manipulation | ~14% error rate | Automated pipelines |
| Untested edge cases | Variable | Property-based testing |
| Long-lived bugs | 15+ years unnoticed | Code review, CI |

### Best Practices from Literature

From "Ten Simple Rules for Clean Scientific Software":

1. **Modular Code**: Small functions with single responsibility
2. **Meaningful Names**: Variables/functions describe purpose
3. **Consistent Style**: Use formatters (ruff)
4. **Unit Tests**: Test each component independently
5. **Integration Tests**: Test workflows end-to-end
6. **CI/CD**: Automate testing on every commit
7. **Documentation**: Docstrings + user guide
8. **Version Control**: Git with meaningful commits
9. **Code Review**: All changes via PRs
10. **Reproducibility**: Lock files, containerization optional

---

## 7. Actionable Roadmap for bittr-tess-vetter

### Phase 1: Pre-Release Hygiene (Before First PyPI Release)

- [ ] **CITATION.cff**: Create using cffinit, validate with cffconvert
- [ ] **LICENSE**: Confirm MIT license file exists (currently only declared in pyproject.toml)
- [ ] **CODE_OF_CONDUCT.md**: Adopt Astropy or Contributor Covenant
- [ ] **CONTRIBUTING.md**: Document PR workflow, dev setup
- [ ] **codemeta.json**: Generate for ASCL/Zenodo compatibility
- [ ] **.github/ISSUE_TEMPLATE/**: Bug report, feature request
- [ ] **.github/PULL_REQUEST_TEMPLATE.md**: PR checklist

### Phase 2: Documentation (Required for JOSS/pyOpenSci)

- [ ] **docs/**: Sphinx documentation structure
  - [ ] Installation guide
  - [ ] Quickstart tutorial
  - [ ] API reference (autodoc from docstrings)
  - [ ] Vetting methodology explainer
- [ ] **RTD Integration**: readthedocs.yaml
- [ ] **Changelog**: CHANGELOG.md or HISTORY.rst

### Phase 3: Testing and CI (Required for JOSS)

- [ ] **Test Coverage**: Target >80% for public API
- [ ] **GitHub Actions**:
  - [ ] Test matrix (Python 3.11, 3.12)
  - [ ] Lint check
  - [ ] Doc build check
- [ ] **Codecov or Similar**: Track coverage over time

### Phase 4: Community Readiness

- [ ] **PyPI Publication**: First stable release
- [ ] **Zenodo Integration**: DOI for each release
- [ ] **ASCL Registration**: After first paper uses the code
- [ ] **pyOpenSci Submission**: After docs and tests complete
- [ ] **Astropy Affiliated**: Via pyOpenSci review pathway

### Phase 5: JOSS Paper (Optional but Recommended)

- [ ] **Paper Draft**: 250-1000 words
  - Statement of Need
  - Comparison to existing tools (lightkurve, TRICERATOPS, exovetter)
  - Research applications
- [ ] **Submit via JOSS**: ~4-8 week review timeline

---

## 8. Comparison with Related Packages

| Package | ASCL | JOSS | Astropy Affiliated | pyOpenSci |
|---------|------|------|-------------------|-----------|
| lightkurve | Yes (2018.12013) | Yes | Yes | - |
| TRICERATOPS | Yes (2010.021) | - | - | - |
| exovetter | - | - | - | - |
| transitleastsquares | Yes (1905.001) | Yes | - | - |
| **bittr-tess-vetter** | Not yet | Not yet | Not yet | Not yet |

---

## 9. Key Files to Create

### CITATION.cff (Template)

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
type: software
title: "bittr-tess-vetter"
abstract: "Domain library for TESS transit detection and vetting"
version: "0.1.0"
date-released: "2026-XX-XX"
license: MIT
repository-code: "https://github.com/bittr-ai/bittr-tess-vetter"
url: "https://bittr-tess-vetter.readthedocs.io"
keywords:
  - TESS
  - exoplanet
  - transit
  - vetting
  - false positive
  - astronomy
  - Python
authors:
  - name: "bittr-tess-vetter contributors"
```

### codemeta.json (Partial Template)

```json
{
  "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
  "@type": "SoftwareSourceCode",
  "name": "bittr-tess-vetter",
  "description": "Domain library for TESS transit detection and vetting",
  "license": "https://spdx.org/licenses/MIT",
  "codeRepository": "https://github.com/bittr-ai/bittr-tess-vetter",
  "programmingLanguage": "Python"
}
```

---

## 10. Summary of Priorities

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | CITATION.cff | 30 min | High - enables citation |
| **P0** | LICENSE file | 5 min | Required for any release |
| **P1** | CONTRIBUTING.md + CODE_OF_CONDUCT.md | 1 hour | Community trust |
| **P1** | GitHub issue/PR templates | 30 min | Contribution quality |
| **P2** | Sphinx docs skeleton | 4 hours | Required for JOSS/pyOpenSci |
| **P2** | GitHub Actions CI | 2 hours | Required for JOSS |
| **P3** | PyPI + Zenodo publication | 2 hours | Enables DOI and pip install |
| **P4** | ASCL registration | 30 min | ADS discoverability |
| **P5** | pyOpenSci/JOSS submission | Ongoing | Community validation |

---

## Sources

- [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/index.html)
- [pyOpenSci Peer Review](https://www.pyopensci.org/python-packages.html)
- [Astropy Affiliated Packages](https://www.astropy.org/affiliated/)
- [Astropy Affiliated Package Review Guidelines](https://github.com/astropy/astropy-project/blob/main/affiliated/affiliated_package_review_guidelines.md)
- [CITATION.cff Format](https://citation-file-format.github.io/)
- [The Turing Way - Software Citation](https://book.the-turing-way.org/communication/citable/citable-cff/)
- [Zenodo CITATION.cff Integration](https://help.zenodo.org/docs/github/describe-software/citation-file/)
- [ASCL Submissions](https://ascl.net/submissions)
- [JOSS Review Criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
- [JOSS Submission Guide](https://joss.readthedocs.io/en/latest/submitting.html)
- [AAS Policy on Software](https://journals.aas.org/policy-statement-on-software/)
- [Ten Simple Rules for Clean Scientific Software](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009481)
- [Academics: You're Doing Open Source Wrong](https://chanind.github.io/2023/06/04/academics-open-source-research-code-python-tips.html)
- [Astropy Package Template](https://github.com/astropy/package-template)
- [Lightkurve GitHub](https://github.com/lightkurve/lightkurve)
- [TESS Data Analysis Tools](https://heasarc.gsfc.nasa.gov/docs/tess/data-analysis-tools.html)
