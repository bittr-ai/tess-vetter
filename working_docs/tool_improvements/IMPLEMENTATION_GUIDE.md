# Implementation Guide: Vetting Check Improvements (V02, V04, V05, V08, V11, V12)

This guide provides actionable instructions for implementing improvements to the vetting checks. All agents must follow these patterns to ensure consistency, backward compatibility, and code quality.

## Table of Contents

1. [Code Structure](#1-code-structure)
2. [Citation Requirements](#2-citation-requirements)
3. [Backward Compatibility Rules](#3-backward-compatibility-rules)
4. [Testing Requirements](#4-testing-requirements)
5. [Common Utilities](#5-common-utilities)
6. [Output Field Standards](#6-output-field-standards)
7. [Implementation Checklist](#7-implementation-checklist)
8. [Priority Order](#8-priority-order)

---

## 1. Code Structure

### File Locations

| Check | File Path | Class/Function |
|-------|-----------|----------------|
| V02 (Secondary Eclipse) | `src/bittr_tess_vetter/validation/lc_checks.py` | `check_secondary_eclipse()` |
| V04 (Depth Stability) | `src/bittr_tess_vetter/validation/lc_checks.py` | `check_depth_stability()` |
| V05 (V-Shape) | `src/bittr_tess_vetter/validation/lc_checks.py` | `check_v_shape()` |
| V08 (Centroid Shift) | `src/bittr_tess_vetter/validation/lc_checks.py` | `check_centroid_shift()` |
| V11 (Modshift) | `src/bittr_tess_vetter/validation/exovetter_checks.py` | `ModshiftCheck` class |
| V12 (SWEET) | `src/bittr_tess_vetter/validation/exovetter_checks.py` | `SWEETCheck` class |

### Function Signature Pattern (LC Checks)

LC checks in `lc_checks.py` use standalone functions with this signature:

```python
def check_<name>(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
    config: <Name>Config | None = None,  # Add if not present
) -> VetterCheckResult:
    """V##: <description>.

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        config: Optional configuration overrides

    Returns:
        VetterCheckResult with pass/fail status and details
    """
```

### Class Pattern (Exovetter Checks)

Exovetter checks use classes inheriting from `VetterCheck`:

```python
class <Name>Check(VetterCheck):
    """V##: <description>.

    Astronomical Significance:
    --------------------------
    <Explain why this check matters astronomically>

    Pass Criteria:
    - <criterion 1>
    - <criterion 2>

    Confidence Calculation:
    - <how confidence is determined>
    """

    id = "V##"
    name = "<snake_case_name>"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default configuration."""
        return CheckConfig(
            enabled=True,
            threshold=<default>,
            additional={
                "<key>": <value>,
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Execute the check."""
```

### Adding New Fields to VetterCheckResult.details

Fields are added via the `details` dict (additive only):

```python
return VetterCheckResult(
    id="V##",
    name="<name>",
    passed=passed,
    confidence=round(confidence, 3),
    details={
        # PRESERVE existing keys
        "existing_key": existing_value,

        # ADD new keys (snake_case, units in name if applicable)
        "new_metric_ppm": round(value * 1e6, 1),
        "new_ratio": round(ratio, 4),
        "warnings": warnings,  # Always include warnings list
    },
)
```

### Adding a Config Dataclass

If the check does not have a config dataclass, add one:

```python
@dataclass
class <Name>Config:
    """Configuration for V## <name> check.

    Attributes:
        <attr>: <description> (default: <value>)
    """
    threshold: float = 3.0
    secondary_threshold: float = 0.5
    # ... additional parameters with defaults
```

---

## 2. Citation Requirements

### Reference Framework Location

All citations are defined in `src/bittr_tess_vetter/api/references.py`.

### Adding a New Reference

1. **Add the Reference constant** in the appropriate section of `references.py`:

```python
# In the relevant section (e.g., "LC-only check references")
NEW_AUTHOR_YEAR = Reference(
    id="new_author_year",
    bibcode="2024ApJ...XXX...YY",  # ADS bibcode
    title="Full Paper Title Here",
    authors=("Author, A.", "Coauthor, B."),
    journal="ApJ XXX, YY",
    year=2024,
    doi="10.XXXX/YYYY",
    arxiv="2401.12345",  # Optional
    note="Brief note about relevance to this check",
)
```

2. **Add to the registry** in `_REGISTRY`:

```python
_REGISTRY: dict[str, Reference] = {
    # ... existing entries
    "new_author_year": NEW_AUTHOR_YEAR,
}
```

3. **Add to `__all__`**:

```python
__all__ = [
    # ... existing exports
    "NEW_AUTHOR_YEAR",
]
```

### Using Citations in Code

**Module docstring** (preferred for foundational references):

```python
"""Module description.

References:
- Thompson et al. (2018) - Kepler DR25 Robovetter methodology
- Coughlin et al. (2014) - Modshift technique for EB detection
"""
```

**Function/class docstring** (for specific implementations):

```python
def check_v_shape(...) -> VetterCheckResult:
    """V05: V-shape analysis.

    References:
        - Prsa et al. 2011 (EB morphology classification)
        - Seager & Mallen-Ornelas 2003 (transit shape theory)
    """
```

**Decorator** (for formal citation tracking):

```python
from bittr_tess_vetter.api.references import THOMPSON_2018, PRSA_2011, cites, cite

@cites(
    cite(THOMPSON_2018, "Section 4.3 V-shape metric"),
    cite(PRSA_2011, "EB morphology classification")
)
def check_v_shape(...):
    ...
```

### Required Citations per Check

| Check | Required Citations |
|-------|-------------------|
| V02 | Thompson et al. 2018 (secondary eclipse search), Coughlin & Lopez-Morales 2012 |
| V04 | Thompson et al. 2018 (depth stability), Pont et al. 2006 (correlated noise) |
| V05 | Prsa et al. 2011 (EB morphology), Seager & Mallen-Ornelas 2003 |
| V08 | Bryson et al. 2013 (centroid shift methodology) |
| V11 | Coughlin et al. 2014 (Modshift), Thompson et al. 2018 |
| V12 | Thompson et al. 2018 (SWEET test) |

---

## 3. Backward Compatibility Rules

### MUST NOT Change

1. **Function signatures** - Do not change existing parameter names, order, or types
2. **Check IDs** - V01, V02, etc. are permanent
3. **Check names** - "odd_even_depth", "secondary_eclipse", etc. are permanent
4. **Existing output field names** - Never rename existing keys in `details`
5. **Default behavior** - Existing thresholds must remain defaults

### CAN Change

1. **Add new fields** to `details` dict (additive)
2. **Add new config parameters** with defaults that preserve legacy behavior
3. **Add optional parameters** with `None` defaults
4. **Improve internal algorithms** as long as outputs remain compatible

### Gating New Behavior Behind Config Flags

When adding new capabilities that change behavior:

```python
@dataclass
class SecondaryEclipseConfig:
    """Configuration for V02 secondary eclipse check."""
    # EXISTING defaults (do not change)
    depth_threshold: float = 0.005
    sigma_threshold: float = 3.0

    # NEW: Add with default that preserves legacy behavior
    search_eccentric_phase: bool = False  # Default False = legacy behavior
    eccentric_phase_range: tuple[float, float] = (0.3, 0.7)

    # NEW: Red noise inflation (disabled by default for backward compat)
    use_red_noise_inflation: bool = False
```

Then in the function:

```python
def check_secondary_eclipse(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    config: SecondaryEclipseConfig | None = None,
) -> VetterCheckResult:
    if config is None:
        config = SecondaryEclipseConfig()  # Use defaults

    # Legacy behavior
    if not config.search_eccentric_phase:
        # ... existing phase 0.5 search
    else:
        # ... new eccentric phase search
```

### Preserving Legacy Output Keys

When a new metric replaces a legacy one, keep BOTH:

```python
details={
    # LEGACY keys (keep forever)
    "secondary_depth": round(depth, 6),
    "secondary_depth_sigma": round(sigma, 2),

    # NEW keys with more information
    "secondary_depth_ppm": round(depth * 1e6, 1),
    "secondary_depth_err_ppm": round(depth_err * 1e6, 1),
    "secondary_snr": round(snr, 2),
}
```

---

## 4. Testing Requirements

### Test File Location Pattern

Tests mirror the source structure:

```
src/bittr_tess_vetter/validation/lc_checks.py
    -> tests/validation/test_lc_checks.py (or test_<check_name>.py)

src/bittr_tess_vetter/validation/exovetter_checks.py
    -> tests/validation/test_exovetter_checks.py
```

### Required Test Categories

Every check improvement must include:

#### 1. Deterministic Unit Tests

Test with known inputs and expected outputs:

```python
def test_planet_like_signal_passes(self, make_synthetic_lc):
    """Planet-like signal should pass with high confidence."""
    lc, t0 = make_synthetic_lc(
        n_transits=20,
        depth_ppm=1000,
        noise_ppm=50,
    )
    result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

    assert result.passed is True
    assert result.confidence >= 0.7
    assert result.details["shape_ratio"] > 1.5
```

#### 2. Synthetic Data Tests

Test the detection capability with controlled signals:

```python
def test_eb_like_signal_fails(self, make_synthetic_lc):
    """EB-like V-shaped transit should fail."""
    lc, t0 = make_synthetic_lc(
        n_transits=20,
        depth_ppm=5000,
        shape="v_shape",  # Grazing EB
        noise_ppm=100,
    )
    result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

    assert result.passed is False
    assert result.confidence >= 0.7
```

#### 3. Edge Case Tests

```python
def test_insufficient_data_returns_pass_with_warning(self, make_synthetic_lc):
    """Insufficient data should return low-confidence pass."""
    lc, t0 = make_synthetic_lc(n_transits=1, depth_ppm=1000)
    result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

    assert result.passed is True
    assert result.confidence <= 0.3
    assert len(result.details.get("warnings", [])) > 0

def test_nan_handling(self, make_synthetic_lc):
    """Should handle NaN values gracefully."""
    lc, t0 = make_synthetic_lc(n_transits=10, depth_ppm=1000)
    lc.flux[100:110] = np.nan
    result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

    assert result.passed is not None  # Should not crash
```

#### 4. Backward Compatibility Tests

```python
def test_legacy_keys_present(self, make_synthetic_lc):
    """Legacy output keys must still be present."""
    lc, t0 = make_synthetic_lc(n_transits=10, depth_ppm=1000)
    result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

    # Legacy keys
    assert "depth_bottom" in result.details
    assert "depth_edge" in result.details
    assert "shape_ratio" in result.details

def test_default_config_preserves_behavior(self, make_synthetic_lc):
    """Default config should match legacy behavior."""
    lc, t0 = make_synthetic_lc(n_transits=10, depth_ppm=1000)

    result_default = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)
    result_explicit = check_v_shape(
        lc, period=5.0, t0=t0, duration_hours=3.0,
        config=VShapeConfig()  # Explicit default
    )

    assert result_default.passed == result_explicit.passed
```

### Creating Synthetic Light Curves

Use the fixture pattern from `tests/validation/test_odd_even_depth.py`:

```python
@pytest.fixture
def make_synthetic_lc():
    """Factory for deterministic synthetic light curves."""

    def _make(
        n_transits: int,
        depth_ppm: float,
        period: float = 5.0,
        duration_hours: float = 3.0,
        noise_ppm: float = 100.0,
        cadence_minutes: float = 2.0,
        seed: int = 42,
        # Check-specific parameters
        shape: str = "box",  # "box", "v_shape", "u_shape"
        secondary_depth_ppm: float = 0.0,
    ) -> tuple[LightCurveData, float]:
        rng = np.random.default_rng(seed)

        # ... time array generation ...

        # Generate appropriate transit shape
        if shape == "box":
            # Flat-bottomed (planet-like)
            flux[in_transit] -= depth_ppm * 1e-6
        elif shape == "v_shape":
            # V-shaped (grazing EB)
            # Linear ingress/egress with no flat bottom
            ...

        return LightCurveData(...), t0

    return _make
```

### Test Naming Conventions

```python
def test_<scenario>_<expected_outcome>(self):
    """<Scenario description> should <expected outcome>."""

# Examples:
def test_planet_like_equal_depths_passes(self):
def test_eb_like_deep_secondary_fails(self):
def test_insufficient_data_returns_low_confidence_pass(self):
def test_config_override_changes_threshold(self):
def test_legacy_keys_present_in_output(self):
```

### Running Tests

```bash
# Run tests for a specific check
uv run pytest tests/validation/test_<check_name>.py -v --tb=short

# Run with coverage
uv run pytest tests/validation/test_<check_name>.py -v --cov=src/bittr_tess_vetter/validation

# Run specific test class
uv run pytest tests/validation/test_v_shape.py::TestVShape -v

# Run specific test
uv run pytest tests/validation/test_v_shape.py::TestVShape::test_planet_like_passes -v
```

---

## 5. Common Utilities

### Utilities in `validation/base.py`

The following utilities are available and SHOULD be reused:

```python
from bittr_tess_vetter.validation.base import (
    phase_fold,              # Phase-fold a light curve
    get_in_transit_mask,     # Boolean mask for in-transit points
    get_out_of_transit_mask, # Boolean mask for out-of-transit points
    bin_phase_curve,         # Bin a phase-folded light curve
    sigma_clip,              # Sigma-clip an array
    measure_transit_depth,   # Measure depth from in/out flux
    count_transits,          # Count observable transits
    get_odd_even_transit_indices,  # Get odd/even transit indices
    search_secondary_eclipse,      # Search for secondary at given phase
)
```

### Computing In-Transit/Out-of-Transit Masks

```python
from bittr_tess_vetter.validation.base import get_in_transit_mask, get_out_of_transit_mask

# In-transit mask (within duration)
in_transit = get_in_transit_mask(
    time=lightcurve.time[lightcurve.valid_mask],
    period=period,
    t0=t0,
    duration_hours=duration_hours,
    buffer_factor=1.0,  # Exact duration
)

# Out-of-transit mask (excludes 2x duration around transit)
out_of_transit = get_out_of_transit_mask(
    time=lightcurve.time[lightcurve.valid_mask],
    period=period,
    t0=t0,
    duration_hours=duration_hours,
    buffer_factor=2.0,  # Exclude 2x duration
)
```

### Handling flux_err

Always check if `flux_err` is available:

```python
time = lightcurve.time[lightcurve.valid_mask]
flux = lightcurve.flux[lightcurve.valid_mask]

if lightcurve.flux_err is not None:
    flux_err = lightcurve.flux_err[lightcurve.valid_mask]
else:
    # Estimate from scatter
    flux_err = np.full_like(flux, np.std(flux))
```

### Red Noise Inflation Pattern

Red noise inflation is already implemented in `lc_checks.py`:

```python
from bittr_tess_vetter.validation.lc_checks import _compute_red_noise_inflation

# Compute inflation factor from OOT residuals
oot_residuals = flux[out_of_transit] - np.median(flux[out_of_transit])
oot_time = time[out_of_transit]

inflation_factor, success = _compute_red_noise_inflation(
    oot_residuals=oot_residuals,
    oot_time=oot_time,
    bin_size_days=duration_hours / 24.0 / 2,  # Half transit duration
)

if success:
    uncertainty *= inflation_factor
```

### Robust Statistics

```python
from bittr_tess_vetter.validation.lc_checks import _robust_std

# MAD-based robust standard deviation
robust_scatter = _robust_std(flux_values)
```

---

## 6. Output Field Standards

### Naming Conventions

- Use `snake_case` for all field names
- Include units in the name when applicable:
  - `_ppm` for parts per million (depth measurements)
  - `_sigma` for significance (number of standard deviations)
  - `_hours` for time durations
  - `_days` for orbital periods
  - `_ratio` for dimensionless ratios
- Boolean fields: use `is_` or `has_` prefix, or past tense verb (`passed`, `detected`)

```python
details = {
    "depth_ppm": 1000.0,           # Depth in ppm
    "depth_err_ppm": 50.0,         # Uncertainty in ppm
    "depth_sigma": 5.2,            # Detection significance
    "duration_ratio": 0.85,        # Dimensionless ratio
    "is_significant": True,        # Boolean with is_ prefix
    "secondary_detected": False,   # Boolean with past tense
}
```

### Required `warnings` List

Every check MUST include a `warnings` list (even if empty):

```python
warnings: list[str] = []

if n_transits < min_required:
    warnings.append(f"Only {n_transits} transits, need {min_required}")

if red_noise_factor > 2.0:
    warnings.append(f"High red noise (factor={red_noise_factor:.1f})")

# Always include in details
details["warnings"] = warnings
```

### Confidence Values

Confidence is a float in [0.0, 1.0] indicating certainty in the result:

| Confidence Range | Meaning |
|------------------|---------|
| 0.0 - 0.2 | Very low: Insufficient data, unable to assess |
| 0.2 - 0.5 | Low: Marginal data, result uncertain |
| 0.5 - 0.7 | Moderate: Adequate data, some uncertainty |
| 0.7 - 0.9 | High: Good data, reliable result |
| 0.9 - 0.95 | Very high: Excellent data, highly reliable |
| > 0.95 | Should be rare, reserved for overwhelming evidence |

Confidence calculation pattern:

```python
def _compute_confidence(
    n_transits: int,
    snr: float,
    has_warnings: bool,
    near_threshold: bool,
) -> float:
    # Base confidence from data quantity
    if n_transits < 2:
        base = 0.2
    elif n_transits < 5:
        base = 0.5
    elif n_transits < 10:
        base = 0.7
    else:
        base = 0.85

    # Boost for high SNR
    if snr > 10:
        base = min(0.95, base * 1.1)

    # Degrade if near threshold or warnings present
    if near_threshold:
        base *= 0.85
    if has_warnings:
        base *= 0.9

    return round(min(0.95, base), 3)
```

### Setting `passed`

- `passed = True`: Check does NOT indicate false positive
- `passed = False`: Check indicates likely false positive

Default to `passed = True` when:
- Insufficient data (cannot reject)
- Check errors/exceptions (fail-safe)

```python
# Fail-safe pattern
if len(data) < min_required:
    return VetterCheckResult(
        id="V##",
        name="check_name",
        passed=True,  # Cannot reject with insufficient data
        confidence=0.2,  # Low confidence
        details={"warnings": ["Insufficient data"]},
    )
```

---

## 7. Implementation Checklist

Use this checklist for each check improvement:

### Pre-Implementation

- [ ] Read the design document thoroughly
- [ ] Identify all new output fields
- [ ] Identify all new config parameters
- [ ] List required citations
- [ ] Review existing implementation

### Implementation

- [ ] Add config dataclass (if not present)
- [ ] Add new config parameters with backward-compatible defaults
- [ ] Implement algorithm changes
- [ ] Add all new output fields to `details`
- [ ] Preserve all legacy output fields
- [ ] Include `warnings` list in output
- [ ] Add citations to docstrings
- [ ] Add Reference constants to `api/references.py` (if new papers)

### Testing

- [ ] Create/update test file
- [ ] Add deterministic unit tests (planet-like passes, EB-like fails)
- [ ] Add synthetic data tests
- [ ] Add edge case tests (insufficient data, NaN handling)
- [ ] Add backward compatibility tests (legacy keys, default config)
- [ ] Run tests: `uv run pytest tests/validation/test_<name>.py -v`

### Code Quality

- [ ] Run linter: `uv run ruff check src/bittr_tess_vetter/validation/<file>.py --fix`
- [ ] Run formatter: `uv run ruff format src/bittr_tess_vetter/validation/<file>.py`
- [ ] Run type checker: `uv run pyright src/bittr_tess_vetter/validation/<file>.py`
- [ ] Run strict type checker: `uv run mypy src/bittr_tess_vetter/validation/<file>.py`

### Documentation

- [ ] Update function/class docstring
- [ ] Add inline comments for complex logic
- [ ] Document new config parameters

### Final Verification

- [ ] Run full validation tests: `uv run pytest tests/validation/ -v --tb=short`
- [ ] Verify backward compatibility: existing code still works unchanged
- [ ] Check that default behavior matches legacy behavior

---

## 8. Priority Order

### First: V05 (V-Shape) - CRITICAL BUG FIX

The V05 check has a critical bug where the trapezoidal model produces inverted results (U-shaped transits are flagged, V-shaped pass). This must be fixed first.

**Priority**: Immediate
**Reason**: Bug fix - current behavior is inverted from intended

### Then: Parallel Implementation

The remaining checks can be implemented in parallel:

| Check | Complexity | Dependencies |
|-------|------------|--------------|
| V02 (Secondary Eclipse) | Medium | None |
| V04 (Depth Stability) | Medium | None |
| V08 (Centroid Shift) | High | Requires TPF handling |
| V11 (Modshift) | Low | exovetter library |
| V12 (SWEET) | Low | exovetter library |

**Recommended parallel groupings**:
- **Agent A**: V02 + V04 (both in `lc_checks.py`, similar patterns)
- **Agent B**: V11 + V12 (both in `exovetter_checks.py`, similar patterns)
- **Agent C**: V08 (complex, requires TPF understanding)

---

## Quick Reference

### Import Template

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult
from bittr_tess_vetter.validation.base import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    phase_fold,
)

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters

logger = logging.getLogger(__name__)
```

### VetterCheckResult Template

```python
return VetterCheckResult(
    id="V##",
    name="check_name",
    passed=passed,
    confidence=round(confidence, 3),
    details={
        # Legacy keys (preserve)
        "legacy_key": legacy_value,

        # New keys (add)
        "new_metric_ppm": round(value, 1),
        "new_sigma": round(sigma, 2),
        "new_ratio": round(ratio, 4),

        # Always include
        "warnings": warnings,
        "method": "algorithm_name",
    },
)
```

### Test Template

```python
"""Tests for V## <name> check."""

import numpy as np
import pytest

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.lc_checks import check_<name>, <Name>Config


@pytest.fixture
def make_synthetic_lc():
    """Factory for synthetic light curves."""
    def _make(...) -> tuple[LightCurveData, float]:
        ...
        return lc, t0
    return _make


class Test<Name>:
    """Test suite for check_<name>."""

    def test_planet_like_passes(self, make_synthetic_lc):
        """Planet-like signal should pass."""
        ...

    def test_eb_like_fails(self, make_synthetic_lc):
        """EB-like signal should fail."""
        ...

    def test_insufficient_data_passes_with_warning(self, make_synthetic_lc):
        """Insufficient data should pass with low confidence."""
        ...

    def test_legacy_keys_present(self, make_synthetic_lc):
        """Legacy keys must be present."""
        ...

    def test_config_override(self, make_synthetic_lc):
        """Config overrides should work."""
        ...
```
