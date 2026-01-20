# Tutorial Plot ↔ Narrative Review (2026-01-20)

Scope: `docs/tutorials/10-toi-5807-check-by-check.ipynb` plus pre-rendered plots under `docs/tutorials/artifacts/10-toi-5807-check-by-check/`.

## Summary

- Reviewed each embedded plot (V01–V15, excluding V14 which is not present in this tutorial).
- Adjusted the tutorial text to match what the plots actually show (especially “last sector” pixel diagnostics).
- Applied small plotting polish to reduce common visual/layout failure modes (ExoFOP notes overflow; sensitivity-sweep label width).

## Notes by Plot

- V01 Odd/Even depth: Plot matches section intent; odd/even means indicated with error bars + mean lines.
- V02 Secondary eclipse: Window shading matches narrative; “secondary depth” annotation reads as a null/low signal in the 0.4–0.6 phase window.
- V03 Duration consistency: Bar chart matches “observed vs expected” narrative; ratio annotation is visible.
- V04 Depth stability: Expected-scatter band and mean line correspond to the text; “dominating epoch” marker is visible.
- V05 Transit shape: Trapezoid overlay + binned data match the “U vs V” framing; note that a few bins are deeper than the trapezoid floor, consistent with the narrative’s “ingress/egress not well constrained at this depth/noise”.
- V06 Nearby EBs: Plot is consistent with “no nearby EBs found” when only the target marker + search radius are shown.
- V07 ExoFOP card: Card format matches narrative; long notes are now wrapped more conservatively to avoid horizontal overflow.
- V08 Centroid shift: Narrative already called out that the plotted stamp is “last sector”; this matches the figure.
- V09 Difference image: Tutorial now explicitly notes the plot is “last sector”, while results are reported per-sector.
- V10 Aperture dependence: Tutorial now explicitly notes the plot is “last sector”, while results are reported per-sector.
- V11 ModShift: Plot matches the narrative’s “secondary/other phase peak exists but is not necessarily at 0.5”.
- V12 SWEET: Plot matches “sinusoid fits at P / P/2 / 2P” framing; legend is readable.
- V13 Data gaps: Plot annotation includes both “Max missing” and “In coverage”, matching the written discussion.
- V15 Asymmetry: Left/right bins and σ annotation match the narrative.

