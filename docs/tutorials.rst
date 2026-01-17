Tutorials
=========

This project includes runnable Jupyter notebooks in `docs/tutorials/` and a
few rendered tutorial pages in the Sphinx site.

Rendered tutorials
------------------

- :doc:`tutorials/blend_localization`

Notebook tutorials (download)
-----------------------------

These notebooks are intended to be executed locally (they are not rendered in
the Sphinx build by default).

- :download:`00-quick-fp-kill.ipynb <tutorials/00-quick-fp-kill.ipynb>` — LC-only “false positive” demo (offline).
- :download:`01-basic-vetting.ipynb <tutorials/01-basic-vetting.ipynb>` — `LightCurve` + `Candidate` → `vet_candidate`.
- :download:`02-periodogram-detection.ipynb <tutorials/02-periodogram-detection.ipynb>` — `run_periodogram` → `Candidate` → vetting.
- :download:`03-pixel-analysis.ipynb <tutorials/03-pixel-analysis.ipynb>` — pixel diagnostics concepts and helpers.
- :download:`04-real-candidate-validation.ipynb <tutorials/04-real-candidate-validation.ipynb>` — end-to-end validation workflow on a real TOI (includes AO-assisted FPP).
- :download:`05-extended-metrics.ipynb <tutorials/05-extended-metrics.ipynb>` — opt-in extended metrics checks (V16–V21) and how to compare against the default preset.
