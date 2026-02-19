Tutorials
=========

This project includes runnable Jupyter notebooks in `docs/tutorials/` and a
few rendered tutorial pages in the Sphinx site.

Rendered tutorials
------------------

- :doc:`tutorials/blend_localization`
- :doc:`tutorials/10-toi-5807-check-by-check`
- :doc:`tutorials/tutorial_toi-5807-incremental/README`

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/10-toi-5807-check-by-check
   tutorials/tutorial_toi-5807-incremental/README

Notebook tutorials (download)
-----------------------------

These notebooks are intended to be executed locally (they are not rendered in
the Sphinx build by default).

- :download:`09-toi-5807-validation-walkthrough.ipynb <tutorials/09-toi-5807-validation-walkthrough.ipynb>` — recommended step-by-step statistical validation walkthrough for TOI-5807.01 (domain-language, traceable outputs).
- :download:`10-toi-5807-check-by-check.ipynb <tutorials/10-toi-5807-check-by-check.ipynb>` — check-by-check walkthrough for TOI-5807.01 (run each vetting check individually and inspect metrics).
- :download:`08-toi-5807-end-to-end.ipynb <tutorials/08-toi-5807-end-to-end.ipynb>` — recommended end-to-end TOI-5807.01 workflow (consolidates 04–07).
- :download:`00-quick-fp-kill.ipynb <tutorials/00-quick-fp-kill.ipynb>` — LC-only “false positive” demo (offline).
- :download:`01-basic-vetting.ipynb <tutorials/01-basic-vetting.ipynb>` — `LightCurve` + `Candidate` → `vet_candidate`.
- :download:`02-periodogram-detection.ipynb <tutorials/02-periodogram-detection.ipynb>` — `run_periodogram` → `Candidate` → vetting.
- :download:`03-pixel-analysis.ipynb <tutorials/03-pixel-analysis.ipynb>` — pixel diagnostics concepts and helpers.
- :download:`04-real-candidate-validation.ipynb <tutorials/04-real-candidate-validation.ipynb>` — legacy end-to-end workflow on TOI-5807.01 (includes AO-assisted FPP).
- :download:`05-extended-metrics.ipynb <tutorials/05-extended-metrics.ipynb>` — legacy extended metrics notebook for TOI-5807.01 (V16–V21).
- :download:`06-toi-5807-robustness.ipynb <tutorials/06-toi-5807-robustness.ipynb>` — legacy robustness notebook for TOI-5807.01.
- :download:`07-toi-5807-consolidated-validation.ipynb <tutorials/07-toi-5807-consolidated-validation.ipynb>` — legacy consolidated notebook for TOI-5807.01.
