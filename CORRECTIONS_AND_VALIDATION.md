# Corrections and Validation Status — 2.0.0rc2

This release candidate supersedes 2.0.0rc1 and fixes a critical real-data integration regression found by an external end-to-end review.

## Critical rc1 regression fixed

`fit_fourier_series()` intentionally stores `result["model"]` as a NumPy array evaluated at the input phases. `render_observational_summary()` in rc1 incorrectly treated that array as a callable, causing every non-synthetic analysis to fail at the final rendering stage with `TypeError: 'numpy.ndarray' object is not callable`.

The plotting module now reconstructs the Fourier model at arbitrary phases only through the public `evaluate_fourier()` function, using the fitted mean, cosine coefficients and sine coefficients. The stored `model` array remains unchanged for backward compatibility.

## New integration/regression coverage

- Added a plotting regression test that creates a real `fit_fourier_series()` result and passes it directly to `render_observational_summary()`.
- Added a full local-file CLI integration test covering local acquisition, preprocessing, Lomb–Scargle period recovery, phase folding, Fourier fitting, observational rendering, provenance output and `results.json`.
- Added an AAVSO XML-wrapped fixture with CSV inside a `<Data>`/CDATA section, matching the documented behavior of the public `api.object&...&csv` route more closely than the prior idealized plain-CSV-only mock.

## AAVSO parser hardening

The adapter now explicitly extracts CSV from the XML data element when the response is well-formed XML, while retaining a conservative fallback for legacy/non-well-formed wrapper responses. It continues to use header-driven column discovery and fails loudly when the observation schema cannot be recognized.

## Packaging correction

Lightkurve is no longer a mandatory core dependency. TESS support is lazy-loaded in code and is now matched by an optional `tess` install extra in `pyproject.toml`. `requirements-tess.txt` is provided for requirements-file workflows. Core AAVSO/OGLE/local/synthetic users therefore do not need to install Lightkurve.

## Real-data redesign retained

- Canonical `PhotometryData` with source, target, band, time system, magnitude system and provenance.
- Adapter architecture for AAVSO, OGLE, TESS and local files.
- Interactive source selection with explicit fallback after `TargetNotFound`.
- Raw normalized photometry plus metadata/provenance saved before analysis.
- Observational summary plots for real data; the synthetic radius animation is not used for observations.
- ASAS-SN remains local-file-only because the current automated client requires credentials and the legacy public form should not be scripted.

## Scientific/numerical safeguards retained

- Baseline-aware Lomb–Scargle discovery plus continuous dominant-peak refinement.
- Weighted linear Fourier fitting and explicit phase convention.
- δ Cep amplitude/phase morphology benchmark and extrema regression checks.
- MAD-based robust clipping and irregular-sampling-aware detrending.
- Explicit single-band degeneracy safeguards in physical inference.
- Split-Rhat MCMC diagnostics and honest non-convergence reporting.
- PL relation treated as a consistency diagnostic, not an independent validation of all physical parameters.

## Validation boundary

This environment does not contain Astropy, emcee or Lightkurve, so the dependency-backed end-to-end CLI test cannot be executed here. Tests that do not require those missing packages, including the exact observational-render regression and XML-wrapped AAVSO parser tests, pass locally. CI installs the declared scientific dependencies and runs the complete suite on Python 3.10–3.12.

A final `2.0.0` tag should additionally require network-enabled smoke tests against the live AAVSO, OGLE and TESS/MAST services because remote schemas and service behavior are outside the package's control.
