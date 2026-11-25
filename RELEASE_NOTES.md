# Release notes — 2.0.0rc2

## Fixed

- Fixed the critical real-data rendering crash caused by treating the Fourier fit's stored NumPy `model` array as a callable.
- `render_observational_summary()` now evaluates fitted curves through `evaluate_fourier()`.
- Hardened AAVSO parsing for the public XML-wrapped response whose data section contains CSV observations.
- Aligned packaging with TESS lazy loading: Lightkurve is now an optional `tess` extra instead of a mandatory dependency.

## Tests added

- Observational plot integration regression using the actual `fit_fourier_series()` return object.
- End-to-end local-file CLI regression covering analysis through final `summary.png`/`results.json`.
- AAVSO XML + CDATA + CSV response fixture.

## Status

This remains a release candidate until the full dependency-backed suite and live remote-service smoke tests are executed in a network-enabled release environment.
