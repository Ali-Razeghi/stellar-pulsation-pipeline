from types import SimpleNamespace
from pathlib import Path

import numpy as np

from stellar_pulsation.fourier_decomposition import (
    evaluate_fourier,
    fit_fourier_series,
    phase_fold,
)
from stellar_pulsation.observational_plots import render_observational_summary
from stellar_pulsation.photometry import PhotometryData


def _observed_like_dataset(period=5.3663, n=180):
    rng = np.random.default_rng(20260826)
    t = np.sort(rng.uniform(0.0, 55.0, n))
    phase = phase_fold(t, period)
    mag = (
        3.90
        + 0.32 * np.cos(2 * np.pi * phase + 0.5)
        + 0.13 * np.cos(4 * np.pi * phase + 5.36)
        + 0.05 * np.cos(6 * np.pi * phase + 3.98)
    )
    err = np.full(n, 0.02)
    mag = mag + rng.normal(0.0, err)
    return t, phase, mag, err


def test_observational_summary_accepts_actual_fourier_fit_result(tmp_path):
    """Regression test for the ndarray/callable integration contract bug."""
    t, phase, mag, err = _observed_like_dataset()
    fit = fit_fourier_series(phase, mag, err, n_harmonics=3)

    # The fit-result contract intentionally stores model values at input phases
    # as an ndarray; arbitrary-phase evaluation is done by evaluate_fourier().
    assert isinstance(fit["model"], np.ndarray)
    grid = np.linspace(0.0, 1.0, 100, endpoint=False)
    reconstructed = evaluate_fourier(
        grid, fit["mean"], fit["cosine_coeffs"], fit["sine_coeffs"]
    )
    assert reconstructed.shape == grid.shape

    data = PhotometryData(
        t,
        mag,
        err,
        target="Delta Cep test fixture",
        source="Local file",
        band="V",
        time_system="JD",
        magnitude_system="apparent",
    )
    ls_result = {
        "frequency_grid": np.linspace(0.1, 1.0, 100),
        "power": np.linspace(0.0, 1.0, 100),
        "best_period_days": 5.3663,
    }
    out = tmp_path / "observational_summary.png"
    returned = render_observational_summary(data, mag, ls_result, phase, fit, out)
    assert Path(returned).exists()
    assert Path(returned).stat().st_size > 0


def test_cli_local_end_to_end_creates_real_data_outputs(tmp_path):
    """Exercise the user-facing local-data path through analysis and rendering.

    This test requires the normal scientific runtime dependencies, including
    Astropy. It intentionally skips MCMC so the regression is focused on the
    acquisition -> period search -> Fourier fit -> observational plot path.
    """
    from stellar_pulsation.cli import main

    t, _phase, mag, err = _observed_like_dataset(n=240)
    local = tmp_path / "fake_local.csv"
    np.savetxt(local, np.column_stack([t, mag, err]), delimiter=",")
    outdir = tmp_path / "out"

    rc = main([
        "--source", "local",
        "--data-file", str(local),
        "--delimiter", ",",
        "--target", "Delta Cep local integration fixture",
        "--band", "V",
        "--time-system", "JD",
        "--min-period", "4.5",
        "--max-period", "6.2",
        "--skip-mcmc",
        "--outdir", str(outdir),
    ])

    assert rc == 0
    assert (outdir / "raw_photometry.csv").exists()
    assert (outdir / "raw_photometry.metadata.json").exists()
    assert (outdir / "summary.png").exists()
    assert (outdir / "results.json").exists()
    assert (outdir / "summary.png").stat().st_size > 0
