"""Plots designed for real observational data rather than a synthetic star cartoon."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fourier_decomposition import evaluate_fourier


def render_observational_summary(data, detrended_mag, ls_result, phase, fourier_result, out_path):
    """Create a four-panel provenance-aware summary for an observed light curve."""
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, wspace=.28, hspace=.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(data.time, data.magnitude, yerr=data.uncertainty, fmt=".", ms=3, alpha=.55)
    ax.invert_yaxis()
    ax.set_title(f"{data.target} — {data.source} {data.band}")
    ax.set_xlabel(data.time_system)
    ax.set_ylabel("Magnitude")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ls_result["frequency_grid"], ls_result["power"])
    best = ls_result["best_period_days"]
    ax.set_title(f"Lomb–Scargle: P = {best:.8g} d")
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel("Power")

    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(phase, detrended_mag, yerr=data.uncertainty, fmt=".", ms=3, alpha=.45, label="data")
    grid = np.linspace(0, 1, 1000, endpoint=False)
    model = evaluate_fourier(
        grid,
        fourier_result["mean"],
        fourier_result["cosine_coeffs"],
        fourier_result["sine_coeffs"],
    )
    ax.plot(grid, model, lw=2, label="weighted Fourier fit")
    ax.invert_yaxis()
    ax.set_title("Phase-folded light curve")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Detrended magnitude")
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    fitted_at_data = evaluate_fourier(
        phase,
        fourier_result["mean"],
        fourier_result["cosine_coeffs"],
        fourier_result["sine_coeffs"],
    )
    residual = detrended_mag - fitted_at_data
    ax.axhline(0.0, lw=1)
    ax.errorbar(phase, residual, yerr=data.uncertainty, fmt=".", ms=3, alpha=.5)
    ax.set_title("Fourier residuals")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Residual [mag]")

    fig.suptitle("Observed Pulsating-Variable Analysis")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path
