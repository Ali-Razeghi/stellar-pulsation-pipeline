"""Weighted linear Fourier decomposition for phase-folded light curves."""
from __future__ import annotations
import numpy as np


def phase_fold(t, period, t0=0.0):
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive")
    return np.mod((np.asarray(t, dtype=float) - t0) / period, 1.0)


def _design_matrix(phase, n_harmonics):
    cols = [np.ones_like(phase, dtype=float)]
    for k in range(1, n_harmonics + 1):
        cols.extend([np.cos(2 * np.pi * k * phase), np.sin(2 * np.pi * k * phase)])
    return np.column_stack(cols)


def evaluate_fourier(phase, mean, cosine_coeffs, sine_coeffs):
    phase = np.asarray(phase, dtype=float)
    y = np.full_like(phase, mean)
    for k, (c, s) in enumerate(zip(cosine_coeffs, sine_coeffs), 1):
        y += c * np.cos(2 * np.pi * k * phase) + s * np.sin(2 * np.pi * k * phase)
    return y


def _circular_extrema_counts(y):
    """Count strict local maxima/minima on a periodic uniformly sampled curve."""
    y = np.asarray(y, dtype=float)
    maxima = (y > np.roll(y, 1)) & (y > np.roll(y, -1))
    minima = (y < np.roll(y, 1)) & (y < np.roll(y, -1))
    return int(np.sum(maxima)), int(np.sum(minima))


def fit_fourier_series(phase, y, yerr=None, n_harmonics=4):
    """Weighted linear least squares in cosine/sine coefficients.

    Reported phases use the same cosine convention as
    ``A_k cos(2*pi*k*phase + phi_k)``.  Therefore ``phi_i1`` is directly
    comparable with standard Cepheid Fourier phase combinations such as
    phi21 and phi31 in the cited delta-Cephei benchmark.
    """
    phase = np.asarray(phase, float)
    y = np.asarray(y, float)
    if len(phase) != len(y) or len(y) < 2 * n_harmonics + 1:
        raise ValueError("insufficient/mismatched data")
    X = _design_matrix(phase, n_harmonics)
    if yerr is None:
        w = np.ones_like(y)
    else:
        yerr = np.asarray(yerr, float)
        if yerr.shape != y.shape or np.any(~np.isfinite(yerr)) or np.any(yerr <= 0):
            raise ValueError("yerr must be positive and finite")
        w = 1 / yerr**2
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    if rank < X.shape[1]:
        raise np.linalg.LinAlgError("Fourier design matrix is rank deficient")
    model = X @ beta
    resid = y - model
    dof = max(len(y) - len(beta), 1)
    chi2 = float(np.sum(w * resid**2))
    s2 = chi2 / dof
    cov = np.linalg.inv(Xw.T @ Xw) * s2
    mean = float(beta[0])
    cos = beta[1::2]
    sin = beta[2::2]
    amps = np.hypot(cos, sin)
    # c*cos(theta)+s*sin(theta) = A*cos(theta+phi), phi=atan2(-s,c)
    phases = np.mod(np.arctan2(-sin, cos), 2 * np.pi)
    R_i1 = {}
    phi_i1 = {}
    if amps[0] <= np.finfo(float).eps:
        raise ValueError("fundamental Fourier amplitude is zero")
    for i in range(2, n_harmonics + 1):
        R_i1[i] = float(amps[i - 1] / amps[0])
        phi_i1[i] = float(np.mod(phases[i - 1] - i * phases[0], 2 * np.pi))
    amp_err = []
    for k, (c, s, a) in enumerate(zip(cos, sin, amps)):
        idx = 1 + 2 * k
        C = cov[idx:idx + 2, idx:idx + 2]
        if a > 0:
            g = np.array([c / a, s / a])
            amp_err.append(float(np.sqrt(max(g @ C @ g, 0))))
        else:
            amp_err.append(float("nan"))
    return {
        "mean": mean,
        "cosine_coeffs": cos.tolist(),
        "sine_coeffs": sin.tolist(),
        "amplitudes": amps.tolist(),
        "phases": phases.tolist(),
        "phase_convention": "A_k cos(2*pi*k*phase + phi_k)",
        "amplitude_errors": amp_err,
        "R_i1": R_i1,
        "phi_i1": phi_i1,
        "rms_residual": float(np.sqrt(np.mean(resid**2))),
        "reduced_chi2": float(chi2 / dof),
        "covariance": cov.tolist(),
        "model": model,
    }


def morphology_metrics(phase, y, n_harmonics=3):
    fit = fit_fourier_series(phase, y, n_harmonics=n_harmonics)
    grid = np.linspace(0, 1, 20000, endpoint=False)
    model = evaluate_fourier(grid, fit["mean"], fit["cosine_coeffs"], fit["sine_coeffs"])
    faint = int(np.argmax(model))
    bright = int(np.argmin(model))
    rise = ((bright - faint) % len(grid)) / len(grid)
    n_max, n_min = _circular_extrema_counts(model)
    return {
        "peak_to_peak_mag": float(np.ptp(model)),
        "rise_fraction": float(rise),
        "R21": fit["R_i1"].get(2),
        "R31": fit["R_i1"].get(3),
        "phi21_rad": fit["phi_i1"].get(2),
        "phi31_rad": fit["phi_i1"].get(3),
        "local_maxima_per_cycle": n_max,
        "local_minima_per_cycle": n_min,
        "total_local_extrema_per_cycle": n_max + n_min,
    }
