"""Lomb-Scargle period search with baseline-aware sampling and peak refinement."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from astropy.timeseries import LombScargle


def _refine_peak_frequency(ls, frequency, power, best_idx):
    """Refine the strongest sampled peak with bounded scalar optimization.

    ``autopower`` supplies a baseline-aware discovery grid.  The final
    frequency is then optimized continuously inside the two neighboring grid
    points, avoiding the quantization error of returning the raw grid maximum.
    """
    if best_idx <= 0 or best_idx >= len(frequency) - 1:
        return float(frequency[best_idx]), float(power[best_idx]), False
    lo = float(frequency[best_idx - 1])
    hi = float(frequency[best_idx + 1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return float(frequency[best_idx]), float(power[best_idx]), False

    result = minimize_scalar(
        lambda f: -float(ls.power(f)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": max((hi - lo) * 1e-8, np.finfo(float).eps)},
    )
    if result.success and lo <= result.x <= hi and np.isfinite(result.fun):
        return float(result.x), float(-result.fun), True
    return float(frequency[best_idx]), float(power[best_idx]), False


def lomb_scargle_period(
    t,
    y,
    yerr=None,
    min_period=0.1,
    max_period=50.0,
    samples_per_peak=8,
    n_freq=None,
    refine_peak=True,
):
    """Return the dominant Lomb-Scargle period and diagnostics.

    The discovery grid is baseline-aware through ``LombScargle.autopower`` by
    default.  ``refine_peak=True`` performs a local continuous optimization of
    the strongest peak so the reported period is not limited to the grid
    spacing.  ``samples_per_peak`` therefore controls discovery sampling, not
    the final period quantization.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) != len(y) or len(t) < 5:
        raise ValueError("need at least five matched samples")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("t and y must be finite")
    if min_period <= 0 or max_period <= min_period:
        raise ValueError("require 0 < min_period < max_period")
    if samples_per_peak < 1:
        raise ValueError("samples_per_peak must be >= 1")
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        if yerr.shape != y.shape or np.any(yerr <= 0) or np.any(~np.isfinite(yerr)):
            raise ValueError("invalid yerr")

    ls = LombScargle(t, y, yerr)
    fmin = 1 / max_period
    fmax = 1 / min_period
    if n_freq is None:
        frequency, power = ls.autopower(
            minimum_frequency=fmin,
            maximum_frequency=fmax,
            samples_per_peak=samples_per_peak,
        )
    else:
        if n_freq < 100:
            raise ValueError("n_freq must be >=100")
        frequency = np.linspace(fmin, fmax, int(n_freq))
        power = ls.power(frequency)

    best_idx = int(np.argmax(power))
    best_frequency = float(frequency[best_idx])
    best_power = float(power[best_idx])
    refined = False
    if refine_peak:
        best_frequency, best_power, refined = _refine_peak_frequency(
            ls, frequency, power, best_idx
        )

    peak_mask = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_idx = np.where(peak_mask)[0] + 1
    peak_idx = peak_idx[np.argsort(power[peak_idx])[::-1]][:5]
    top = [(float(1 / frequency[i]), float(power[i])) for i in peak_idx]
    if top and refined:
        # Keep ranking semantics while reporting the refined dominant solution.
        top[0] = (float(1 / best_frequency), best_power)

    fap = float(ls.false_alarm_probability(best_power))
    baseline = float(np.ptp(t))
    resolution = 1 / baseline if baseline > 0 else np.nan
    return {
        "best_period_days": float(1 / best_frequency),
        "best_frequency": best_frequency,
        "best_power": best_power,
        "peak_refined": refined,
        "false_alarm_probability": fap,
        "frequency_resolution_approx": float(resolution),
        "frequency_grid": frequency,
        "power": power,
        "top_periods_by_power": top,
    }
