"""Static and animated visualizations for the semi-empirical model."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from .physical_model import PulsationModel


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _temperature_color(T, Tmin, Tmax):
    # Visual diagnostic only; not a calibrated stellar colour index.
    x = (T - Tmin) / (Tmax - Tmin + 1e-12)
    return matplotlib.colormaps["coolwarm"](np.clip(x, 0, 1))


def render_summary_figure(model, t_obs=None, m_obs=None, m_err=None,
                          ls_result=None, out_path="summary.png"):
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 3, wspace=.35, hspace=.4)
    t = np.linspace(0, 2 * model.period, 2000)
    R = model.radius(t)
    T = model.temperature(t)
    m = model.apparent_magnitude(t)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    R0 = model.radius(np.array([0.0]))[0]
    T0 = model.temperature(np.array([0.0]))[0]
    star = Circle(
        (.5, .5),
        .27 * _safe_scale(R0, R.min(), R.max()),
        color=_temperature_color(T0, T.min(), T.max()),
    )
    ax0.add_patch(star)
    ax0.set_title("Semi-empirical pulsation state")

    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(t, m, label="observed-band model")
    if t_obs is not None and m_obs is not None:
        ax1.errorbar(
            np.mod(t_obs, 2 * model.period), m_obs, yerr=m_err,
            fmt=".", alpha=.35, label="data"
        )
    ax1.invert_yaxis()
    ax1.legend()
    _style(ax1, "V-band light curve", "Time [d]", "Magnitude")

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t, R, label="Radius [Rsun]")
    ax2b = ax2.twinx()
    ax2b.plot(t, T, label="Teff [K]")
    _style(ax2, "Radius and temperature", "Time [d]", "Radius [Rsun]")
    ax2b.set_ylabel("Teff [K]")

    ax3 = fig.add_subplot(gs[1, 2])
    if ls_result is not None:
        ax3.plot(ls_result["frequency_grid"], ls_result["power"])
    _style(ax3, "Lomb-Scargle periodogram", "Frequency [1/d]", "Power")

    fig.suptitle("Stellar Pulsation & Variable-Star Light-Curve Analysis")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _safe_scale(x, xmin, xmax):
    return .8 + .4 * (x - xmin) / (xmax - xmin + 1e-12)


def render_animation(model, out_path="pulsation.gif", n_frames=60, fps=15):
    fig, (ax_star, ax_lc) = plt.subplots(
        1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    ax_star.set_aspect("equal")
    ax_star.axis("off")
    ax_star.set_xlim(0, 1)
    ax_star.set_ylim(0, 1)
    phase = np.linspace(0, 1, n_frames, endpoint=False)
    t = phase * model.period
    grid = np.linspace(0, model.period, 500)
    Rg = model.radius(grid)
    Tg = model.temperature(grid)
    mg = model.apparent_magnitude(grid)
    patch = Circle((.5, .5), .2)
    ax_star.add_patch(patch)
    ax_lc.plot(grid / model.period, mg, alpha=.5)
    ax_lc.invert_yaxis()
    marker, = ax_lc.plot([], [], "o")
    _style(ax_lc, "V-band light curve", "Phase", "Magnitude")

    def update(i):
        R = model.radius(np.array([t[i]]))[0]
        T = model.temperature(np.array([t[i]]))[0]
        patch.set_radius(.22 * _safe_scale(R, Rg.min(), Rg.max()))
        patch.set_color(_temperature_color(T, Tg.min(), Tg.max()))
        marker.set_data([phase[i]], [model.apparent_magnitude(np.array([t[i]]))[0]])
        return patch, marker

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
