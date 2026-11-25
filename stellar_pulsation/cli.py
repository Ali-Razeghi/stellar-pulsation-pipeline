"""Interactive and scriptable end-to-end pulsating-variable analysis.

With no ``--source`` argument, the program presents the user with available
public data sources. Remote sources are queried only after the user selects one.
If a target has no usable data in the selected archive/band, the interactive
workflow reports that fact and lets the user choose another source instead of
silently substituting data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

from .physical_model import PulsationModel, FourierMode, delta_cep_like
from .synthetic_data import generate_light_curve
from .photometry import PhotometryData
from .data_sources import (
    build_source_registry, PUBLIC_REMOTE_SOURCES,
    TargetNotFound, RemoteServiceError, SourceConfigurationError,
)
from .detrending import sigma_clip, auto_detrend
from .period_search import lomb_scargle_period
from .fourier_decomposition import phase_fold, fit_fourier_series, morphology_metrics
from .bayesian_fit import run_mcmc
from .period_luminosity import PLRelation
from .animation import render_summary_figure, render_animation
from .observational_plots import render_observational_summary


def build_simulation_model(args):
    """Build the optional synthetic benchmark model from CLI parameters."""
    structural_override = any(
        getattr(args, name) is not None
        for name in ("r0", "period", "t0", "t_amp", "radius_amp")
    )
    base = delta_cep_like()
    if not structural_override:
        if args.distance_modulus is not None:
            base.distance_modulus = args.distance_modulus
        base.A_V = args.av
        return base

    r0 = base.R0 if args.r0 is None else args.r0
    period = base.period if args.period is None else args.period
    t0 = base.T0 if args.t0 is None else args.t0
    t_amp = 0.08 * t0 if args.t_amp is None else args.t_amp
    radius_amp = 0.05 * r0 if args.radius_amp is None else args.radius_amp
    mu = base.distance_modulus if args.distance_modulus is None else args.distance_modulus
    if radius_amp < 0 or t_amp < 0:
        raise ValueError("--radius-amp and --t-amp must be non-negative")
    radius_modes = [FourierMode(1, radius_amp, 0.0)] if radius_amp else []
    temperature_modes = [FourierMode(1, t_amp, np.pi)] if t_amp else []
    return PulsationModel(
        R0=r0, period=period, modes=radius_modes, T0=t0,
        temperature_modes=temperature_modes, distance_modulus=mu,
        A_V=args.av, photometric_template=None,
    )


def _synthetic_data(args, model) -> PhotometryData:
    t, m, e = generate_light_curve(
        model, n_cycles=args.synthetic_cycles,
        photometric_noise_mag=args.synthetic_noise,
        gap_fraction=args.synthetic_gap_fraction,
    )
    return PhotometryData(
        t, m, e, target="delta Cep-like synthetic benchmark",
        source="Synthetic benchmark", band="V", time_system="model days",
        magnitude_system="simulated apparent", source_identifier="delta_cep_like",
        metadata={"purpose": "testing/education; not an observed data set"},
    )


def _print_sources(registry) -> None:
    print("\nAvailable data sources:")
    for i, key in enumerate(PUBLIC_REMOTE_SOURCES, 1):
        info = registry[key].info
        print(f"  {i}. {info.name} [{key}] — {info.description}")
    print(f"  {len(PUBLIC_REMOTE_SOURCES)+1}. Local CSV/ASCII file [local]")
    print(f"  {len(PUBLIC_REMOTE_SOURCES)+2}. Synthetic benchmark [synthetic]")
    print("  q. Quit")


def _choose_source(registry, input_fn=input) -> str:
    keys = list(PUBLIC_REMOTE_SOURCES) + ["local", "synthetic"]
    while True:
        _print_sources(registry)
        raw = input_fn("Select source: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if raw in keys:
            return raw
        try:
            idx = int(raw) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(keys):
            return keys[idx]
        print("Invalid selection. Please choose one of the listed sources.")


def _interactive_request(registry, args, input_fn=input):
    """Acquire one data set, looping on source/target misses as requested by user."""
    while True:
        source_key = _choose_source(registry, input_fn)
        if source_key == "synthetic":
            return source_key, "delta Cep-like synthetic benchmark", None
        if source_key == "local":
            path = input_fn("Path to CSV/ASCII photometry file: ").strip()
            target = input_fn("Target label/name: ").strip() or "local target"
            band = input_fn("Photometric band (e.g. V, I, g; optional): ").strip() or "unknown"
            args.data_file = path
            return source_key, target, band

        source = registry[source_key]
        target = input_fn("Star name or catalogue identifier: ").strip()
        if not target:
            print("A target name/identifier is required.")
            continue
        default_band = source.info.default_band
        band = input_fn(f"Photometric band [{default_band}]: ").strip() or default_band
        try:
            data = source.fetch(
                target, band,
                max_points=args.max_points, from_jd=args.from_jd, to_jd=args.to_jd,
            )
            return source_key, target, band, data
        except TargetNotFound as exc:
            print(f"\nNo usable data found in {source.info.name}: {exc}")
            print("Choose another source, target identifier, or band.")
        except (RemoteServiceError, SourceConfigurationError) as exc:
            print(f"\n{source.info.name} could not be used: {exc}")
            print("Choose another source or correct the local installation/network issue.")


def acquire_data(args, model, registry=None, input_fn=input) -> tuple[PhotometryData, str]:
    registry = registry or build_source_registry()
    source_key = args.source

    if source_key is None:
        result = _interactive_request(registry, args, input_fn)
        if len(result) == 4:
            source_key, _target, _band, data = result
            return data, source_key
        source_key, target, band = result
        args.target = target
        args.band = band

    if source_key == "synthetic":
        return _synthetic_data(args, model), source_key

    if source_key == "local":
        if not args.data_file:
            raise SourceConfigurationError("--data-file is required with --source local")
        data = registry["local"].fetch(
            args.target or "local target", args.band,
            path=args.data_file, time_col=args.time_col, mag_col=args.mag_col,
            err_col=args.err_col, delimiter=args.delimiter, skiprows=args.skiprows,
            time_system=args.time_system,
        )
        return data, source_key

    if source_key not in registry:
        raise SourceConfigurationError(f"unknown source {source_key!r}")
    if not args.target:
        raise SourceConfigurationError(f"--target is required with --source {source_key}")
    source = registry[source_key]
    data = source.fetch(
        args.target, args.band or source.info.default_band,
        max_points=args.max_points, from_jd=args.from_jd, to_jd=args.to_jd,
    )
    return data, source_key


def build_parser():
    p = argparse.ArgumentParser(
        description="Retrieve real variable-star photometry and run a reproducible time-series analysis pipeline."
    )
    p.add_argument("--outdir", default="./out")
    p.add_argument(
        "--source", "--catalog", dest="source",
        choices=["aavso", "ogle", "tess", "local", "synthetic"], default=None,
        help="data source; omit to use the interactive source menu",
    )
    p.add_argument("--list-sources", action="store_true", help="print supported source policy and exit")
    p.add_argument("--target", help="common star name or source identifier")
    p.add_argument("--band", default=None, help="photometric band (source default if omitted)")
    p.add_argument("--max-points", type=int, default=50000, help="AAVSO retrieval limit")
    p.add_argument("--from-jd", type=float, default=None)
    p.add_argument("--to-jd", type=float, default=None)

    p.add_argument("--data-file")
    p.add_argument("--time-col", type=int, default=0)
    p.add_argument("--mag-col", type=int, default=1)
    p.add_argument("--err-col", type=int, default=2)
    p.add_argument("--delimiter", default=None)
    p.add_argument("--skiprows", type=int, default=0)
    p.add_argument("--time-system", default="unknown")

    p.add_argument("--min-period", type=float, default=.1)
    p.add_argument("--max-period", type=float, default=100.0)
    p.add_argument("--samples-per-peak", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=1500)
    p.add_argument("--n-walkers", type=int, default=24)
    p.add_argument("--skip-mcmc", action="store_true", help="skip Bayesian photometric fit for a faster run")

    p.add_argument("--r0", type=float, default=None, help="synthetic mean radius [R_sun]")
    p.add_argument("--period", type=float, default=None, help="synthetic period [d]")
    p.add_argument("--t0", type=float, default=None, help="synthetic mean effective temperature [K]")
    p.add_argument("--t-amp", dest="t_amp", type=float, default=None)
    p.add_argument("--radius-amp", type=float, default=None)
    p.add_argument("--distance-modulus", type=float, default=None)
    p.add_argument("--av", type=float, default=.25)
    p.add_argument("--synthetic-cycles", type=float, default=8.0)
    p.add_argument("--synthetic-noise", type=float, default=.02)
    p.add_argument("--synthetic-gap-fraction", type=float, default=.1)

    p.add_argument("--independent-mu", type=float, default=None,
                   help="independent distance modulus for a V-band PL consistency diagnostic")
    return p


def _source_policy(registry) -> dict:
    return {
        key: {
            "name": src.info.name,
            "requires_credentials": src.info.requires_credentials,
            "supports_common_names": src.info.supports_common_names,
            "default_band": src.info.default_band,
        }
        for key, src in registry.items()
    } | {
        "asas-sn": {
            "automated": False,
            "reason": "Sky Patrol V2 Python client requires credentials; legacy public form explicitly forbids scripted automation. Use local export only.",
        }
    }


def run_analysis(args, data: PhotometryData, source_key: str, model: PulsationModel):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_csv = data.save_csv(outdir / "raw_photometry.csv")

    t, m, e = sigma_clip(data.time, data.magnitude, data.uncertainty, sigma=6)
    if len(t) < 20:
        raise ValueError(f"Only {len(t)} usable points remain; at least 20 are required for this pipeline")
    initial_window = max(min(np.ptp(t) / 3.0, 20.0), .5)
    md = auto_detrend(t, m, initial_window)

    ls = lomb_scargle_period(
        t, md, e, args.min_period, args.max_period,
        samples_per_peak=args.samples_per_peak, refine_peak=True,
    )
    period = float(ls["best_period_days"])
    t, m, e = sigma_clip(t, m, e, sigma=5, period=period)
    md = auto_detrend(t, m, max(period * 4, .5))
    phase = phase_fold(t, period)
    fourier = fit_fourier_series(phase, md, e, n_harmonics=3)
    morph = morphology_metrics(phase, md, 3)

    if args.skip_mcmc:
        post = {"_diagnostics": {"status": "skipped by user"}}
    else:
        _sampler, post = run_mcmc(
            phase, md, e, n_harmonics=3,
            n_walkers=args.n_walkers, n_steps=args.n_steps,
            burn_in=args.n_steps // 3,
        )

    pl = None
    is_v = str(data.band).strip().upper() == "V"
    if args.independent_mu is not None and is_v and data.magnitude_system != "relative":
        pl = PLRelation.benedict2007_v().consistency_check(
            period, float(np.mean(m)), args.independent_mu, args.av,
        )
    elif args.independent_mu is not None:
        pl = {"status": "not evaluated", "reason": "V-band calibrated apparent magnitudes are required"}

    if source_key == "synthetic":
        figure = render_summary_figure(model, t, md, e, ls, str(outdir / "summary.png"))
        animation = render_animation(model, str(outdir / "pulsation.gif"))
    else:
        # Use the cleaned uncertainties in the real-data figure.
        plot_data = PhotometryData(
            t, m, e, target=data.target, source=data.source, band=data.band,
            time_system=data.time_system, magnitude_system=data.magnitude_system,
            provenance_url=data.provenance_url, source_identifier=data.source_identifier,
            metadata=data.metadata,
        )
        figure = render_observational_summary(
            plot_data, md, ls, phase, fourier, str(outdir / "summary.png")
        )
        animation = None

    result = {
        "target": data.target,
        "source": data.source,
        "source_key": source_key,
        "source_identifier": data.source_identifier,
        "provenance_url": data.provenance_url,
        "band": data.band,
        "time_system": data.time_system,
        "magnitude_system": data.magnitude_system,
        "n_points_downloaded": data.n_points,
        "n_points_analyzed": len(t),
        "raw_photometry_csv": raw_csv,
        "period_search": {k: v for k, v in ls.items() if k not in ("frequency_grid", "power")},
        "fourier": {k: v for k, v in fourier.items() if k not in ("model", "covariance")},
        "morphology": morph,
        "bayesian_photometric_posterior": {k: v for k, v in post.items() if not k.startswith("_")},
        "mcmc_diagnostics": post.get("_diagnostics", {}),
        "pl_consistency": pl,
        "figure": figure,
        "animation": animation,
        "scientific_scope": (
            "observational time-series analysis plus phenomenological Fourier/Bayesian modelling; "
            "single-band photometry alone does not independently determine stellar radius, temperature, and distance"
        ),
    }
    (outdir / "results.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    return result


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def main(argv=None):
    args = build_parser().parse_args(argv)
    registry = build_source_registry()
    if args.list_sources:
        print(json.dumps(_source_policy(registry), indent=2))
        return 0

    model = build_simulation_model(args)
    try:
        data, source_key = acquire_data(args, model, registry=registry)
        print(f"Retrieved {data.n_points} points from {data.source} for {data.target} [{data.band}].")
        result = run_analysis(args, data, source_key, model)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except (TargetNotFound, RemoteServiceError, SourceConfigurationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "target": result["target"],
        "source": result["source"],
        "best_period_days": result["period_search"]["best_period_days"],
        "n_points_analyzed": result["n_points_analyzed"],
        "results": str(Path(args.outdir) / "results.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
