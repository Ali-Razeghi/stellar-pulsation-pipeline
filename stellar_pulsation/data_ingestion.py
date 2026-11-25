"""Backward-compatible ingestion helpers.

New code should prefer :mod:`stellar_pulsation.data_sources` and
:class:`stellar_pulsation.photometry.PhotometryData`.
"""
from __future__ import annotations
import numpy as np
from .data_sources.tess import TESSSource


def fetch_tess_light_curve(target_name: str, mission: str = "TESS"):
    data = TESSSource().fetch(target_name)
    # Recover a flux-like quantity only for compatibility with older callers.
    flux = 10 ** (-0.4 * data.magnitude)
    ferr = data.uncertainty * flux * np.log(10.0) / 2.5
    return data.time, flux, ferr


def fetch_flux_as_magnitude(target_name: str, mission: str = "TESS"):
    data = TESSSource().fetch(target_name)
    return data.time, data.magnitude, data.uncertainty


def load_ascii_photometry(path, time_col=0, mag_col=1, err_col=2, delimiter=None, skiprows=0):
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skiprows)
    data = np.atleast_2d(data)
    needed = max(time_col, mag_col, err_col if err_col is not None else 0)
    if data.shape[1] <= needed:
        raise ValueError("requested column index exceeds file width")
    t = np.asarray(data[:, time_col], float)
    mag = np.asarray(data[:, mag_col], float)
    err = np.full_like(mag, np.nan) if err_col is None else np.asarray(data[:, err_col], float)
    valid = np.isfinite(t) & np.isfinite(mag) & (np.isfinite(err) if err_col is not None else True)
    return t[valid], mag[valid], err[valid]


DATA_SOURCE_NOTES = {
    "aavso": "anonymous AAVSO VSX/AID light-curve retrieval; prefer a homogeneous photometric band",
    "ogle": "anonymous OGLE-IV OCVS retrieval; exact OGLE identifier is preferred",
    "tess": "public TESS/MAST access via lightkurve; magnitudes are relative",
    "local": "local CSV/ASCII import; user is responsible for column/time-system metadata",
    "asas-sn": "automatic access is intentionally not implemented: Sky Patrol V2 Python access requires credentials and the legacy public form forbids scripted automation",
}
