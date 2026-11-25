"""Local CSV/ASCII photometry input, including exported ASAS-SN data."""
from __future__ import annotations
import numpy as np
from .base import PhotometrySource, SourceInfo, SourceConfigurationError
from ..photometry import PhotometryData


class LocalFileSource(PhotometrySource):
    info = SourceInfo(
        key="local", name="Local CSV / ASCII file",
        description="User-supplied photometry, including manually exported ASAS-SN data.",
        requires_credentials=False, supports_common_names=False, default_band="unknown",
    )

    def fetch(self, target: str, band: str | None = None, *, path: str | None = None,
              time_col: int = 0, mag_col: int = 1, err_col: int | None = 2,
              delimiter: str | None = None, skiprows: int = 0, time_system: str = "unknown", **kwargs) -> PhotometryData:
        if not path:
            raise SourceConfigurationError("local source requires --data-file")
        data = np.genfromtxt(path, delimiter=delimiter, skip_header=skiprows)
        data = np.atleast_2d(data)
        needed = max(time_col, mag_col, err_col if err_col is not None else 0)
        if data.shape[1] <= needed:
            raise SourceConfigurationError("requested local-file column index exceeds file width")
        t = np.asarray(data[:, time_col], float)
        mag = np.asarray(data[:, mag_col], float)
        err = np.full_like(mag, np.nan) if err_col is None else np.asarray(data[:, err_col], float)
        return PhotometryData(
            t, mag, err, target=target or "local target", source="Local file",
            band=band or "unknown", time_system=time_system, magnitude_system="user-specified/unknown",
            provenance_url=None, source_identifier=path,
            metadata={"path": path, "time_col": time_col, "mag_col": mag_col, "err_col": err_col},
        ).with_default_uncertainty()
