"""Anonymous TESS/MAST retrieval via Lightkurve."""
from __future__ import annotations
import numpy as np
from .base import PhotometrySource, SourceInfo, TargetNotFound, RemoteServiceError, SourceConfigurationError
from ..photometry import PhotometryData


class TESSSource(PhotometrySource):
    info = SourceInfo(
        key="tess", name="TESS / MAST",
        description="Public TESS light curves from MAST via Lightkurve; converted to relative magnitudes.",
        requires_credentials=False, supports_common_names=True, default_band="TESS",
    )

    def fetch(self, target: str, band: str | None = None, **kwargs) -> PhotometryData:
        try:
            import lightkurve as lk
        except ImportError as exc:
            raise SourceConfigurationError(
                "TESS access requires lightkurve. Install the project requirements or `pip install lightkurve`."
            ) from exc
        try:
            search = lk.search_lightcurve(target, mission="TESS")
            if len(search) == 0:
                raise TargetNotFound(f"No TESS light-curve products were found for {target!r}")
            collection = search.download_all()
            if collection is None or len(collection) == 0:
                raise TargetNotFound(f"TESS search matched {target!r}, but no downloadable light curve was returned")
            lc = collection.stitch().remove_nans()
        except TargetNotFound:
            raise
        except Exception as exc:
            raise RemoteServiceError(f"TESS/MAST request failed: {exc}") from exc

        t = np.asarray(lc.time.value, float)
        flux = np.asarray(lc.flux.value, float)
        ferr = np.asarray(lc.flux_err.value, float)
        valid = np.isfinite(t) & np.isfinite(flux) & np.isfinite(ferr) & (flux > 0) & (ferr >= 0)
        t, flux, ferr = t[valid], flux[valid], ferr[valid]
        if not len(t):
            raise TargetNotFound(f"TESS returned no finite positive-flux measurements for {target!r}")
        mag = -2.5 * np.log10(flux)
        mag_err = (2.5 / np.log(10.0)) * ferr / flux
        return PhotometryData(
            t, mag, mag_err, target=target, source="TESS/MAST", band="TESS",
            time_system="BTJD (TDB)", magnitude_system="relative",
            provenance_url="https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html",
            source_identifier=target,
            metadata={
                "warning": "Relative magnitudes derived from TESS flux; not Johnson V and not suitable for a V-band PL zero-point test.",
                "n_products": len(collection),
            },
        ).with_default_uncertainty()
