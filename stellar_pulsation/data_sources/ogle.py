"""Anonymous OGLE-IV OCVS photometry retrieval.

Exact OGLE identifiers are the most reliable input. For common star names the
adapter may resolve coordinates through CDS Sesame (via Astropy) and search
curated OGLE-IV identification tables for a positional match. No login/API key
is used. The archive itself is the source of truth; a failed HTTP lookup is
reported as 'not found' rather than silently substituting another object.
"""
from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import quote
import numpy as np
import requests

from .base import PhotometrySource, SourceInfo, TargetNotFound, RemoteServiceError
from ..photometry import PhotometryData


class OGLESource(PhotometrySource):
    info = SourceInfo(
        key="ogle",
        name="OGLE-IV Collection of Variable Stars",
        description="Public OCVS time-series photometry; exact OGLE IDs are preferred, with optional coordinate matching.",
        requires_credentials=False,
        supports_common_names=True,
        default_band="I",
    )
    root = "https://ogle.astrouw.edu.pl/ogle/ogle4/OCVS"

    # Curated collections relevant to pulsating variables. New collections can
    # be added without changing the analysis pipeline.
    COLLECTIONS = {
        "OGLE-LMC-CEP": "lmc/cep", "OGLE-SMC-CEP": "smc/cep",
        "OGLE-GD-CEP": "gd/cep", "OGLE-BLG-CEP": "blg/cep",
        "OGLE-LMC-T2CEP": "lmc/t2cep", "OGLE-SMC-T2CEP": "smc/t2cep",
        "OGLE-GD-T2CEP": "gd/t2cep", "OGLE-BLG-T2CEP": "blg/t2cep",
        "OGLE-LMC-ACEP": "lmc/acep", "OGLE-SMC-ACEP": "smc/acep",
        "OGLE-GAL-ACEP": "gal/acep",
        "OGLE-LMC-RRLYR": "lmc/rrlyr", "OGLE-SMC-RRLYR": "smc/rrlyr",
        "OGLE-GD-RRLYR": "gd/rrlyr", "OGLE-BLG-RRLYR": "blg/rrlyr",
        "OGLE-LMC-DSCT": "lmc/dsct", "OGLE-SMC-DSCT": "smc/dsct",
    }

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0, match_radius_arcsec: float = 3.0):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.match_radius_arcsec = float(match_radius_arcsec)

    @classmethod
    def _collection_for_id(cls, target: str) -> str | None:
        upper = target.strip().upper()
        matches = [(prefix, path) for prefix, path in cls.COLLECTIONS.items() if upper.startswith(prefix + "-")]
        if not matches:
            return None
        return max(matches, key=lambda item: len(item[0]))[1]

    @staticmethod
    def parse_photometry(text: str, target: str, band: str, url: str) -> PhotometryData:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t, mag = float(parts[0]), float(parts[1])
                err = float(parts[2]) if len(parts) > 2 else np.nan
            except ValueError:
                continue
            rows.append((t, mag, err))
        if not rows:
            raise TargetNotFound(f"OGLE returned no photometry for {target!r} in band {band!r}")
        arr = np.asarray(rows, dtype=float)
        return PhotometryData(
            arr[:, 0], arr[:, 1], arr[:, 2], target=target, source="OGLE-IV OCVS",
            band=band, time_system="HJD", magnitude_system="apparent",
            provenance_url=url, source_identifier=target,
            metadata={"format": "OGLE OCVS whitespace photometry (HJD, mag, err)"},
        ).with_default_uncertainty()

    def _get_text(self, url: str, *, not_found_is_target: bool = False) -> str:
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 404 and not_found_is_target:
                raise TargetNotFound(f"OGLE archive has no file at {url}")
            response.raise_for_status()
        except TargetNotFound:
            raise
        except requests.RequestException as exc:
            raise RemoteServiceError(f"OGLE request failed: {exc}") from exc
        return response.text

    @staticmethod
    def _parse_ident_coordinates(line: str):
        """Extract (id, RA deg, Dec deg) from common OCVS ident.dat layouts."""
        # Colon-separated format first.
        m = re.search(
            r"^\s*(OGLE-\S+)\s+.*?(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s+([+-]\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)",
            line,
        )
        if not m:
            # Many OCVS READMEs define sexagesimal coordinates as separate fields.
            m = re.search(
                r"^\s*(OGLE-\S+)\s+(?:\S+\s+)?(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s+([+-]\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)",
                line,
            )
        if not m:
            return None
        ident = m.group(1)
        rh, rm, rs = map(float, m.group(2, 3, 4))
        dd = float(m.group(5)); dm = float(m.group(6)); ds = float(m.group(7))
        ra = 15.0 * (rh + rm / 60.0 + rs / 3600.0)
        sign = -1.0 if dd < 0 else 1.0
        dec = sign * (abs(dd) + dm / 60.0 + ds / 3600.0)
        return ident, ra, dec

    @lru_cache(maxsize=32)
    def _ident_table(self, collection: str):
        url = f"{self.root}/{collection}/ident.dat"
        text = self._get_text(url)
        rows = []
        for line in text.splitlines():
            parsed = self._parse_ident_coordinates(line)
            if parsed is not None:
                rows.append(parsed)
        return rows

    def resolve_common_name(self, target: str) -> tuple[str, str]:
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coord = SkyCoord.from_name(target)
        except Exception as exc:
            raise TargetNotFound(
                f"Could not resolve {target!r} to coordinates. For OGLE, use an exact identifier "
                "such as OGLE-LMC-CEP-0005, or check network/Sesame availability."
            ) from exc

        best = None
        for collection in dict.fromkeys(self.COLLECTIONS.values()):
            try:
                rows = self._ident_table(collection)
            except (RemoteServiceError, TargetNotFound):
                continue
            for ident, ra, dec in rows:
                candidate = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                sep = coord.separation(candidate).arcsec
                if best is None or sep < best[0]:
                    best = (sep, ident, collection)
        if best is None or best[0] > self.match_radius_arcsec:
            raise TargetNotFound(
                f"No object within {self.match_radius_arcsec:g} arcsec of {target!r} was found "
                "in the configured OGLE pulsating-variable collections."
            )
        return best[1], best[2]

    def fetch(self, target: str, band: str | None = None, **kwargs) -> PhotometryData:
        band = (band or self.info.default_band).upper()
        collection = self._collection_for_id(target)
        resolved_from_name = False
        if collection is None:
            target, collection = self.resolve_common_name(target)
            resolved_from_name = True

        url = f"{self.root}/{collection}/phot/{quote(band)}/{quote(target)}.dat"
        try:
            text = self._get_text(url, not_found_is_target=True)
        except TargetNotFound as exc:
            raise TargetNotFound(
                f"OGLE object {target!r} was identified, but no {band}-band photometry file exists. "
                "Try another band/source."
            ) from exc
        data = self.parse_photometry(text, target, band, url)
        data.metadata["resolved_from_common_name"] = resolved_from_name
        data.metadata["collection"] = collection
        return data
