"""Anonymous AAVSO light-curve retrieval through the public VSX/AID endpoint.

The endpoint used here is the public ``view=api.object`` light-curve route
illustrated by AAVSO community documentation. It does not use the authenticated
AAVSO Target Tool API. Remote schemas can change, so parsing is deliberately
header-driven and fails loudly rather than guessing columns.
"""
from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import numpy as np
import requests

from .base import PhotometrySource, SourceInfo, TargetNotFound, RemoteServiceError
from ..photometry import PhotometryData


class AAVSOSource(PhotometrySource):
    info = SourceInfo(
        key="aavso",
        name="AAVSO International Database",
        description="Public optical/NIR variable-star observations via the anonymous VSX/AID light-curve route.",
        requires_credentials=False,
        supports_common_names=True,
        default_band="V",
    )
    endpoint = "https://www.aavso.org/vsx/index.php"

    _TIME = {"jd", "julian date", "date", "hjd"}
    _MAG = {"magnitude", "mag", "magnitude (mag)"}
    _ERR = {"uncertainty", "error", "err", "mag error", "magnitude error"}
    _BAND = {"band", "filter", "passband"}

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0):
        self.session = session or requests.Session()
        self.timeout = timeout

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())

    @classmethod
    def _column(cls, fieldnames, aliases, required=True):
        normalized = {cls._norm(name): name for name in fieldnames if name is not None}
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        # tolerate decorated headers such as "JD (UTC)"
        for norm, original in normalized.items():
            if any(alias in norm for alias in aliases):
                return original
        if required:
            raise RemoteServiceError(
                f"AAVSO CSV schema changed: could not identify one of {sorted(aliases)}; "
                f"received columns={list(fieldnames)}"
            )
        return None

    @staticmethod
    def _extract_csv_payload(text: str) -> str:
        """Extract the CSV observation block from AAVSO's XML-wrapped response.

        The public ``api.object&...&csv`` route returns object metadata as XML
        while the observation records are CSV text inside the ``<Data>``
        element.  Plain CSV is also accepted to keep the parser reusable and
        easy to unit-test.
        """
        raw = text.strip()
        if not raw:
            return ""
        if raw.startswith("<"):
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                # Fail over to the conservative line filter below; this also
                # handles legacy responses with non-well-formed wrapper text.
                pass
            else:
                for elem in root.iter():
                    local_name = elem.tag.rsplit("}", 1)[-1].lower()
                    if local_name == "data":
                        payload = "".join(elem.itertext()).strip()
                        if payload:
                            return payload
        return "\n".join(
            ln for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith(("#", "<"))
        )

    @classmethod
    def parse_csv(cls, text: str, target: str, requested_band: str, url: str) -> PhotometryData:
        payload = cls._extract_csv_payload(text)
        stripped = [ln for ln in payload.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not stripped:
            raise TargetNotFound(f"AAVSO returned no observations for {target!r} in band {requested_band!r}")

        # Locate the actual CSV header instead of assuming it is line 1.
        header_index = None
        for i, line in enumerate(stripped):
            cells = [cls._norm(c) for c in next(csv.reader([line]))]
            if any(c in cls._TIME or "julian" in c for c in cells) and any(c in cls._MAG or c == "mag" for c in cells):
                header_index = i
                break
        if header_index is None:
            sample = "\n".join(stripped[:5])
            if "no data" in sample.lower() or "not found" in sample.lower():
                raise TargetNotFound(f"AAVSO has no usable data for {target!r} in band {requested_band!r}")
            raise RemoteServiceError("AAVSO response was not recognizable CSV light-curve data")

        reader = csv.DictReader(io.StringIO("\n".join(stripped[header_index:])))
        if not reader.fieldnames:
            raise RemoteServiceError("AAVSO CSV response has no header")
        tcol = cls._column(reader.fieldnames, cls._TIME)
        mcol = cls._column(reader.fieldnames, cls._MAG)
        ecol = cls._column(reader.fieldnames, cls._ERR, required=False)
        bcol = cls._column(reader.fieldnames, cls._BAND, required=False)

        times, mags, errs, bands = [], [], [], []
        for row in reader:
            try:
                t = float(row[tcol])
                m = float(row[mcol])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                e = float(row[ecol]) if ecol and row.get(ecol, "").strip() else np.nan
            except (TypeError, ValueError):
                e = np.nan
            times.append(t); mags.append(m); errs.append(e)
            if bcol and row.get(bcol):
                bands.append(row[bcol].strip())

        if not times:
            raise TargetNotFound(f"AAVSO returned no parseable observations for {target!r} in band {requested_band!r}")
        band = requested_band if requested_band else (max(set(bands), key=bands.count) if bands else "unknown")
        return PhotometryData(
            np.asarray(times), np.asarray(mags), np.asarray(errs),
            target=target, source="AAVSO", band=band, time_system="JD",
            magnitude_system="apparent", provenance_url=url,
            metadata={"retrieval": "anonymous AAVSO VSX/AID light-curve endpoint"},
        ).with_default_uncertainty()

    def fetch(self, target: str, band: str | None = None, *, max_points: int = 50000,
              from_jd: float | None = None, to_jd: float | None = None, **kwargs) -> PhotometryData:
        band = band or self.info.default_band
        params = {
            "view": "api.object",
            "ident": target,
            "data": int(max_points),
            "csv": "",
            "band": band,
            "mtype": "std",
        }
        if from_jd is not None:
            params["fromjd"] = float(from_jd)
        if to_jd is not None:
            params["tojd"] = float(to_jd)
        try:
            response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RemoteServiceError(f"AAVSO request failed: {exc}") from exc
        data = self.parse_csv(response.text, target, band, response.url)
        data.source_identifier = target
        return data
