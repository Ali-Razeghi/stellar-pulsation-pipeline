"""Canonical photometry container used by all remote and local data sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import json
import numpy as np


@dataclass
class PhotometryData:
    """A homogeneous light-curve representation.

    Time is kept in the source's native astronomical time system (documented in
    ``time_system``). Magnitudes may be absolute-calibrated or relative; that is
    explicitly recorded in ``magnitude_system`` so downstream code does not
    silently treat TESS relative magnitudes as Johnson V.
    """

    time: np.ndarray
    magnitude: np.ndarray
    uncertainty: np.ndarray
    target: str
    source: str
    band: str
    time_system: str
    magnitude_system: str = "apparent"
    provenance_url: str | None = None
    source_identifier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.magnitude = np.asarray(self.magnitude, dtype=float)
        self.uncertainty = np.asarray(self.uncertainty, dtype=float)
        if not (len(self.time) == len(self.magnitude) == len(self.uncertainty)):
            raise ValueError("time, magnitude, and uncertainty must have equal length")
        valid = np.isfinite(self.time) & np.isfinite(self.magnitude)
        self.time = self.time[valid]
        self.magnitude = self.magnitude[valid]
        self.uncertainty = self.uncertainty[valid]
        if len(self.time) == 0:
            raise ValueError("photometry contains no finite observations")
        order = np.argsort(self.time)
        self.time = self.time[order]
        self.magnitude = self.magnitude[order]
        self.uncertainty = self.uncertainty[order]

    @property
    def n_points(self) -> int:
        return len(self.time)

    def with_default_uncertainty(self, value: float = 0.02) -> "PhotometryData":
        err = self.uncertainty.copy()
        bad = ~np.isfinite(err) | (err <= 0)
        if np.any(bad):
            finite = err[np.isfinite(err) & (err > 0)]
            fallback = float(np.median(finite)) if finite.size else float(value)
            err[bad] = fallback
        self.uncertainty = err
        return self

    def save_csv(self, path: str | Path) -> str:
        """Save a normalized, analysis-ready CSV plus provenance sidecar JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time", "magnitude", "uncertainty", "band", "source", "target"])
            for t, m, e in zip(self.time, self.magnitude, self.uncertainty):
                writer.writerow([f"{t:.10f}", f"{m:.8f}", f"{e:.8f}", self.band, self.source, self.target])
        sidecar = path.with_suffix(".metadata.json")
        payload = {
            "target": self.target,
            "source": self.source,
            "source_identifier": self.source_identifier,
            "band": self.band,
            "time_system": self.time_system,
            "magnitude_system": self.magnitude_system,
            "provenance_url": self.provenance_url,
            "n_points": self.n_points,
            "metadata": self.metadata,
        }
        sidecar.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)
