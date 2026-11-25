"""Interfaces and exceptions for remote photometric archives."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..photometry import PhotometryData


class DataSourceError(RuntimeError):
    """Base class for catalog access failures."""


class TargetNotFound(DataSourceError):
    """The target has no usable data in the selected source/band."""


class RemoteServiceError(DataSourceError):
    """The remote archive could not be reached or returned an invalid response."""


class SourceConfigurationError(DataSourceError):
    """The requested source cannot operate with the supplied options."""


@dataclass(frozen=True)
class SourceInfo:
    key: str
    name: str
    description: str
    requires_credentials: bool
    supports_common_names: bool
    default_band: str


class PhotometrySource(ABC):
    info: SourceInfo

    @abstractmethod
    def fetch(self, target: str, band: str | None = None, **kwargs) -> PhotometryData:
        raise NotImplementedError
