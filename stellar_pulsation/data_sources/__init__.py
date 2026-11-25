"""Remote-source registry for real astronomical photometry."""
from .base import (
    DataSourceError, TargetNotFound, RemoteServiceError,
    SourceConfigurationError, SourceInfo, PhotometrySource,
)
from .aavso import AAVSOSource
from .ogle import OGLESource
from .tess import TESSSource
from .local import LocalFileSource


def build_source_registry():
    return {
        "aavso": AAVSOSource(),
        "ogle": OGLESource(),
        "tess": TESSSource(),
        "local": LocalFileSource(),
    }


PUBLIC_REMOTE_SOURCES = ("aavso", "ogle", "tess")

__all__ = [
    "DataSourceError", "TargetNotFound", "RemoteServiceError", "SourceConfigurationError",
    "SourceInfo", "PhotometrySource", "AAVSOSource", "OGLESource", "TESSSource",
    "LocalFileSource", "build_source_registry", "PUBLIC_REMOTE_SOURCES",
]
