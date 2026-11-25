from types import SimpleNamespace
import numpy as np
from stellar_pulsation.cli import _interactive_request
from stellar_pulsation.data_sources.base import TargetNotFound, SourceInfo
from stellar_pulsation.photometry import PhotometryData


class MissingSource:
    info = SourceInfo("aavso", "AAVSO", "test missing", False, True, "V")
    def fetch(self, target, band=None, **kwargs):
        raise TargetNotFound("not present")


class WorkingSource:
    info = SourceInfo("ogle", "OGLE", "test working", False, True, "I")
    def fetch(self, target, band=None, **kwargs):
        return PhotometryData(
            np.arange(25.0), 15 + .1*np.sin(np.arange(25.0)), np.full(25, .02),
            target=target, source="OGLE", band=band or "I", time_system="HJD"
        )


class DummySource:
    def __init__(self, key, name, band):
        self.info = SourceInfo(key, name, "dummy", False, True, band)


def test_interactive_target_miss_returns_to_source_selection():
    registry = {
        "aavso": MissingSource(),
        "ogle": WorkingSource(),
        "tess": DummySource("tess", "TESS", "TESS"),
        "local": DummySource("local", "Local", "unknown"),
    }
    answers = iter([
        "1", "Missing Star", "V",   # AAVSO -> TargetNotFound
        "2", "OGLE-LMC-CEP-0005", "I",  # OGLE -> success
    ])
    args = SimpleNamespace(max_points=50000, from_jd=None, to_jd=None, data_file=None)
    result = _interactive_request(registry, args, input_fn=lambda prompt: next(answers))
    assert result[0] == "ogle"
    assert result[3].source == "OGLE"
    assert result[3].n_points == 25
