import numpy as np
import pytest

from stellar_pulsation.data_sources.aavso import AAVSOSource
from stellar_pulsation.data_sources.ogle import OGLESource
from stellar_pulsation.data_sources.base import TargetNotFound, SourceInfo
from stellar_pulsation.photometry import PhotometryData


class FakeResponse:
    def __init__(self, text, status_code=200, url="https://example.test/query"):
        self.text = text
        self.status_code = status_code
        self.url = url
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status={self.status_code}")


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.last_url = None
        self.last_params = None
    def get(self, url, params=None, timeout=None):
        self.last_url = url
        self.last_params = params
        return self.response


def test_aavso_public_csv_parser_and_request_semantics():
    text = "JD,Magnitude,Uncertainty,Band\n2450000.1,3.50,0.02,V\n2450001.1,3.65,0.03,V\n"
    session = FakeSession(FakeResponse(text, url="https://www.aavso.org/vsx/index.php?..."))
    source = AAVSOSource(session=session)
    data = source.fetch("Delta Cep", "V", max_points=1234)
    assert data.source == "AAVSO"
    assert data.band == "V"
    assert data.time_system == "JD"
    assert data.n_points == 2
    assert np.allclose(data.magnitude, [3.50, 3.65])
    assert session.last_params["view"] == "api.object"
    assert session.last_params["ident"] == "Delta Cep"
    assert session.last_params["data"] == 1234
    assert session.last_params["mtype"] == "std"




def test_aavso_xml_wrapped_csv_response_is_parsed():
    text = """<?xml version="1.0" encoding="UTF-8"?>
<VSXObject>
  <Name>Delta Cep</Name>
  <Data><![CDATA[
JD,Magnitude,Uncertainty,Band
2450000.1,3.50,0.02,V
2450001.1,3.65,0.03,V
]]></Data>
</VSXObject>
"""
    data = AAVSOSource.parse_csv(text, "Delta Cep", "V", "https://example.test/aavso")
    assert data.n_points == 2
    assert np.allclose(data.time, [2450000.1, 2450001.1])
    assert np.allclose(data.magnitude, [3.50, 3.65])
    assert data.band == "V"

def test_aavso_unrecognized_no_data_is_not_silently_parsed():
    with pytest.raises(TargetNotFound):
        AAVSOSource.parse_csv("No data found", "NoSuchStar", "V", "https://example")


def test_ogle_exact_id_constructs_public_photometry_url_and_parses_three_columns():
    text = "2450000.1 15.21 0.01\n2450001.1 15.34 0.02\n"
    session = FakeSession(FakeResponse(text))
    source = OGLESource(session=session)
    data = source.fetch("OGLE-LMC-CEP-0005", "I")
    assert data.source == "OGLE-IV OCVS"
    assert data.time_system == "HJD"
    assert data.band == "I"
    assert data.n_points == 2
    assert session.last_url.endswith("/lmc/cep/phot/I/OGLE-LMC-CEP-0005.dat")


def test_ogle_missing_band_is_target_not_found():
    session = FakeSession(FakeResponse("not found", status_code=404))
    source = OGLESource(session=session)
    with pytest.raises(TargetNotFound):
        source.fetch("OGLE-LMC-CEP-0005", "V")


def test_photometry_saves_provenance_sidecar(tmp_path):
    data = PhotometryData(
        np.array([2., 1.]), np.array([12.2, 12.1]), np.array([.02, .03]),
        target="X", source="TEST", band="V", time_system="JD",
        provenance_url="https://example.test/data", source_identifier="X-1",
    )
    path = data.save_csv(tmp_path / "raw.csv")
    assert (tmp_path / "raw.csv").exists()
    sidecar = tmp_path / "raw.metadata.json"
    assert sidecar.exists()
    assert "https://example.test/data" in sidecar.read_text()
    assert data.time.tolist() == [1.0, 2.0]


def test_ogle_ident_coordinate_parser_handles_space_sexagesimal_layout():
    row = "OGLE-GD-RRLYR-00001 RRab 17 51 12.34 -29 59 08.1 15.1 16.0"
    parsed = OGLESource._parse_ident_coordinates(row)
    assert parsed is not None
    ident, ra, dec = parsed
    assert ident == "OGLE-GD-RRLYR-00001"
    assert 267 < ra < 269
    assert -31 < dec < -29
