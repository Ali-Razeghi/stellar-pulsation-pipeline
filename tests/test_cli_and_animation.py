import warnings
import numpy as np
from stellar_pulsation.cli import build_parser, build_simulation_model
from stellar_pulsation.animation import _temperature_color


def test_cli_restores_custom_synthetic_parameters():
    args=build_parser().parse_args([
        '--source','synthetic','--r0','5.2','--period','0.5668','--t0','6500',
        '--t-amp','450','--radius-amp','0.25','--distance-modulus','10.4','--av','0.31'
    ])
    model=build_simulation_model(args)
    assert np.isclose(model.R0,5.2)
    assert np.isclose(model.period,.5668)
    assert np.isclose(model.T0,6500)
    assert np.isclose(model.distance_modulus,10.4)
    assert np.isclose(model.A_V,.31)
    assert model.photometric_template is None


def test_cli_av_changes_delta_cep_simulated_magnitude():
    p=build_parser()
    a=build_simulation_model(p.parse_args(['--av','0.25']))
    b=build_simulation_model(p.parse_args(['--av','0.55']))
    t=np.linspace(0,a.period,50)
    assert np.allclose(b.apparent_magnitude(t)-a.apparent_magnitude(t),.30)


def test_temperature_colour_uses_non_deprecated_colormap_api():
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=DeprecationWarning)
        rgba=_temperature_color(6000,5000,7000)
    assert len(rgba)==4
