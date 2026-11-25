import numpy as np
import pytest
from stellar_pulsation.physical_model import PulsationModel, FourierMode, delta_cep_like, TEFF_SUN
from stellar_pulsation.fourier_decomposition import phase_fold, morphology_metrics


def test_input_validation():
    with pytest.raises(ValueError): PulsationModel(R0=-1, period=1)
    with pytest.raises(ValueError): FourierMode(0,1,0)
    with pytest.raises(ValueError): PulsationModel(R0=1,period=1,modes=[FourierMode(1,2,0)])


def test_periodicity():
    m=delta_cep_like(); t=np.linspace(0,m.period,100)
    assert np.allclose(m.radius(t),m.radius(t+m.period),atol=1e-10)
    assert np.allclose(m.temperature(t),m.temperature(t+m.period),atol=1e-10)
    assert np.allclose(m.apparent_magnitude(t),m.apparent_magnitude(t+m.period),atol=1e-10)


def test_solar_stefan_boltzmann_sanity():
    m=PulsationModel(R0=1,period=1,T0=TEFF_SUN)
    assert np.isclose(m.luminosity(np.array([0.0]))[0],1.0,rtol=1e-12)


def test_distance_and_extinction_are_additive_zero_points():
    a=delta_cep_like(); b=delta_cep_like(); b.distance_modulus += 1.25; b.A_V += .17
    t=np.linspace(0,a.period,20)
    assert np.allclose(b.apparent_magnitude(t)-a.apparent_magnitude(t),1.42)


def test_delta_cep_full_morphology_regression():
    m=delta_cep_like(); t=np.linspace(0,m.period,12000,endpoint=False); mag=m.apparent_magnitude(t)
    metrics=morphology_metrics(phase_fold(t,m.period),mag,n_harmonics=3)
    assert np.isclose(np.ptp(m.radius(t)),4.81,atol=0.06)
    assert np.isclose(metrics['peak_to_peak_mag'],0.854439,atol=5e-5)
    assert np.isclose(metrics['R21'],0.415874,atol=5e-5)
    assert np.isclose(metrics['R31'],0.154702,atol=5e-5)
    assert np.isclose(metrics['phi21_rad'],4.363117,atol=5e-5)
    assert np.isclose(metrics['phi31_rad'],2.483084,atol=5e-5)
    assert np.isclose(metrics['rise_fraction'],0.2885,atol=5e-4)
    assert metrics['local_maxima_per_cycle']==1
    assert metrics['local_minima_per_cycle']==1
    assert metrics['total_local_extrema_per_cycle']==2


def test_bc_warns_when_extrapolated():
    with pytest.warns(RuntimeWarning):
        PulsationModel.bolometric_correction_v(np.array([4000.0]))
