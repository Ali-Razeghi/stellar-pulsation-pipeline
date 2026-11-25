import numpy as np
from stellar_pulsation.physical_model import delta_cep_like
from stellar_pulsation.synthetic_data import generate_light_curve
from stellar_pulsation.period_search import lomb_scargle_period
from stellar_pulsation.fourier_decomposition import phase_fold,fit_fourier_series


def test_lomb_scargle_recovers_known_period_after_refinement():
    m=delta_cep_like(); t,y,e=generate_light_curve(m,n_cycles=10,photometric_noise_mag=.01,seed=1)
    r=lomb_scargle_period(t,y,e,min_period=1,max_period=15,samples_per_peak=8,refine_peak=True)
    assert r['peak_refined'] is True
    assert abs(r['best_period_days']-m.period)<.02


def test_top_periods_remain_power_ranked():
    m=delta_cep_like(); t,y,e=generate_light_curve(m,n_cycles=8,photometric_noise_mag=.02,gap_fraction=.15,seed=4)
    r=lomb_scargle_period(t,y,e,min_period=1,max_period=15)
    powers=[p for _,p in r['top_periods_by_power']]
    assert powers==sorted(powers,reverse=True)


def test_weighted_linear_fourier_recovers_known_coefficients():
    rng=np.random.default_rng(3); ph=np.sort(rng.random(1000)); err=np.linspace(.01,.04,len(ph))
    y=10 + .3*np.cos(2*np.pi*ph)+.4*np.sin(2*np.pi*ph)+.1*np.cos(4*np.pi*ph)-.05*np.sin(4*np.pi*ph)
    obs=y+rng.normal(0,err)
    fit=fit_fourier_series(ph,obs,err,n_harmonics=2)
    assert np.isclose(fit['cosine_coeffs'][0],.3,atol=.01)
    assert np.isclose(fit['sine_coeffs'][0],.4,atol=.01)
    assert all(a>=0 for a in fit['amplitudes'])
    assert fit['phase_convention'].startswith('A_k cos')


def test_phase_fold_validation_and_range():
    ph=phase_fold(np.array([0,2.5,5,7.3]),5)
    assert np.all((ph>=0)&(ph<1))
