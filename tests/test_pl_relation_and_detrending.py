import numpy as np
import pytest
from stellar_pulsation.period_luminosity import PLRelation
from stellar_pulsation.detrending import sigma_clip,savgol_detrend,auto_detrend,sampling_irregularity

def test_named_benedict_relation():
    pl=PLRelation.benedict2007_v()
    assert np.isclose(pl.absolute_magnitude(10),-4.05,atol=1e-12)
    assert pl.absolute_magnitude(30)<pl.absolute_magnitude(3)

def test_pl_roundtrip():
    pl=PLRelation.benedict2007_v(); P=5.366531; mu=7.19; av=.25; mean=pl.absolute_magnitude(P)+mu+av
    assert np.isclose(pl.distance_modulus_from_mean_mag(P,mean,av),mu)

def test_mad_clip_removes_extreme_outlier_without_destroying_variability():
    t=np.linspace(0,20,500); y=10+.4*np.sin(2*np.pi*t/5); y[100]=30; e=np.full_like(y,.02)
    t2,y2,e2=sigma_clip(t,y,e,sigma=6)
    assert len(y2)==499 and np.max(y2)<11

def test_phase_aware_clip():
    t=np.linspace(0,30,900); y=10+.5*np.sin(2*np.pi*t/5); y[222]+=3; e=np.full_like(y,.02)
    t2,y2,_=sigma_clip(t,y,e,sigma=5,period=5)
    assert len(y2)<len(y) and len(y2)>890

def test_savgol_warns_on_irregular_sampling():
    rng=np.random.default_rng(2); t=np.sort(rng.uniform(0,30,300)); y=np.sin(t)
    assert sampling_irregularity(t)>.15
    with pytest.warns(RuntimeWarning): savgol_detrend(t,y,10)
    assert auto_detrend(t,y,10).shape==y.shape
