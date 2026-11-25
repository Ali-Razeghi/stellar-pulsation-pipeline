import numpy as np
import pytest
from stellar_pulsation.bayesian_fit import run_mcmc,run_constrained_physical_mcmc,GaussianPrior
from stellar_pulsation.physical_model import delta_cep_like
from stellar_pulsation.data_ingestion import load_ascii_photometry

def test_photometric_mcmc_runs_and_reports_rhat():
    rng=np.random.default_rng(1); ph=np.linspace(0,1,120,endpoint=False); err=np.full_like(ph,.03); y=10+.2*np.sin(2*np.pi*ph)+rng.normal(0,.03,len(ph))
    _,post=run_mcmc(ph,y,err,n_harmonics=1,n_walkers=12,n_steps=180,burn_in=60,seed=1)
    assert 'mean' in post and '_diagnostics' in post and 'split_rhat' in post['_diagnostics']

def test_physical_mcmc_requires_external_priors():
    m=delta_cep_like(); t=np.linspace(0,m.period,20); y=m.apparent_magnitude(t); e=np.full_like(y,.02)
    with pytest.raises(ValueError): run_constrained_physical_mcmc(t,y,e,m,priors={})

def test_ascii_loader(tmp_path):
    p=tmp_path/'x.txt'; np.savetxt(p,np.array([[1,10,.1],[2,11,.2]]))
    t,m,e=load_ascii_photometry(str(p)); assert np.allclose(t,[1,2]) and np.allclose(e,[.1,.2])
