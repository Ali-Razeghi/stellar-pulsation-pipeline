"""Bayesian photometric-shape inference and optional constrained physical inference.

The default MCMC samples Fourier light-curve coefficients, avoiding the
R0--T0--distance degeneracy of single-band photometry.  A separate constrained
physical sampler is provided only when external Gaussian priors on R0, T0 and
mu are supplied.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import emcee
from .fourier_decomposition import _design_matrix
from .physical_model import PulsationModel

def split_rhat(chain):
    """Classic split-Rhat diagnostic using walker halves as chains."""
    chain=np.asarray(chain,float)  # steps, walkers, ndim
    nstep,nwalk,ndim=chain.shape
    half=nstep//2
    if half<10 or nwalk<4: return np.full(ndim,np.nan)
    pieces=np.concatenate([np.transpose(chain[:half],(1,0,2)),np.transpose(chain[-half:],(1,0,2))],axis=0)
    n=pieces.shape[1]; means=pieces.mean(axis=1); vars_=pieces.var(axis=1,ddof=1)
    B=n*means.var(axis=0,ddof=1); W=vars_.mean(axis=0); varhat=(n-1)/n*W+B/n
    return np.sqrt(varhat/W)

def run_mcmc(phase,y,yerr,n_harmonics=3,n_walkers=32,n_steps=2000,burn_in=500,seed=42):
    """MCMC posterior for mean + cosine/sine Fourier coefficients."""
    phase=np.asarray(phase,float); y=np.asarray(y,float); yerr=np.asarray(yerr,float)
    X=_design_matrix(phase,n_harmonics); beta=np.linalg.lstsq(X/yerr[:,None],y/yerr,rcond=None)[0]
    ndim=len(beta); min_walkers=2*ndim+2
    if n_walkers<min_walkers: n_walkers=min_walkers
    data_span=max(np.ptp(y),0.1)
    def logp(theta):
        if not (np.min(y)-2*data_span < theta[0] < np.max(y)+2*data_span): return -np.inf
        if np.any(np.abs(theta[1:])>2*data_span): return -np.inf
        resid=(y-X@theta)/yerr
        return -0.5*np.sum(resid**2+np.log(2*np.pi*yerr**2))
    rng=np.random.default_rng(seed); scale=np.maximum(np.abs(beta)*0.01,1e-4); pos=beta+scale*rng.normal(size=(n_walkers,ndim))
    sampler=emcee.EnsembleSampler(n_walkers,ndim,logp); sampler.run_mcmc(pos,n_steps,progress=False)
    flat=sampler.get_chain(discard=burn_in,flat=True); names=['mean']+[z for k in range(1,n_harmonics+1) for z in (f'c{k}',f's{k}')]
    out={}
    for i,name in enumerate(names):
        lo,med,hi=np.percentile(flat[:,i],[16,50,84]); out[name]={"median":float(med),"minus":float(med-lo),"plus":float(hi-med)}
    try: tau=sampler.get_autocorr_time(quiet=True).tolist()
    except Exception: tau=None
    rhat=split_rhat(sampler.get_chain(discard=burn_in))
    out['_diagnostics']={"autocorr_time":tau,"split_rhat":rhat.tolist(),"rhat_max":float(np.nanmax(rhat)) if np.any(np.isfinite(rhat)) else None,
                         "converged_rhat_1p05":bool(np.all(rhat[np.isfinite(rhat)]<1.05)) if np.any(np.isfinite(rhat)) else False}
    return sampler,out

@dataclass(frozen=True)
class GaussianPrior:
    mean: float
    sigma: float
    def __post_init__(self):
        if self.sigma<=0: raise ValueError("prior sigma must be positive")

def run_constrained_physical_mcmc(t,m_obs,m_err,initial_model,priors,n_walkers=32,n_steps=2500,burn_in=600,seed=42):
    """Fit R0, T0 and distance modulus only with mandatory external priors.

    ``priors`` must contain GaussianPrior entries for R0, T0 and
    distance_modulus.  Only these three mean/zero-point parameters are sampled;
    the radius and temperature harmonic amplitudes/phases in ``initial_model``
    are held fixed throughout the fit.  This API deliberately refuses
    unconstrained single-band physical inference because R0, T0 and distance
    modulus are degenerate without external information.
    """
    required={'R0','T0','distance_modulus'}
    if set(priors)<required: raise ValueError("external Gaussian priors for R0, T0 and distance_modulus are required")
    t=np.asarray(t,float); obs=np.asarray(m_obs,float); err=np.asarray(m_err,float)
    theta0=np.array([initial_model.R0,initial_model.T0,initial_model.distance_modulus],float)
    def logp(theta):
        model=deepcopy(initial_model); model.R0,model.T0,model.distance_modulus=theta
        try: pred=model.apparent_magnitude(t)
        except ValueError: return -np.inf
        lp=0.0
        for val,key in zip(theta,('R0','T0','distance_modulus')):
            p=priors[key]; lp += -0.5*((val-p.mean)/p.sigma)**2-np.log(p.sigma*np.sqrt(2*np.pi))
        resid=(obs-pred)/err
        return lp-0.5*np.sum(resid**2+np.log(2*np.pi*err**2))
    rng=np.random.default_rng(seed); scale=np.array([priors['R0'].sigma,priors['T0'].sigma,priors['distance_modulus'].sigma])*0.05
    n_walkers=max(n_walkers,8); pos=theta0+scale*rng.normal(size=(n_walkers,3))
    sampler=emcee.EnsembleSampler(n_walkers,3,logp); sampler.run_mcmc(pos,n_steps,progress=False)
    flat=sampler.get_chain(discard=burn_in,flat=True); out={}
    for i,name in enumerate(('R0','T0','distance_modulus')):
        lo,med,hi=np.percentile(flat[:,i],[16,50,84]); out[name]={"median":float(med),"minus":float(med-lo),"plus":float(hi-med)}
    out['_diagnostics']={"split_rhat":split_rhat(sampler.get_chain(discard=burn_in)).tolist()}
    return sampler,out
