"""Synthetic photometry for pipeline tests and demonstrations."""
from __future__ import annotations
import numpy as np
from .physical_model import PulsationModel

def generate_light_curve(model,n_cycles=6.0,cadence_minutes=30.0,photometric_noise_mag=0.02,gap_fraction=0.0,seed=42):
    if n_cycles<=0 or cadence_minutes<=0 or photometric_noise_mag<0 or not (0<=gap_fraction<1): raise ValueError('invalid synthetic-data parameters')
    rng=np.random.default_rng(seed); t=np.arange(0,n_cycles*model.period,cadence_minutes/1440.0); truth=model.apparent_magnitude(t); err=np.full_like(t,max(photometric_noise_mag,1e-6)); obs=truth+rng.normal(0,photometric_noise_mag,size=t.shape)
    if gap_fraction: keep=rng.random(len(t))>gap_fraction; t,obs,err=t[keep],obs[keep],err[keep]
    return t,obs,err
