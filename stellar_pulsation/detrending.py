"""Robust clipping and detrending utilities for variable-star photometry."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import LSQUnivariateSpline

def _mad_sigma(x):
    x=np.asarray(x,float); med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med)); return 1.4826*mad

def sampling_irregularity(t):
    dt=np.diff(np.sort(np.asarray(t,float))); dt=dt[dt>0]
    return float(np.std(dt)/np.median(dt)) if len(dt)>1 else 0.0

def sigma_clip(t,y,yerr=None,sigma=5.0,max_iter=5,period=None,n_phase_bins=30):
    """MAD-based clipping; with ``period`` clips residuals around a phase-binned template."""
    t=np.asarray(t,float); y=np.asarray(y,float); err=np.full_like(y,np.nan) if yerr is None else np.asarray(yerr,float)
    if len(t)!=len(y) or len(err)!=len(y): raise ValueError("array lengths differ")
    mask=np.isfinite(t)&np.isfinite(y)
    for _ in range(max_iter):
        if period is None:
            center=np.nanmedian(y[mask]); resid=y-center
        else:
            phase=np.mod(t/period,1.0); edges=np.linspace(0,1,n_phase_bins+1); template=np.full_like(y,np.nan)
            global_med=np.nanmedian(y[mask])
            for i in range(n_phase_bins):
                b=mask&(phase>=edges[i])&(phase<edges[i+1]); val=np.nanmedian(y[b]) if np.any(b) else global_med
                template[(phase>=edges[i])&(phase<edges[i+1])]=val
            resid=y-template
        scale=_mad_sigma(resid[mask])
        if not np.isfinite(scale) or scale==0: break
        new=mask&(np.abs(resid)<=sigma*scale)
        if np.array_equal(new,mask): break
        mask=new
    return t[mask],y[mask],err[mask]

def savgol_detrend(t,y,window_days,polyorder=2,irregularity_threshold=0.15):
    t=np.asarray(t,float); y=np.asarray(y,float)
    irr=sampling_irregularity(t)
    if irr>irregularity_threshold:
        warnings.warn("irregular sampling: Savitzky-Golay assumes near-uniform cadence; using spline_detrend is recommended",RuntimeWarning)
    if len(t)<5: return y-np.median(y)+np.median(y)
    order=np.argsort(t); ts=t[order]; ys=y[order]; dt=np.median(np.diff(ts)); n=int(round(window_days/dt)); n=max(n,polyorder+2)
    if n%2==0: n+=1
    if n>len(y): n=len(y) if len(y)%2 else len(y)-1
    if n<=polyorder: return y.copy()
    trend_sorted=savgol_filter(ys,n,polyorder); out=np.empty_like(y); out[order]=ys-trend_sorted+np.median(ys); return out

def spline_detrend(t,y,window_days):
    """Slow-trend spline for irregular samples with knots separated by ``window_days``.

    Knots are deliberately much more widely spaced than a pulsation period in
    normal use, so the spline follows long-timescale systematics rather than
    fitting individual pulsation cycles.
    """
    t=np.asarray(t,float); y=np.asarray(y,float); order=np.argsort(t); ts=t[order]; ys=y[order]
    if len(t)<8 or np.ptp(ts) <= window_days:
        return y.copy()
    knots=np.arange(ts[0]+window_days,ts[-1]-window_days+1e-12,window_days)
    if len(knots)==0:
        return y.copy()
    try:
        spl=LSQUnivariateSpline(ts,ys,knots,k=min(2,len(knots)))
        trend=spl(ts)
    except Exception:
        coeff=np.polyfit(ts-ts.mean(),ys,1); trend=np.polyval(coeff,ts-ts.mean())
    out=np.empty_like(y); out[order]=ys-trend+np.median(trend); return out

def auto_detrend(t,y,window_days,irregularity_threshold=0.15):
    return spline_detrend(t,y,window_days) if sampling_irregularity(t)>irregularity_threshold else savgol_detrend(t,y,window_days)
