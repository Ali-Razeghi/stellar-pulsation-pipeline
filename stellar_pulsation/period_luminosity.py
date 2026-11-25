"""Named, citable Period-Luminosity relations used as consistency diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PLRelation:
    a: float
    b: float
    sigma: float
    band: str
    reference: str
    @classmethod
    def benedict2007_v(cls):
        # M_V = -2.43 (logP - 1) - 4.05 = -2.43 logP - 1.62
        return cls(-2.43,-1.62,0.10,"V","Benedict et al. 2007, AJ 133, 1810; Galactic fundamental-mode Cepheids")
    def absolute_magnitude(self,period_days):
        if period_days<=0: raise ValueError("period must be positive")
        return float(self.a*np.log10(period_days)+self.b)
    def distance_modulus_from_mean_mag(self,period_days,mean_apparent_mag,A_V=0.0):
        return float(mean_apparent_mag-A_V-self.absolute_magnitude(period_days))
    def distance_pc_from_mean_mag(self,period_days,mean_apparent_mag,A_V=0.0):
        mu=self.distance_modulus_from_mean_mag(period_days,mean_apparent_mag,A_V); return float(10**((mu+5)/5))
    def consistency_check(self,period_days,mean_apparent_mag,independent_distance_modulus,A_V=0.0,mu_uncertainty=0.0):
        mu_pl=self.distance_modulus_from_mean_mag(period_days,mean_apparent_mag,A_V); delta=independent_distance_modulus-mu_pl
        sig=float(np.hypot(self.sigma,mu_uncertainty))
        return {"mu_independent":float(independent_distance_modulus),"mu_from_pl":mu_pl,"delta_mu":float(delta),
                "combined_sigma_mag":sig,"consistent_within_2sigma":bool(abs(delta)<=2*sig),"band":self.band,"reference":self.reference,
                "interpretation":"empirical consistency diagnostic, not an independent validation of a physical pulsation model"}
