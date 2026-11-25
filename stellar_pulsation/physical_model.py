"""Semi-empirical pulsation forward model.

Radius and effective-temperature curves are descriptive Fourier series, not
solutions of stellar hydrodynamics.  A model may optionally carry an empirical
photometric Fourier template.  When present, ``apparent_magnitude`` evaluates
that observed-band template and only applies changes in distance modulus and
extinction as additive zero-point shifts.  This keeps the observational
morphology benchmark separate from the illustrative radius/temperature model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import warnings
import numpy as np

TEFF_SUN = 5772.0
M_BOL_SUN = 4.74


@dataclass(frozen=True)
class FourierMode:
    """One harmonic ``A sin(2*pi*n*phase + phase)`` in curve units."""
    n: int
    amplitude: float
    phase: float

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("harmonic order n must be >= 1")
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("harmonic amplitude must be finite and non-negative")
        if not np.isfinite(self.phase):
            raise ValueError("harmonic phase must be finite")


@dataclass(frozen=True)
class PhotometricFourierTemplate:
    """Empirical magnitude template in cosine/sine coefficient form.

    ``mean_magnitude`` and coefficients refer to ``reference_distance_modulus``
    and ``reference_A_V``.  Changing those values on ``PulsationModel`` shifts
    only the magnitude zero point; it does not change the empirical morphology.
    """
    mean_magnitude: float
    cosine_coeffs: tuple[float, ...]
    sine_coeffs: tuple[float, ...]
    reference_distance_modulus: float
    reference_A_V: float
    label: str = "empirical photometric template"

    def __post_init__(self):
        if len(self.cosine_coeffs) != len(self.sine_coeffs) or not self.cosine_coeffs:
            raise ValueError("cosine_coeffs and sine_coeffs must have equal non-zero length")
        values = (self.mean_magnitude, self.reference_distance_modulus, self.reference_A_V,
                  *self.cosine_coeffs, *self.sine_coeffs)
        if not np.all(np.isfinite(values)):
            raise ValueError("photometric template values must be finite")
        if self.reference_A_V < 0:
            raise ValueError("reference_A_V must be non-negative")

    def evaluate(self, phase):
        phase = np.asarray(phase, dtype=float)
        out = np.full_like(phase, self.mean_magnitude, dtype=float)
        for k, (c, s) in enumerate(zip(self.cosine_coeffs, self.sine_coeffs), 1):
            out += c * np.cos(2 * np.pi * k * phase) + s * np.sin(2 * np.pi * k * phase)
        return out


@dataclass
class PulsationModel:
    """Semi-empirical radius/temperature model for a radial pulsator.

    ``modes`` are radius harmonics in R_sun and ``temperature_modes`` are
    effective-temperature harmonics in kelvin.  If ``photometric_template`` is
    ``None``, V magnitudes are synthesized from R(t), T_eff(t), bolometric
    correction, distance modulus, and A_V.  If a template is present, it is
    used for observed-band morphology while R(t)/T_eff(t) remain independent
    illustrative/semi-empirical curves.

    Single-band photometry does not identify R0, T0, and distance modulus
    independently.
    """
    R0: float
    period: float
    modes: list[FourierMode] = field(default_factory=list)
    T0: float = 6000.0
    temperature_modes: list[FourierMode] = field(default_factory=list)
    distance_modulus: float = 10.0
    A_V: float = 0.0
    photometric_template: PhotometricFourierTemplate | None = None

    def __post_init__(self):
        for name, value in (("R0", self.R0), ("period", self.period), ("T0", self.T0)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.distance_modulus):
            raise ValueError("distance_modulus must be finite")
        if not np.isfinite(self.A_V) or self.A_V < 0:
            raise ValueError("A_V must be finite and non-negative")
        self._validate_curves()

    @property
    def f0(self) -> float:
        return 1.0 / self.period

    def _series(self, t, mean, modes):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, mean, dtype=float)
        for mode in modes:
            out += mode.amplitude * np.sin(2 * np.pi * mode.n * t / self.period + mode.phase)
        return out

    def _validate_curves(self):
        grid = np.linspace(0.0, self.period, 4096, endpoint=False)
        if np.min(self._series(grid, self.R0, self.modes)) <= 0:
            raise ValueError("radius harmonics produce non-positive radius")
        if np.min(self._series(grid, self.T0, self.temperature_modes)) <= 0:
            raise ValueError("temperature harmonics produce non-positive temperature")

    def radius(self, t):
        return self._series(t, self.R0, self.modes)

    def temperature(self, t):
        return self._series(t, self.T0, self.temperature_modes)

    def radius_velocity(self, t):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for mode in self.modes:
            w = 2 * np.pi * mode.n / self.period
            out += mode.amplitude * w * np.cos(w * t + mode.phase)
        return out

    def luminosity(self, t):
        R = self.radius(t)
        T = self.temperature(t)
        if np.any(R <= 0) or np.any(T <= 0):
            raise ValueError("model generated non-positive radius or temperature")
        return R**2 * (T / TEFF_SUN)**4

    @staticmethod
    def bolometric_correction_v(T):
        """BC_V from Torres (2010), correcting Flower (1996).

        Values outside 3.70 <= log10(T) <= 4.10 trigger a warning because
        extrapolation of the published piecewise polynomial is not recommended.
        """
        T = np.asarray(T, dtype=float)
        if np.any(T <= 0):
            raise ValueError("temperature must be positive")
        logT = np.log10(T)
        if np.any((logT < 3.70) | (logT > 4.10)):
            warnings.warn(
                "BC_V is being extrapolated outside 3.70<=log10(T)<=4.10",
                RuntimeWarning,
            )
        c1 = (-0.190537291496456e5, 0.155144866764412e5,
              -0.421278819301717e4, 0.381476328422343e3)
        c2 = (-0.370510203809015e5, 0.385672629965804e5,
              -0.150651486316025e5, 0.261724637119416e4,
              -0.170623810323864e3)
        c3 = (-0.118115450538963e6, 0.137145973583929e6,
              -0.636233812100225e5, 0.147412923562646e5,
              -0.170587278406872e4, 0.788731721804990e2)
        out = np.empty_like(logT)
        for mask, coeff in (
            (logT < 3.70, c1),
            ((logT >= 3.70) & (logT < 3.90), c2),
            (logT >= 3.90, c3),
        ):
            x = logT[mask]
            out[mask] = sum(a * x**i for i, a in enumerate(coeff))
        return out

    bolometric_correction = bolometric_correction_v

    def absolute_v_magnitude(self, t):
        """Synthetic absolute V magnitude from the illustrative R/T model."""
        L = self.luminosity(t)
        T = self.temperature(t)
        Mbol = M_BOL_SUN - 2.5 * np.log10(L)
        return Mbol - self.bolometric_correction_v(T)

    def physical_apparent_magnitude(self, t):
        """Apparent V magnitude synthesized strictly from R/T physics inputs."""
        return self.absolute_v_magnitude(t) + self.distance_modulus + self.A_V

    def apparent_magnitude(self, t):
        """Observed-band model magnitude.

        For a model carrying an empirical photometric template (the delta Cep
        benchmark), return that template with additive distance/extinction
        shifts.  Otherwise use the R/T synthetic photometry.
        """
        if self.photometric_template is None:
            return self.physical_apparent_magnitude(t)
        phase = np.mod(np.asarray(t, dtype=float) / self.period, 1.0)
        template = self.photometric_template
        zero_point_shift = (
            self.distance_modulus - template.reference_distance_modulus
            + self.A_V - template.reference_A_V
        )
        return template.evaluate(phase) + zero_point_shift

    def summary(self):
        t = np.linspace(0, self.period, 4096, endpoint=False)
        return {
            "period_days": float(self.period),
            "mean_radius_Rsun": float(np.mean(self.radius(t))),
            "radius_peak_to_peak_Rsun": float(np.ptp(self.radius(t))),
            "mean_temperature_K": float(np.mean(self.temperature(t))),
            "temperature_peak_to_peak_K": float(np.ptp(self.temperature(t))),
            "mean_luminosity_Lsun": float(np.mean(self.luminosity(t))),
            "v_peak_to_peak_mag": float(np.ptp(self.apparent_magnitude(t))),
            "photometric_morphology_source": (
                self.photometric_template.label if self.photometric_template else "R/T synthetic photometry"
            ),
        }


def delta_cep_like() -> PulsationModel:
    """Semi-empirical delta-Cephei demonstration preset.

    The radius and temperature curves are illustrative/semi-empirical inputs.
    Johnson-V morphology is represented independently by the published N=3
    AAVSO Fourier benchmark (Elahi & Gull 2026, arXiv:2606.29543):
    P=5.366531 d, A0=3.903085 mag, A1=0.343421, A2=0.142820,
    A3=0.053128, phi21=4.363117 rad, phi31=2.483084 rad,
    DeltaV=0.854439 mag, and rise fraction=0.2885.

    Keeping the empirical photometric template separate prevents an arbitrary
    R^2*T^4 parameterization from being claimed as a unique physical
    reconstruction of the observed light curve.
    """
    radius_modes = [
        FourierMode(1, 2.24, 2.20713863),
        FourierMode(2, 0.32, 1.37843645),
        FourierMode(3, 0.11, 3.82648261),
    ]
    temperature_modes = [
        FourierMode(1, 460.216166, 0.0194613921),
        FourierMode(2, 136.209558, 1.43264724),
        FourierMode(3, 73.5929056, 0.949106859),
    ]
    template = PhotometricFourierTemplate(
        mean_magnitude=3.903085,
        cosine_coeffs=(-0.242286, -0.133974, -0.052745),
        sine_coeffs=(-0.243384, -0.049481, 0.006366),
        reference_distance_modulus=7.19,
        reference_A_V=0.25,
        label="AAVSO Johnson-V N=3 delta Cep empirical template (2026)",
    )
    return PulsationModel(
        R0=43.7,
        period=5.366531,
        modes=radius_modes,
        T0=5970.0,
        temperature_modes=temperature_modes,
        distance_modulus=7.19,
        A_V=0.25,
        photometric_template=template,
    )
