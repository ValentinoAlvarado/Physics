"""
Coloreado de dominio para el campo complejo Psi.

ADVERTENCIA MATEMATICA (importante para no sobreinterpretar la imagen)
----------------------------------------------------------------------
Psi NO es holomorfa en z = x + iy. Es solucion de una ecuacion de onda, no de
Cauchy-Riemann. Por tanto NO aplican aqui:
  - el principio del argumento;
  - la ortogonalidad entre curvas de nivel de |f| y de arg(f);
  - la conjugacion armonica de log|f| y arg(f).

Lo que SI se conserva, y es lo que justifica el coloreado de dominio, es que
en cualquier campo complejo continuo los ceros son DISLOCACIONES DE FASE con
carga topologica entera (Nye-Berry). Eso es topologia, no analiticidad.

ASIGNACION DE VARIABLES VISUALES
---------------------------------
    hue          -> arg(Psi)      (ciclico, como la fase)
    luminancia   -> |Psi|         (comprimido en log, normalizado por percentiles)
    bandas       -> arg(Psi) = const   (frentes de fase)
    (la altura del relieve, en el renderer, lleva Re[Psi e^{-i omega t}])

Normalizacion del modulo por PERCENTILES de la propia malla (no min/max):
unos pocos valores extremos cerca del horizonte destruirian el contraste en
el resto del dominio. Se comprime en log2 y se recorta a [q_lo, q_hi].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import hsv_to_rgb


@dataclass
class ColorConfig:
    q_lo: float = 2.0             # percentil inferior del modulo
    q_hi: float = 98.0            # percentil superior
    gamma: float = 0.6            # compresion de la luminancia
    saturation: float = 0.9
    contrast: float = 1.0         # ganancia global de luminancia
    # bandas de fase
    phase_bands: bool = True
    band_spacing: float = np.pi / 8
    band_strength: float = 0.30
    band_sharpness: float = 6.0
    # bandas de modulo (octavas de |Psi|)
    modulus_bands: bool = False
    modulus_band_strength: float = 0.18
    # color de los puntos fuera del dominio resuelto
    invalid_rgb: tuple = (0.02, 0.02, 0.03)
    # realce de ceros: oscurece donde |Psi| es muy pequeno
    darken_zeros: bool = True


class DomainColoring:
    """Convierte un campo complejo en RGB. Sin estado mutable salvo config."""

    def __init__(self, cfg: ColorConfig = None):
        self.cfg = cfg or ColorConfig()

    # ------------------------------------------------------------ interno
    def _normalized_log_modulus(self, psi):
        """log2|Psi| recortado a percentiles y llevado a [0,1]. NaN-safe."""
        a = np.abs(psi)
        finite = np.isfinite(a) & (a > 0)
        if not finite.any():
            return np.zeros(a.shape), finite
        la = np.full(a.shape, np.nan)
        la[finite] = np.log2(a[finite])
        lo, hi = np.nanpercentile(la[finite], [self.cfg.q_lo, self.cfg.q_hi])
        if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
            hi = lo + 1.0
        v = np.clip((la - lo) / (hi - lo), 0.0, 1.0)
        return v, finite

    # ------------------------------------------------------------ publico
    def rgb(self, psi: np.ndarray) -> np.ndarray:
        """Devuelve RGB (..., 3) en [0,1]. Los NaN salen como invalid_rgb."""
        cfg = self.cfg
        psi = np.asarray(psi)
        v_mod, finite = self._normalized_log_modulus(psi)

        ph = np.angle(psi)
        hue = (ph + np.pi) / (2.0 * np.pi)
        hue = np.nan_to_num(hue, nan=0.0)

        val = np.nan_to_num(v_mod, nan=0.0) ** cfg.gamma

        if cfg.modulus_bands:
            # una banda por octava de |Psi|
            with np.errstate(invalid="ignore"):
                la = np.log2(np.where(np.abs(psi) > 0, np.abs(psi), np.nan))
            frac = np.mod(np.nan_to_num(la, nan=0.0), 1.0)
            ring = 0.5 + 0.5 * np.cos(2 * np.pi * frac)
            val *= (1.0 - cfg.modulus_band_strength
                    + cfg.modulus_band_strength * ring)

        if cfg.phase_bands:
            val *= self.phase_band_factor(ph)

        if cfg.darken_zeros:
            # refuerza visualmente los ceros: |Psi| pequeno -> mas oscuro
            val *= (0.12 + 0.88 * np.nan_to_num(v_mod, nan=0.0))

        val = np.clip(val * cfg.contrast, 0.0, 1.0)
        sat = np.full(hue.shape, cfg.saturation)

        hsv = np.stack([hue, sat, val], axis=-1)
        hsv = np.nan_to_num(hsv, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = hsv_to_rgb(np.clip(hsv, 0.0, 1.0))
        rgb[~finite] = np.asarray(cfg.invalid_rgb)
        return rgb

    def phase_band_factor(self, ph: np.ndarray) -> np.ndarray:
        """Factor multiplicativo en [1-s, 1] con minimos en
        arg(Psi) = n * band_spacing. Son literalmente isolineas de fase
        constante, es decir los frentes de fase del campo."""
        cfg = self.cfg
        u = np.mod(np.nan_to_num(ph, nan=0.0), cfg.band_spacing) / cfg.band_spacing
        band = 0.5 + 0.5 * np.cos(2 * np.pi * u)
        band = band ** cfg.band_sharpness
        return 1.0 - cfg.band_strength + cfg.band_strength * band

    def legend_wheel(self, n: int = 256) -> np.ndarray:
        """Rueda de referencia: mapa fase->color para poner en la figura."""
        y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
        z = x + 1j * y
        z[np.abs(z) > 1] = np.nan
        return self.rgb(z)
