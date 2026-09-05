"""
Campo complejo de dispersion Psi(x,y) por superposicion de ondas parciales.

CONSTRUCCION (reutiliza integramente los solvers existentes)
-------------------------------------------------------------
Para cada modo azimutal m, `radial_profile` integra

    d^2 H_m / dr_*^2 + V_h(r; omega, m) H_m = 0

desde el horizonte con la condicion PURAMENTE ENTRANTE
H_m ~ exp(-i k_H r_*), normalizada como H_m(r_0) = 1 con r_*(r_0) := 0.

Esa normalizacion es "amplitud unidad en el horizonte", NO "amplitud unidad
incidente desde el infinito". Para superponer una onda plana incidente hay que
renormalizar cada modo por su amplitud entrante asintotica A_in^(m), que
`integrate_radial` ya calcula:

    Psi(r,phi) = sum_m  c_m * [H_m(r) / A_in^(m)] * r^(-1/2) * exp(i m phi)

con c_m = i^m, los coeficientes del desarrollo de una onda plana en 2D
(e^{ikx} = sum_m i^m J_m(kr) e^{i m phi}).

IMPORTANTE — la normalizacion NO se da por buena a priori. El test de simetria
con B=0 (ver analysis.SymmetryTest) es el arbitro: con B=0 el potencial depende
de m solo a traves de m^2, luego H_{-m} = H_m y A_in^{-m} = A_in^{m}; como
ademas c_m = c_{-m} = i^m, el campo DEBE ser exactamente simetrico bajo
phi -> -phi. Cualquier error de normalizacion o de indexado rompe esa simetria.

Con B != 0 el potencial contiene (omega - m B/r^2)^2, que distingue +m de -m:
la asimetria resultante es el arrastre de marco, y es fisica.

CONSISTENCIA DE NORMALIZACION: `radial_profile` e `integrate_radial` arrancan
con condiciones iniciales identicas (mismo r_0 = r_h(1+delta), mismo r_*=0,
H=1). Por eso H_m(r) y A_in^(m) son de la MISMA solucion siempre que se use el
mismo `delta`; esta clase lo garantiza pasando un unico delta a ambas.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, Optional

import numpy as np

from ..numerics.radial import integrate_radial, radial_profile
from ..physics.background import DrainingBathtub
from ..physics.deformation import Deformation, NullDeformation


@dataclass
class PartialWave:
    """Una onda parcial resuelta: H_m(r) ya renormalizada."""

    m: int
    r: np.ndarray             # malla radial de la integracion
    H: np.ndarray             # H_m(r) con normalizacion de horizonte
    A_in: complex             # amplitud entrante asintotica
    A_out: complex

    @property
    def H_norm(self) -> np.ndarray:
        """H_m(r)/A_in^(m): amplitud unidad incidente desde el infinito."""
        return self.H / self.A_in

    @property
    def reflection(self) -> complex:
        return self.A_out / self.A_in


class ScatteringField:
    """Campo complejo Psi(x,y) de una onda plana dispersada por el vortice.

    Cachea las ondas parciales: cambiar M o la malla espacial NO reintegra las
    ODEs si omega, el fondo y la deformacion no cambiaron.
    """

    def __init__(self, bg: DrainingBathtub, omega: float,
                 defo: Optional[Deformation] = None,
                 m_max: int = 15, delta: float = 1e-8,
                 r_max: float = 60.0, n_out: int = 1200,
                 A_in_extract: float = 400.0,
                 backend: str = "auto"):
        """backend: 'auto' | 'cuda' | 'scipy'.

        'auto' usa el kernel CUDA si CuPy esta disponible (resuelve TODOS los
        modos en un solo lanzamiento) y cae a scipy si no. El resultado es el
        mismo campo; ver src/field/cuda_radial.validate_against_scipy.
        """
        self.bg = bg
        self.omega = float(omega)
        self.defo = defo if defo is not None else NullDeformation(r_h=bg.r_h)
        self.m_max = int(m_max)
        self.delta = float(delta)
        self.r_max = float(r_max)
        self.n_out = int(n_out)
        self.A_in_extract = float(A_in_extract)
        self.backend = backend
        self._waves: Dict[int, PartialWave] = {}
        self._batch_done_for = None     # (m_max, omega, backend)

    # -------------------------------------------------------- resolucion
    def _use_cuda(self) -> bool:
        if self.backend == "scipy":
            return False
        try:
            from .cuda_radial import HAVE_CUPY
        except ImportError:
            return False
        if self.backend == "cuda" and not HAVE_CUPY:
            raise RuntimeError("backend='cuda' pero CuPy no esta instalado")
        return HAVE_CUPY

    def _solve_batch(self, m_max: int):
        """Resuelve TODOS los modos de golpe con el kernel CUDA.

        NOTA sobre r_extract: NO se reduce por debajo de ~400. Medido, bajar a
        120 introduce un 9% de error en A_in, y a 80 un 15%, porque
        V_h - k^2 decae solo como 1/r^2 y la asintotica de onda plana desnuda
        converge despacio. El test de simetria B=0 NO detecta este error
        (es simetrico en +-m), asi que no sirve de control aqui.
        En el kernel el coste de llegar a r=400 es despreciable.
        """
        from .cuda_radial import solve_modes_cuda
        ms = list(range(-m_max, m_max + 1))
        res = solve_modes_cuda(
            self.bg, self.omega, ms, defo=self.defo, delta=self.delta,
            r_profile=self.r_max, r_extract=self.A_in_extract,
            n_steps=200000, record_every=60)
        for j, m in enumerate(ms):
            self._waves[m] = PartialWave(
                m=m, r=res.r, H=res.H[:, j],
                A_in=complex(res.A_in[j]), A_out=complex(res.A_out[j]))
        self._batch_done_for = (m_max, self.omega, "cuda")

    # ------------------------------------------------------------ ondas
    def partial_wave(self, m: int) -> PartialWave:
        """Resuelve (o recupera de cache) la onda parcial del modo m."""
        if m in self._waves:
            return self._waves[m]

        if self._use_cuda():
            need = max(abs(m), self.m_max)
            self._solve_batch(need)
            if m in self._waves:
                return self._waves[m]

        r, H = radial_profile(self.omega, m, self.bg, self.defo,
                              n_out=self.n_out, delta=self.delta,
                              r_max=self.r_max)
        res = integrate_radial(self.omega, m, self.bg, self.defo,
                               delta=self.delta, r_max=self.A_in_extract)
        pw = PartialWave(m=m, r=r, H=H, A_in=res.A_in, A_out=res.A_out)
        self._waves[m] = pw
        return pw

    def modes(self, m_max: Optional[int] = None):
        M = self.m_max if m_max is None else int(m_max)
        return range(-M, M + 1)

    @staticmethod
    def plane_wave_coeff(m: int) -> complex:
        """c_m para una onda plana incidente, EN NUESTRA NORMALIZACION.

        NO es i^m. Ese es el coeficiente del desarrollo en Bessel,
        e^{ikx} = sum_m i^m J_m(kr) e^{imphi}, y depende de la paridad
        J_{-m} = (-1)^m J_m. Nuestro H_m NO tiene esa paridad: con B=0 la
        ecuacion depende de m solo por m^2 y k_H = omega/c no depende de m,
        de modo que H_{-m} = H_m exactamente (sin factor de signo).

        Empalmando la parte ENTRANTE del desarrollo de la onda plana contra
        nuestra normalizacion H_m/A_in -> e^{-i k r_*}:

            J_m(kr) ~ sqrt(2/(pi k r)) cos(kr - m pi/2 - pi/4)
            parte entrante  ~ (1/2) sqrt(2/(pi k r)) e^{-ikr} e^{i(m pi/2 + pi/4)}
            i^m * e^{i m pi/2} = e^{i m pi} = (-1)^m

        luego  c_m = (-1)^m  (por un factor global irrelevante para el color).

        Cumple c_m = c_{-m}, que es lo que hace simetrico el caso B=0.
        Verificado numericamente en analysis.SymmetryTest.
        """
        return 1.0 if (m % 2 == 0) else -1.0

    # ------------------------------------------------------------ evaluacion
    def evaluate(self, r_grid: np.ndarray, phi_grid: np.ndarray,
                 m_max: Optional[int] = None) -> np.ndarray:
        """Psi sobre una malla polar. r_grid, phi_grid con la misma forma.

        Devuelve un array complejo con esa forma. Los puntos con
        r < r_h(1+delta) o r > r_max quedan como NaN (fuera del dominio
        resuelto) — el coloreado los trata explicitamente.
        """
        M = self.m_max if m_max is None else int(m_max)
        r_grid = np.asarray(r_grid, dtype=float)
        phi_grid = np.asarray(phi_grid, dtype=float)

        psi = np.zeros(r_grid.shape, dtype=complex)
        r_lo = self.bg.r_h * (1.0 + self.delta)
        valid = (r_grid >= r_lo) & (r_grid <= self.r_max)

        # r^{-1/2}: finito en todo el dominio valido (r >= r_h > 0)
        inv_sqrt_r = np.zeros_like(r_grid)
        inv_sqrt_r[valid] = 1.0 / np.sqrt(r_grid[valid])

        for m in range(-M, M + 1):
            pw = self.partial_wave(m)
            Hn = pw.H_norm
            # interpolacion compleja: parte real e imaginaria por separado
            Hr = np.interp(r_grid, pw.r, Hn.real, left=np.nan, right=np.nan)
            Hi = np.interp(r_grid, pw.r, Hn.imag, left=np.nan, right=np.nan)
            H_interp = Hr + 1j * Hi
            psi += (self.plane_wave_coeff(m) * H_interp * inv_sqrt_r
                    * np.exp(1j * m * phi_grid))

        psi[~valid] = np.nan
        return psi

    def evaluate_cartesian(self, X: np.ndarray, Y: np.ndarray,
                           m_max: Optional[int] = None) -> np.ndarray:
        """Psi sobre una malla cartesiana."""
        R = np.hypot(X, Y)
        PHI = np.arctan2(Y, X)
        return self.evaluate(R, PHI, m_max=m_max)

    # ------------------------------------------------------------ utilidades
    def instantaneous(self, psi: np.ndarray, t: float) -> np.ndarray:
        """Re[Psi e^{-i omega t}]: la onda instantanea (altura del relieve)."""
        return np.real(psi * np.exp(-1j * self.omega * t))

    def clear_cache(self):
        self._waves.clear()