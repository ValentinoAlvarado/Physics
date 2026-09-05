"""
Familia de perfiles radiales deformados.

Perfil deformado:

    v_r(r) = -(A/r) [1 + h(r)]

Condiciones que preservan la posicion del horizonte y la gravedad superficial
(derivadas y verificadas analiticamente):

    h(r_h)  = 0      -> preserva r_h = A/c
    h'(r_h) = 0      -> preserva kappa_h = c^2/A

LIMITACION DECLARADA (opcion cinematica / fenomenologica):
la ecuacion de continuidad estacionaria en 2D, rho*v_r*r = const, con rho
constante FUERZA v_r ~ 1/r. Por tanto, cualquier h(r) != 0 no corresponde a un
flujo estacionario realizable con densidad de fondo homogenea. Se adopta la
deformacion como parametrizacion puramente geometrica de la metrica efectiva.
Esto debe declararse explicitamente como limitacion del modelo, y acota lo que
puede afirmarse: se estudia la identificabilidad de la GEOMETRIA EFECTIVA
parametrizada por h, no de un perfil de flujo fisicamente realizable bajo
rho = const.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Deformation(ABC):
    """Interfaz comun de las familias de deformacion h(r)."""

    r_h: float

    @abstractmethod
    def h(self, r):
        """h(r), adimensional."""

    @abstractmethod
    def dh(self, r):
        """dh/dr, dimensiones [L^-1]."""

    @property
    @abstractmethod
    def theta(self) -> np.ndarray:
        """Vector de parametros de forma (para el analisis de Fisher)."""

    @property
    @abstractmethod
    def theta_names(self) -> list[str]:
        """Nombres de los parametros de forma, en el mismo orden que `theta`."""

    def check_boundary_conditions(self, atol: float = 1e-12) -> None:
        """Verifica h(r_h)=0 y h'(r_h)=0. Lanza AssertionError si fallan."""
        h0 = float(self.h(self.r_h))
        dh0 = float(self.dh(self.r_h))
        assert abs(h0) < atol, f"h(r_h) = {h0:.3e} != 0"
        assert abs(dh0) < atol, f"h'(r_h) = {dh0:.3e} != 0"


@dataclass
class NullDeformation(Deformation):
    """h(r) = 0. Fondo no deformado, usado como caso de referencia."""

    r_h: float

    def h(self, r):
        return np.zeros_like(np.asarray(r, dtype=float))

    def dh(self, r):
        return np.zeros_like(np.asarray(r, dtype=float))

    @property
    def theta(self) -> np.ndarray:
        return np.array([])

    @property
    def theta_names(self) -> list[str]:
        return []


@dataclass
class PolynomialDeformation(Deformation):
    """Esquema polinomial:

        h(r) = sum_{k=2}^{N} a_k * ((r - r_h)/r)^k

    Cada termino con k >= 2 anula h y h' en r_h de forma exacta, para
    cualquier combinacion de coeficientes.
    """

    r_h: float
    a: np.ndarray  # a[i] corresponde a k = i + 2

    def __post_init__(self) -> None:
        self.a = np.atleast_1d(np.asarray(self.a, dtype=float))

    def _u(self, r):
        r = np.asarray(r, dtype=float)
        return (r - self.r_h) / r

    def h(self, r):
        u = self._u(r)
        out = np.zeros_like(u)
        for i, ak in enumerate(self.a):
            out = out + ak * u ** (i + 2)
        return out

    def dh(self, r):
        # u = 1 - r_h/r  =>  du/dr = r_h/r^2
        r = np.asarray(r, dtype=float)
        u = self._u(r)
        du = self.r_h / r**2
        out = np.zeros_like(u)
        for i, ak in enumerate(self.a):
            k = i + 2
            out = out + ak * k * u ** (k - 1) * du
        return out

    @property
    def theta(self) -> np.ndarray:
        return self.a.copy()

    @property
    def theta_names(self) -> list[str]:
        return [f"a_{i + 2}" for i in range(len(self.a))]


@dataclass
class GaussianDeformation(Deformation):
    """Esquema de deformacion localizada (gaussiana modulada):

        h(r) = eps * ((r - r_h)/r_h)^2 * exp(-sigma (r - r_h)^2)

    Parametros de forma: eps (amplitud), sigma (inverso del ancho al cuadrado).
    El factor cuadratico garantiza h(r_h) = h'(r_h) = 0 exactamente.
    """

    r_h: float
    eps: float
    sigma: float

    def h(self, r):
        r = np.asarray(r, dtype=float)
        d = r - self.r_h
        return self.eps * (d / self.r_h) ** 2 * np.exp(-self.sigma * d**2)

    def dh(self, r):
        r = np.asarray(r, dtype=float)
        d = r - self.r_h
        e = np.exp(-self.sigma * d**2)
        # d/dr [ d^2 e ] = (2d - 2 sigma d^3) e
        return self.eps / self.r_h**2 * (2 * d - 2 * self.sigma * d**3) * e

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.eps, self.sigma])

    @property
    def theta_names(self) -> list[str]:
        return ["eps", "sigma"]
