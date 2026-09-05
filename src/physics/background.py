"""
Geometria de fondo del agujero negro acustico rotante (draining bathtub).

Convenciones fijadas para todo el proyecto
------------------------------------------
- Coordenada tortuga r_* con dimensiones de longitud: [r_*] = [L].
- Potencial efectivo con dimensiones [L^-2].
- Numero de onda k = omega / c, con [k] = [L^-1].
- Gravedad superficial: kappa = |dv_r/dr|_{r_h},  [kappa] = [T^-1].
  ATENCION: la convencion de Visser es kappa_V = c*|dv_r/dr| (dimensiones de
  aceleracion). Difieren en un factor c. Si se reporta temperatura de Hawking
  analoga, declarar explicitamente cual se usa. Ver `surface_gravity_visser`.

Elemento de linea (gauge estatico, tipo Boyer-Lindquist), dimensionalmente
consistente:

    ds^2 = -(c^2 - (A^2+B^2)/r^2) dt^2
           + (1 - A^2/(c^2 r^2))^{-1} dr^2
           - 2 B dphi dt
           + r^2 dphi^2

con det(g) = -c^2 r^2  =>  sqrt(-g) = c r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrainingBathtub:
    """Parametros del fondo no deformado.

    Parameters
    ----------
    A : float
        Parametro de sumidero, [L^2 T^-1]. A > 0 => agujero negro acustico.
    B : float
        Parametro de circulacion del vortice, [L^2 T^-1].
    c : float
        Velocidad del sonido, constante y homogenea, [L T^-1].
    """

    A: float
    B: float
    c: float = 1.0

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError("A debe ser > 0 (sumidero / agujero negro acustico)")
        if self.c <= 0:
            raise ValueError("c debe ser > 0")

    # ---------------------------------------------------------------- radios

    @property
    def r_h(self) -> float:
        """Radio del horizonte acustico, r_h = |A|/c."""
        return abs(self.A) / self.c

    @property
    def r_e(self) -> float:
        """Radio de la ergosfera, r_e = sqrt(A^2 + B^2)/c."""
        return np.hypot(self.A, self.B) / self.c

    # ------------------------------------------------------------ cinematica

    def v_r0(self, r):
        """Velocidad radial del flujo NO deformado: v_r = -A/r, [L T^-1]."""
        return -self.A / np.asarray(r, dtype=float)

    def v_phi(self, r):
        """Velocidad azimutal: v_phi = B/r, [L T^-1].

        No es afectada por la deformacion radial h(r).
        """
        return self.B / np.asarray(r, dtype=float)

    # ------------------------------------------------------- horizonte: obs.

    @property
    def Omega_H(self) -> float:
        """Velocidad angular del horizonte, [T^-1].

        Omega(r) = -g_{t phi}/g_{phi phi} = B/r^2, evaluada en r_h:

            Omega_H = B/r_h^2 = B c^2 / A^2

        Independiente del gauge (la transformacion PG -> BL solo mezcla
        dt, dphi con dr, y no altera g_{t phi} ni g_{phi phi}).

        Invariante bajo la familia de deformaciones con h(r_h) = 0, porque
        el horizonte permanece en r_h y v_phi no se deforma.
        """
        return self.B * self.c**2 / self.A**2

    @property
    def surface_gravity(self) -> float:
        """kappa_h = |dv_r/dr|_{r_h} = c^2/A, dimensiones [T^-1]."""
        return self.c**2 / self.A

    @property
    def surface_gravity_visser(self) -> float:
        """kappa_V = (1/2)|d(c^2 - v_r^2)/dr|_{r_h} = c^3/A, [L T^-2].

        Convencion de Visser (dimensiones de aceleracion). Difiere de
        `surface_gravity` en un factor c.
        """
        return self.c**3 / self.A

    def superradiant_band(self, m: int) -> tuple[float, float]:
        """Banda superradiante (0, m*Omega_H) para el numero azimutal m.

        El umbral es identico para toda la familia de perfiles deformados,
        dado que Omega_H es invariante. La deformacion h(r) modula la
        AMPLITUD de |R(omega)|^2 dentro de la banda, no su extension.
        """
        if m <= 0:
            return (0.0, 0.0)
        return (0.0, m * self.Omega_H)
