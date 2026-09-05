"""
Funcion metrica deformada g(r) y potencial efectivo V_h(r; omega, m).

Reduccion de Klein-Gordon a forma tipo Schrodinger
--------------------------------------------------
Con v_r(r) = -(A/r)[1 + h(r)], v_phi = B/r, c constante:

    g(r) = 1 - v_r^2/c^2 = 1 - (A^2/(c^2 r^2)) [1 + h(r)]^2
    dr_*/dr = 1/g(r)
    H(r) = R(r) sqrt(r)

conducen a

    d^2 H/dr_*^2 + V_h(r) H = 0

con

    V_h(r) = (1/c^2) (omega - m B/r^2)^2
             - g(r) [ m^2/r^2 + g'(r)/(2r) - g(r)/(4r^2) ]

Verificado: para h = 0 el corchete se reduce a
(1/(4r^2)) (4m^2 - 1 + 5A^2/(c^2 r^2)), recuperando el potencial estandar del
draining bathtub. Ver tests/test_potential.py.

ADVERTENCIA ESTRUCTURAL
-----------------------
V_h DEPENDE de omega (a traves del termino de arrastre). El problema NO es un
Sturm-Liouville estandar: es un problema de autovalores no lineal en omega.
Consecuencia directa: la formula vanilla de integracion caracteristica
(Gundlach-Price-Pullin) 4 d^2H/dudv = -V H NO es aplicable para B != 0, porque
en dominio temporal aparece un termino de primer orden con coeficiente
imaginario:

    d^2H/dr_*^2 - (1/c^2) d_t^2 H - (2 i m B/(c^2 r^2)) d_t H
      + (1/c^2)(m B/r^2)^2 H - g[...] H = 0

Por eso este proyecto resuelve el problema directo en DOMINIO DE FRECUENCIA.
"""

from __future__ import annotations

import numpy as np

from ..physics.background import DrainingBathtub
from ..physics.deformation import Deformation


def g_metric(r, bg: DrainingBathtub, defo: Deformation):
    """g(r) = 1 - (A^2/(c^2 r^2)) [1 + h(r)]^2. Adimensional.

    Se anula exactamente en el horizonte cuando h(r_h) = 0.
    """
    r = np.asarray(r, dtype=float)
    hh = defo.h(r)
    return 1.0 - (bg.A**2 / (bg.c**2 * r**2)) * (1.0 + hh) ** 2


def dg_metric(r, bg: DrainingBathtub, defo: Deformation):
    """g'(r), dimensiones [L^-1]:

        g' = 2A^2/(c^2 r^3) [1+h]^2 - 2A^2/(c^2 r^2) [1+h] h'
    """
    r = np.asarray(r, dtype=float)
    hh = defo.h(r)
    dh = defo.dh(r)
    k = bg.A**2 / bg.c**2
    return 2.0 * k * (1.0 + hh) ** 2 / r**3 - 2.0 * k * (1.0 + hh) * dh / r**2


def V_effective(r, omega, m: int, bg: DrainingBathtub, defo: Deformation):
    """Potencial efectivo V_h(r; omega, m), dimensiones [L^-2].

    `omega` puede ser complejo (busqueda de QNM) o real (superradiancia).
    """
    r = np.asarray(r, dtype=float)
    g = g_metric(r, bg, defo)
    dg = dg_metric(r, bg, defo)

    drag = (omega - m * bg.B / r**2) ** 2 / bg.c**2
    barrier = g * (m**2 / r**2 + dg / (2.0 * r) - g / (4.0 * r**2))
    return drag - barrier


def V_effective_undeformed(r, omega, m: int, bg: DrainingBathtub):
    """Forma cerrada del potencial NO deformado, usada como referencia
    independiente para validar `V_effective` con h = 0:

        V = (1/c^2)(omega - mB/r^2)^2
            - f (1/(4r^2)) (4m^2 - 1 + 5A^2/(c^2 r^2))
        f = 1 - A^2/(c^2 r^2)
    """
    r = np.asarray(r, dtype=float)
    f = 1.0 - bg.A**2 / (bg.c**2 * r**2)
    drag = (omega - m * bg.B / r**2) ** 2 / bg.c**2
    barrier = f / (4.0 * r**2) * (4 * m**2 - 1 + 5 * bg.A**2 / (bg.c**2 * r**2))
    return drag - barrier
