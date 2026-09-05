"""
Integracion radial en dominio de frecuencia.

Se resuelve  d^2H/dr_*^2 + V_h(r) H = 0  como sistema de primer orden en r_*,
integrando simultaneamente r(r_*):

    dr/dr_*  = g(r)
    dH/dr_*  = P
    dP/dr_*  = -V_h(r) H

Integrar en r_* (y no en r) evita la singularidad de coordenadas: el horizonte
se alcanza solo cuando r_* -> -infinito, de modo que r -> r_h de forma
exponencial y suave.

Asintoticas
-----------
Cerca del horizonte  g -> 0, la barrera se anula y

    V_h -> (omega - m Omega_H)^2 / c^2 ,   k_H = (omega - m Omega_H)/c

de modo que la condicion puramente entrante es  H ~ exp(-i k_H r_*).

En el infinito  g -> 1, la barrera decae como 1/r^2 y

    V_h -> omega^2/c^2 ,   k_inf = omega/c
    H ~ A_in exp(-i k_inf r_*) + A_out exp(+i k_inf r_*)

El coeficiente de reflexion es  Rc = A_out/A_in.

Conservacion de flujo
---------------------
Para omega real el potencial es real y el wronskiano se conserva, lo que da

    |A_in|^2 - |A_out|^2 = k_H / k_inf

Cuando k_H < 0 (esto es, omega < m Omega_H) se sigue |Rc|^2 > 1: superradiancia.
El residuo de esta identidad es el diagnostico numerico primario.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..mathematics.potential import V_effective, g_metric
from ..physics.background import DrainingBathtub
from ..physics.deformation import Deformation


@dataclass
class ScatteringResult:
    """Resultado de un problema de dispersion a frecuencia fija."""

    omega: float
    m: int
    A_in: complex
    A_out: complex
    r_max: float

    @property
    def reflection(self) -> complex:
        """Amplitud de reflexion Rc = A_out / A_in."""
        return self.A_out / self.A_in

    @property
    def reflectivity(self) -> float:
        """|Rc|^2. Mayor que 1 dentro de la banda superradiante."""
        return float(abs(self.reflection) ** 2)


def integrate_radial(
    omega: float,
    m: int,
    bg: DrainingBathtub,
    defo: Deformation,
    delta: float = 1e-8,
    r_max: float = 400.0,
    rtol: float = 1e-11,
    atol: float = 1e-13,
):
    """Integra la ecuacion radial desde el horizonte hasta r_max.

    Parameters
    ----------
    delta : float
        Separacion relativa inicial respecto al horizonte, r0 = r_h (1 + delta).
    r_max : float
        Radio de extraccion. Debe cumplir omega * r_max / c >> 1 y r_max >> r_h.

    Returns
    -------
    ScatteringResult
    """
    r_h = bg.r_h
    r0 = r_h * (1.0 + delta)
    k_H = (omega - m * bg.Omega_H) / bg.c

    # Condicion puramente entrante en el horizonte, con r_*(r0) := 0.
    y0 = np.array([r0, 1.0 + 0.0j, -1j * k_H], dtype=complex)

    def rhs(_rs, y):
        r = y[0].real
        g = float(g_metric(r, bg, defo))
        V = complex(V_effective(r, omega, m, bg, defo))
        return np.array([g, y[2], -V * y[1]], dtype=complex)

    def reached(rs, y):
        return y[0].real - r_max

    reached.terminal = True
    reached.direction = 1.0

    # r_* crece aproximadamente como r lejos del horizonte; el tramo cercano
    # anade |r_h ln(delta)|. Se sobredimensiona y se corta por evento.
    rs_span = (0.0, r_max + 2.0 * r_h * abs(np.log(delta)) + 50.0)

    sol = solve_ivp(
        rhs, rs_span, y0, method="DOP853", rtol=rtol, atol=atol, events=reached,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"integracion fallida: {sol.message}")
    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"no se alcanzo r_max={r_max}; r final = {sol.y[0, -1].real:.3f}"
        )

    rs_f = float(sol.t_events[0][0])
    r_f, H, P = sol.y_events[0][0]
    k = omega / bg.c

    # H = A_in e^{-i k r_*} + A_out e^{+i k r_*}
    A_out = 0.5 * (H + P / (1j * k)) * np.exp(-1j * k * rs_f)
    A_in = 0.5 * (H - P / (1j * k)) * np.exp(+1j * k * rs_f)

    return ScatteringResult(
        omega=omega, m=m, A_in=A_in, A_out=A_out, r_max=float(r_f.real)
    )


def flux_residual(res: ScatteringResult, bg: DrainingBathtub) -> float:
    """Residuo relativo de |A_in|^2 - |A_out|^2 = k_H/k_inf.

    Es el diagnostico numerico primario del integrador. Un valor por encima de
    ~1e-8 indica r_max insuficiente, delta demasiado grande o tolerancias laxas.
    """
    k_H = (res.omega - res.m * bg.Omega_H) / bg.c
    k = res.omega / bg.c
    lhs = abs(res.A_in) ** 2 - abs(res.A_out) ** 2
    rhs = k_H / k
    return float(abs(lhs - rhs) / max(abs(rhs), 1e-300))


def reflectivity_curve(
    omegas, m: int, bg: DrainingBathtub, defo: Deformation, **kw
):
    """|Rc(omega)|^2 y residuo de flujo sobre una malla de frecuencias."""
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    refl = np.empty_like(omegas)
    resid = np.empty_like(omegas)
    for i, w in enumerate(omegas):
        r = integrate_radial(w, m, bg, defo, **kw)
        refl[i] = r.reflectivity
        resid[i] = flux_residual(r, bg)
    return refl, resid


def radial_profile(
    omega: float,
    m: int,
    bg: DrainingBathtub,
    defo: Deformation,
    n_out: int = 900,
    delta: float = 1e-8,
    r_max: float = 60.0,
    rtol: float = 1e-11,
    atol: float = 1e-13,
):
    """Devuelve (r, H(r)) muestreado, para reconstruir el campo en el plano.

    El campo fisico es  Psi(r, phi) = H(r)/sqrt(r) * exp(i m phi).
    """
    r_h = bg.r_h
    r0 = r_h * (1.0 + delta)
    k_H = (omega - m * bg.Omega_H) / bg.c
    y0 = np.array([r0, 1.0 + 0.0j, -1j * k_H], dtype=complex)

    def rhs(_rs, y):
        r = y[0].real
        return np.array(
            [
                float(g_metric(r, bg, defo)),
                y[2],
                -complex(V_effective(r, omega, m, bg, defo)) * y[1],
            ],
            dtype=complex,
        )

    def reached(rs, y):
        return y[0].real - r_max

    reached.terminal = True
    reached.direction = 1.0

    rs_end = r_max + 2.0 * r_h * abs(np.log(delta)) + 50.0
    rs_eval = np.linspace(0.0, rs_end, n_out)
    sol = solve_ivp(
        rhs, (0.0, rs_end), y0, method="DOP853", rtol=rtol, atol=atol,
        t_eval=rs_eval, events=reached,
    )
    mask = sol.y[0].real <= r_max
    return sol.y[0].real[mask], sol.y[1][mask]
