"""
Busqueda de modos cuasinormales (QNM) en el plano complejo de omega.

Por que disparo simple (horizonte -> infinito) falla
------------------------------------------------------
Para omega complejo, la solucion "fisica" en infinito decae, pero integrar
desde el horizonte mezcla inevitablemente una componente exponencialmente
CRECIENTE (redondeo numerico la excita, por pequena que sea la mezcla inicial).
Al llegar a r_max esa componente domina por completo y el A_in extraido no
tiene relacion con la condicion de contorno real. Esta es la manifestacion
numerica de la inestabilidad pseudoespectral de los QNM (Nollert 1996;
Jaramillo, Macedo & Al Sheikh 2021): el mapa directo esta mal condicionado.

Metodo estable: disparo desde ambos extremos con acople en un punto medio
--------------------------------------------------------------------------
1. Se integra HACIA AFUERA desde el horizonte (condicion entrante) hasta un
   radio de acople r_match. El trayecto es corto y la solucion aun no fue
   contaminada por el modo creciente.
2. Se integra HACIA ADENTRO desde infinito (condicion saliente pura, la unica
   que decae en esa direccion) hasta r_match. Simetricamente tampoco se
   contamina, por la misma razon.
3. La condicion de QNM es que ambas soluciones sean proporcionales en
   r_match, es decir que su Wronskiano se anule:

       W(omega) = H_out(r_match) H_in'(r_match) - H_out'(r_match) H_in(r_match) = 0

Se buscan raices complejas de W(omega) por Newton con diferencias finitas
complejas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..mathematics.potential import V_effective, g_metric
from ..physics.background import DrainingBathtub
from ..physics.deformation import Deformation


def _shoot_from_horizon(omega, m, bg, defo, r_match, delta=1e-8,
                        rtol=1e-11, atol=1e-13):
    """H, dH/dr en r_match, integrando desde el horizonte (condicion entrante)."""
    r_h = bg.r_h
    r0 = r_h * (1.0 + delta)
    k_H = (omega - m * bg.Omega_H) / bg.c
    y0 = np.array([r0, 1.0 + 0.0j, -1j * k_H], dtype=complex)  # r, H, P=dH/dr_*

    def rhs(_rs, y):
        r = y[0].real
        g = float(g_metric(r, bg, defo))
        V = complex(V_effective(r, omega, m, bg, defo))
        return np.array([g, y[2], -V * y[1]], dtype=complex)

    def reached(rs, y):
        return y[0].real - r_match

    reached.terminal = True
    reached.direction = 1.0
    rs_span = (0.0, 2.0 * r_h * abs(np.log(delta)) + 2.0 * r_match + 50.0)

    sol = solve_ivp(rhs, rs_span, y0, method="DOP853", rtol=rtol, atol=atol,
                    events=reached)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("disparo desde el horizonte no alcanzo r_match")
    r_f, H, P = sol.y_events[0][0]
    g_f = float(g_metric(r_match, bg, defo))
    dHdr = P / g_f  # dH/dr = dH/dr_* * dr_*/dr = P / g
    return H, dHdr


def _asymptotic_series(omega, m, bg, order4=True):
    """Coeficientes b1..b4 de H_out = e^{ikr}(1+b1/r+b2/r^2+b3/r^3+b4/r^4).

    Derivados (sympy) de la expansion EXACTA de V_effective_undeformed en 1/r:

        V(r) = k^2 - L/r^2 - c4/r^4 + O(1/r^5),   k = omega/c

        L  = 2 m B omega / c^2 + m^2 - 1/4
        c4 = (A^2 (m^2 - 3/2) + B^2 m^2) / c^2

    Notablemente c3 = 0 (no hay cola tipo Coulomb ~1/r; el sistema no la
    tiene). La deformacion h(r) no aparece porque se anula en r->infinito
    para las familias localizadas de este proyecto, asi que la asintotica
    depende solo del fondo (A, B, c), no de h.

    Sustituir el plano-onda desnudo e^{ikr} por H_out corrige el error de
    fase O(1/r^2) que antes hacia que el QNM encontrado dependiera de r_inf.
    """
    c = bg.c
    A, B = bg.A, bg.B
    k = omega / c
    L = 2.0 * m * B * omega / c**2 + m**2 - 0.25
    c4 = (A**2 * (m**2 - 1.5) + B**2 * m**2) / c**2

    def coeffs(sign):
        ik = sign * 1j * k
        b1 = sign * 1j * L / (2.0 * k)
        b2 = (-L**2 + 2.0 * L) / (8.0 * k**2)
        b3 = (sign * (-1j * L**3 + 8j * L**2 - 12j * L)
              + sign * 8j * c4 * k**2) / (48.0 * k**3)
        if not order4:
            return b1, b2, b3, 0.0
        b4 = (L**4 - 20 * L**3 + 108 * L**2 - 144 * L
              - 32 * L * c4 * k**2 + 96 * c4 * k**2) / (384.0 * k**4)
        return b1, b2, b3, b4

    return coeffs(+1), coeffs(-1)  # (saliente, entrante)


def _series_value_deriv(r, k, sign, b):
    """H(r) y dH/dr para el ansatz e^{i*sign*k*r}(1+b1/r+...+b4/r^4)."""
    b1, b2, b3, b4 = b
    f = 1.0 + b1 / r + b2 / r**2 + b3 / r**3 + b4 / r**4
    df = -b1 / r**2 - 2 * b2 / r**3 - 3 * b3 / r**4 - 4 * b4 / r**5
    phase = np.exp(1j * sign * k * r)
    H = phase * f
    dH = phase * (1j * sign * k * f + df)
    return H, dH


def _shoot_from_infinity(omega, m, bg, defo, r_match, r_inf,
                         rtol=1e-11, atol=1e-13):
    """H, dH/dr en r_match, integrando desde infinito.

    La condicion de partida en r_inf usa la serie asintotica saliente
    (`_asymptotic_series`), no el plano-onda desnudo: esto es lo que hace el
    resultado independiente de r_inf (antes no lo era).
    """
    k = omega / bg.c
    (b_out, _b_in) = _asymptotic_series(omega, m, bg)
    r0 = r_inf
    H0, dH0 = _series_value_deriv(r0, k, +1, b_out)
    # dH/dr_* = (dr/dr_*) * dH/dr = g(r) * dH/dr
    P0 = float(g_metric(r0, bg, defo)) * dH0

    def rhs_rs(_s, y):
        # Parametro s crece mientras r_* DECRECE (vamos de r_inf hacia
        # r_match). d(.)/ds = -d(.)/dr_*, de ahi los signos invertidos
        # respecto al disparo hacia afuera (_shoot_from_horizon).
        r = y[0].real
        g = float(g_metric(r, bg, defo))
        V = complex(V_effective(r, omega, m, bg, defo))
        return np.array([-g, -y[2], V * y[1]], dtype=complex)

    y0 = np.array([r0, H0, P0], dtype=complex)

    def reached(s, y):
        return r_match - y[0].real

    reached.terminal = True
    reached.direction = 1.0
    rs_span = (0.0, 2.0 * (r_inf - r_match) + 50.0)

    sol = solve_ivp(rhs_rs, rs_span, y0, method="DOP853", rtol=rtol, atol=atol,
                    events=reached)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("disparo desde infinito no alcanzo r_match")
    r_f, H, P = sol.y_events[0][0]
    g_f = float(g_metric(r_match, bg, defo))
    dHdr = P / g_f
    return H, dHdr


def wronskian(omega, m, bg, defo, r_match=None, r_inf=200.0, **kw):
    """W(omega) = H1 H2' - H2 H1'. Se anula en un QNM, pero para Im(omega)<0
    la solucion desde infinito crece exponencialmente con r_inf y W queda mal
    condicionado (verificado: el resultado depende de r_inf si se usa esto
    directamente). Se conserva por referencia; usar `log_match` para resolver.
    """
    if r_match is None:
        r_match = 2.5 * bg.r_h
    H1, dH1 = _shoot_from_horizon(omega, m, bg, defo, r_match, **kw)
    H2, dH2 = _shoot_from_infinity(omega, m, bg, defo, r_match, r_inf, **kw)
    return H1 * dH2 - H2 * dH1


def log_match(omega, m, bg, defo, r_match=None, r_inf=200.0, **kw):
    """M(omega) = H1'/H1 - H2'/H2 en r_match: derivada logaritmica.

    Equivale a W(omega)=0 (mismas raices), pero es invariante de escala: cada
    lado se normaliza por su propia solucion, de modo que el crecimiento
    exponencial del disparo desde infinito no contamina la condicion. Esto es
    lo que corrige la inestabilidad frente a r_inf observada con `wronskian`.
    """
    if r_match is None:
        r_match = 2.5 * bg.r_h
    H1, dH1 = _shoot_from_horizon(omega, m, bg, defo, r_match, **kw)
    H2, dH2 = _shoot_from_infinity(omega, m, bg, defo, r_match, r_inf, **kw)
    return dH1 / H1 - dH2 / H2


def find_qnm(omega0, m, bg: DrainingBathtub, defo: Deformation,
            tol=1e-10, max_iter=60, h_step=1e-6, **kw) -> complex:
    """Newton complejo sobre la derivada logaritmica de acople."""
    w = complex(omega0)
    f0 = log_match(w, m, bg, defo, **kw)
    for _ in range(max_iter):
        if abs(f0) < tol:
            return w
        fp = log_match(w + h_step, m, bg, defo, **kw)
        deriv = (fp - f0) / h_step
        if abs(deriv) < 1e-30:
            raise RuntimeError("derivada nula en Newton (log-match)")
        step = f0 / deriv
        if abs(step) > 0.4 * max(abs(w), 1e-3):
            step *= 0.4 * max(abs(w), 1e-3) / abs(step)
        w = w - step
        f0 = log_match(w, m, bg, defo, **kw)
    raise RuntimeError(f"Newton no convergio: |M|={abs(f0):.2e}")


@dataclass
class QNMTrace:
    theta: np.ndarray
    omega: np.ndarray

    @property
    def omega_r(self):
        return self.omega.real

    @property
    def omega_i(self):
        return self.omega.imag


def trace_qnm(omega_seed, thetas, m, bg, make_defo, **kw) -> QNMTrace:
    """Continuacion parametrica: sigue la raiz al variar `thetas`."""
    out = np.empty(len(thetas), dtype=complex)
    w = complex(omega_seed)
    for i, th in enumerate(thetas):
        defo = make_defo(th)
        w = find_qnm(w, m, bg, defo, **kw)
        out[i] = w
    return QNMTrace(theta=np.asarray(thetas, dtype=float), omega=out)
