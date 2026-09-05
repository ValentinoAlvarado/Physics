"""
Analisis cuantitativo del campo complejo Psi. Funciones puras + resultados
como dataclasses; sin estado ni efectos secundarios sobre el render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .scattering import ScatteringField


# ============================================================ simetria B=0
@dataclass
class SymmetryResult:
    B: float
    asymmetry: float          # ||Psi(phi) - Psi(-phi)|| / ||Psi||
    passed: bool
    tol: float


def symmetry_test(sfield: ScatteringField, R: np.ndarray, PHI: np.ndarray,
                  tol: float = 1e-10) -> SymmetryResult:
    """Mide la asimetria bajo phi -> -phi.

    Con B=0 el potencial depende de m solo por m^2 y k_H = omega/c no depende
    de m, luego H_{-m} = H_m; como ademas c_m = (-1)^m = c_{-m}, el campo debe
    ser simetrico a precision de maquina. Una asimetria apreciable con B=0
    indica error de normalizacion o de indexado en las ondas parciales.

    Con B != 0 el termino (omega - m B/r^2)^2 distingue +m de -m y la
    asimetria es FISICA (arrastre de marco): no es un fallo.
    """
    psi = sfield.evaluate(R, PHI)
    psi_ref = sfield.evaluate(R, -PHI)
    den = np.nanmax(np.abs(psi))
    num = np.nanmax(np.abs(psi - psi_ref))
    a = float(num / den) if den > 0 else float("nan")
    is_zero_B = abs(sfield.bg.B) < 1e-15
    return SymmetryResult(B=sfield.bg.B, asymmetry=a,
                          passed=(a < tol) if is_zero_B else (a > 1e-3),
                          tol=tol)


# ======================================================== convergencia en M
@dataclass
class ConvergenceResult:
    m_values: List[int]
    rel_diff: List[float]     # ||Psi_M - Psi_ref|| / ||Psi_ref||
    m_ref: int
    suggested_m: int


def convergence_test(sfield: ScatteringField, R: np.ndarray, PHI: np.ndarray,
                     m_values: Sequence[int] = (5, 10, 15, 20, 30),
                     tol: float = 1e-3) -> ConvergenceResult:
    """Compara Psi truncado a distintos M contra el M mayor disponible."""
    m_values = sorted(int(m) for m in m_values)
    m_ref = m_values[-1]
    old_max = sfield.m_max
    sfield.m_max = max(sfield.m_max, m_ref)

    psi_ref = sfield.evaluate(R, PHI, m_max=m_ref)
    den = np.sqrt(np.nansum(np.abs(psi_ref) ** 2))

    diffs = []
    for M in m_values:
        psi = sfield.evaluate(R, PHI, m_max=M)
        num = np.sqrt(np.nansum(np.abs(psi - psi_ref) ** 2))
        diffs.append(float(num / den) if den > 0 else float("nan"))

    sug = m_ref
    for M, d in zip(m_values, diffs):
        if M != m_ref and d < tol:
            sug = M
            break
    sfield.m_max = old_max
    return ConvergenceResult(list(m_values), diffs, m_ref, sug)


# ============================================================ modulo y fase
def modulus(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi)


def phase(psi: np.ndarray) -> np.ndarray:
    return np.angle(psi)


def unwrap_phase_radial(psi: np.ndarray, axis: int = 0) -> np.ndarray:
    """Fase desenrollada a lo largo de un eje (util para gradientes)."""
    return np.unwrap(np.angle(psi), axis=axis)


# ========================================================= gradiente de fase
@dataclass
class PhaseGradient:
    kx: np.ndarray
    ky: np.ndarray
    magnitude: np.ndarray


def phase_gradient_cartesian(psi: np.ndarray, X: np.ndarray,
                             Y: np.ndarray) -> PhaseGradient:
    """grad(arg Psi) = Im(grad(Psi)/Psi): el vector de onda local k.

    Se calcula asi y NO derivando np.angle, porque angle tiene saltos de 2*pi
    que generan picos espurios. La identidad

        grad(arg Psi) = Im( grad(Psi) / Psi )

    es exacta y continua salvo en los ceros de Psi (donde la fase no esta
    definida, que es justamente donde viven las dislocaciones).

    En la aproximacion eikonal este k es el vector de onda local que aparece
    en omega = v.k + c|k|, la MISMA relacion de dispersion que integra el ray
    tracer. Comparar sus lineas de flujo con las trayectorias RK4 es una
    verificacion cruzada entre la descripcion ondulatoria y la geometrica.
    """
    # espaciados (mallas regulares)
    dy = np.gradient(Y, axis=0)
    dx = np.gradient(X, axis=1)
    dpsi_dy = np.gradient(psi, axis=0) / np.where(dy == 0, np.nan, dy)
    dpsi_dx = np.gradient(psi, axis=1) / np.where(dx == 0, np.nan, dx)

    with np.errstate(invalid="ignore", divide="ignore"):
        kx = np.imag(dpsi_dx / psi)
        ky = np.imag(dpsi_dy / psi)
    mag = np.hypot(kx, ky)
    return PhaseGradient(kx=kx, ky=ky, magnitude=mag)


# ================================================================== ceros
@dataclass
class Zero:
    ix: int
    iy: int
    x: float
    y: float
    charge: int          # carga topologica (giro de fase / 2pi)


def find_phase_defects(psi: np.ndarray, X: np.ndarray, Y: np.ndarray,
                       min_charge: int = 1) -> List[Zero]:
    """Localiza dislocaciones de fase por circulacion discreta.

    Recorre cada celda 2x2 y suma los saltos de fase envueltos al rango
    (-pi, pi] alrededor del circuito. La suma es 2*pi*q con q entero: es la
    carga topologica. Esto NO asume analiticidad de Psi (que no la tiene);
    solo usa continuidad, que es lo unico que hace falta para que la carga
    este cuantizada.
    """
    ph = np.angle(psi)

    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    # circuito: (i,j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j) -> (i,j)
    d1 = wrap(ph[:-1, 1:] - ph[:-1, :-1])
    d2 = wrap(ph[1:, 1:] - ph[:-1, 1:])
    d3 = wrap(ph[1:, :-1] - ph[1:, 1:])
    d4 = wrap(ph[:-1, :-1] - ph[1:, :-1])
    circ = d1 + d2 + d3 + d4
    q = np.rint(circ / (2 * np.pi)).astype(int)

    finite = np.isfinite(psi[:-1, :-1]) & np.isfinite(psi[1:, 1:])
    ys, xs = np.where((np.abs(q) >= min_charge) & finite)
    out = []
    for iy, ix in zip(ys, xs):
        out.append(Zero(ix=int(ix), iy=int(iy),
                        x=float(X[iy, ix]), y=float(Y[iy, ix]),
                        charge=int(q[iy, ix])))
    return out


# ====================================================== salud numerica
@dataclass
class HealthResult:
    n_nan: int
    n_inf: int
    n_total: int
    max_abs: float
    min_abs_nonzero: float
    passed: bool


def health_check(psi: np.ndarray) -> HealthResult:
    """NaN/Inf y rango dinamico. Los NaN de fuera del dominio resuelto son
    esperados y se cuentan aparte por el llamador si hace falta."""
    finite = np.isfinite(psi)
    a = np.abs(psi[finite])
    nz = a[a > 0]
    return HealthResult(
        n_nan=int(np.isnan(psi).sum()),
        n_inf=int(np.isinf(psi).sum()),
        n_total=int(psi.size),
        max_abs=float(a.max()) if a.size else float("nan"),
        min_abs_nonzero=float(nz.min()) if nz.size else float("nan"),
        passed=bool(np.isinf(psi).sum() == 0),
    )
