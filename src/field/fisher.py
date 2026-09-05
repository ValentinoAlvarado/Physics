"""
Analisis de Fisher del canal superradiante — identificabilidad local.

=============================================================================
OBSERVABLE
=============================================================================
El vector de datos es la curva de reflectividad en la banda superradiante,

    d_k = |R(omega_k)|^2 ,   k = 1..N ,   0 < omega_k < m Omega_H

calculada con `reflectivity_curve` (ya validada: residuo de conservacion de
flujo ~1e-10, y |R|^2 > 1 exactamente cuando omega < m Omega_H).

Solo m = 1. Medido previamente: la barrera centrifuga crece como m^2 y para
m >= 3 la amplificacion es indistinguible de cero en toda la banda, de modo
que esos canales no aportan informacion.

=============================================================================
PARAMETROS ADIMENSIONALES  (correccion metodologica importante)
=============================================================================
Los parametros de forma eps ~ 0.3 y sigma ~ 2.0 tienen escalas y dimensiones
distintas. Una matriz de Fisher construida sobre ellos directamente da un
numero de condicion que DEPENDE DE LAS UNIDADES: reescalar sigma cambiaria
cond(F) sin cambiar la fisica.

Por eso se trabaja con parametros FRACCIONARIOS

    u_i = theta_i / theta_i^(0)      (u = 1 en el punto fiducial)

de modo que F_u es adimensional, cond(F_u) es una propiedad del problema, y
los errores salen directamente como fracciones del valor fiducial. Las cotas
en unidades fisicas se recuperan multiplicando por theta^(0).

=============================================================================
MODELO DE RUIDO
=============================================================================
Ruido gaussiano independiente de desviacion sigma_n en cada |R_k|^2 medido:

    F_ij = (1/sigma_n^2) sum_k (d|R_k|^2/du_i)(d|R_k|^2/du_j)

Las derivadas van por diferencias centradas; `derivative_convergence`
comprueba que el paso esta convergido (sin eso la Fisher no es fiable).

=============================================================================
LIMITACIONES DECLARADAS
=============================================================================
1. Identificabilidad LOCAL: la Fisher linealiza en torno al punto fiducial.
   No sustituye a una inversion bayesiana, que exploraria la posterior
   global y podria revelar multimodalidad que la Fisher no ve.
2. Solo el canal superradiante. El canal QNM no esta incluido (no hay un
   QNM confirmado en este trabajo).
3. El modelo de ruido es una suposicion, no una medida experimental: por eso
   los resultados se presentan como BARRIDO en sigma_n, no para un valor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..numerics.radial import reflectivity_curve
from ..physics.background import DrainingBathtub
from ..physics.deformation import GaussianDeformation


# ================================================================ resultado
@dataclass
class FisherResult:
    theta_names: List[str]
    theta0: np.ndarray            # punto fiducial en unidades fisicas
    omegas: np.ndarray
    signal: np.ndarray            # |R|^2 fiducial
    flux_residual: np.ndarray     # diagnostico del integrador
    jac_u: np.ndarray             # (N, 2) d|R|^2/du, u = theta/theta0
    F_unit: np.ndarray            # Fisher adimensional para sigma_n = 1
    sigma_n: float

    @property
    def F(self):
        return self.F_unit / self.sigma_n ** 2

    @property
    def cov_u(self):
        return np.linalg.inv(self.F)

    @property
    def rel_errors(self):
        """Cotas de Cramer-Rao como FRACCION del valor fiducial."""
        return np.sqrt(np.diag(self.cov_u))

    @property
    def abs_errors(self):
        """Cotas en unidades fisicas."""
        return self.rel_errors * self.theta0

    @property
    def eig(self):
        return np.linalg.eigh(self.F)

    @property
    def condition(self):
        ev, _ = self.eig
        return float(ev.max() / max(ev.min(), 1e-300))

    def with_noise(self, sigma_n: float) -> "FisherResult":
        """Mismo jacobiano, otro nivel de ruido (F escala como 1/sigma_n^2)."""
        return FisherResult(
            self.theta_names, self.theta0, self.omegas, self.signal,
            self.flux_residual, self.jac_u, self.F_unit, float(sigma_n))

    def error_ellipse(self, n_sigma: float = 1.0, n_pts: int = 200):
        """Elipse de confianza en el plano (eps, sigma), unidades fisicas."""
        ev, evec = np.linalg.eigh(self.cov_u)
        t = np.linspace(0, 2 * np.pi, n_pts)
        circle = np.stack([np.cos(t), np.sin(t)])
        pts_u = (evec @ (np.sqrt(np.maximum(ev, 0))[:, None] * circle))
        pts = self.theta0[:, None] * (1.0 + n_sigma * pts_u)
        return pts[0], pts[1]


# ================================================================== helpers
def _reflect(bg, omegas, m, eps, sig, **kw):
    defo = GaussianDeformation(r_h=bg.r_h, eps=float(eps), sigma=float(sig))
    refl, resid = reflectivity_curve(omegas, m, bg, defo, **kw)
    return np.asarray(refl), np.asarray(resid)


# =================================================================== fisher
def fisher_superradiance(bg: DrainingBathtub, m: int = 1,
                         theta0: Sequence[float] = (0.30, 2.0),
                         n_omega: int = 36,
                         band_lo: float = 0.06, band_hi: float = 0.96,
                         frac_step: float = 0.08,
                         sigma_n: float = 1e-2,
                         r_max: float = 300.0,
                         verbose: bool = True) -> FisherResult:
    """Matriz de Fisher para theta = (eps, sigma), en parametros fraccionarios.

    frac_step : paso de la diferencia centrada, como FRACCION de theta0.
                Usar el mismo paso fraccionario en ambos parametros es lo que
                hace comparables las dos columnas del jacobiano.
    """
    theta0 = np.asarray(theta0, dtype=float)
    e0, s0 = theta0
    w_max = m * bg.Omega_H
    omegas = np.linspace(band_lo * w_max, band_hi * w_max, n_omega)

    if verbose:
        print(f"banda superradiante m={m}: (0, {w_max:.4f})")
        print(f"fiducial: eps={e0:.3f}  sigma={s0:.3f}   "
              f"paso fraccionario={frac_step}")
        print(f"evaluando {n_omega} frecuencias x 5 configuraciones...")

    sig0, resid0 = _reflect(bg, omegas, m, e0, s0, r_max=r_max)

    he, hs = frac_step * e0, frac_step * s0
    ep, _ = _reflect(bg, omegas, m, e0 + he, s0, r_max=r_max)
    em, _ = _reflect(bg, omegas, m, e0 - he, s0, r_max=r_max)
    sp, _ = _reflect(bg, omegas, m, e0, s0 + hs, r_max=r_max)
    sm, _ = _reflect(bg, omegas, m, e0, s0 - hs, r_max=r_max)

    # derivadas respecto a u = theta/theta0  ->  d/du = theta0 * d/dtheta
    d_u_eps = (ep - em) / (2 * frac_step)
    d_u_sig = (sp - sm) / (2 * frac_step)
    J = np.stack([d_u_eps, d_u_sig], axis=1)

    return FisherResult(
        theta_names=["eps", "sigma"], theta0=theta0, omegas=omegas,
        signal=sig0, flux_residual=resid0, jac_u=J,
        F_unit=J.T @ J, sigma_n=float(sigma_n))


def derivative_convergence(bg, m=1, theta0=(0.30, 2.0),
                           steps=(0.16, 0.08, 0.04), n_omega=18,
                           r_max=300.0) -> List[Tuple[float, float]]:
    """Convergencia del jacobiano al reducir el paso de la diferencia.

    Si el jacobiano cambia apreciablemente entre pasos, la Fisher no es
    fiable y hay que reducir mas el paso (o subir la precision del
    integrador radial).
    """
    out, ref = [], None
    for h in steps:
        r = fisher_superradiance(bg, m=m, theta0=theta0, n_omega=n_omega,
                                 frac_step=h, r_max=r_max, verbose=False)
        if ref is None:
            out.append((h, 0.0))
        else:
            d = np.max(np.abs(r.jac_u - ref)) / (np.abs(ref).max() + 1e-300)
            out.append((h, float(d)))
        ref = r.jac_u
    return out


def noise_scan(res: FisherResult,
               sigmas: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """Errores relativos y condicionamiento en funcion del ruido asumido."""
    rows = []
    for s in sigmas:
        r = res.with_noise(s)
        rows.append((s, r.rel_errors[0], r.rel_errors[1], r.condition))
    return rows


def identifiability_threshold(res: FisherResult, rel_tol: float = 0.30,
                              lo=1e-4, hi=1.0, n_iter=60) -> float:
    """Nivel de ruido sigma_n al que el error relativo en eps alcanza rel_tol.

    Como los errores escalan linealmente con sigma_n, esto es exacto:
        sigma_error(eps) = rel_errors(1) * sigma_n
    """
    base = res.with_noise(1.0).rel_errors[0]
    if base <= 0:
        return float("nan")
    return float(rel_tol / base)


# ============================================================ interpretacion
def interpret(res: FisherResult, rel_tol: float = 0.30) -> str:
    """Genera la lectura fisica de los numeros. Texto listo para el poster."""
    ev, evec = res.eig
    worst = evec[:, 0]
    best = evec[:, 1]
    thr = identifiability_threshold(res, rel_tol)
    amp = res.signal.max() - 1.0
    k_peak = int(np.argmax(np.abs(res.jac_u).sum(axis=1)))
    w_peak = res.omegas[k_peak]
    w_max = res.omegas[-1] / 0.96

    L = []
    L.append("INTERPRETACION")
    L.append("-" * 60)
    L.append(
        f"1. AMPLIFICACION. El fondo amplifica hasta {100*amp:.2f}% dentro de "
        f"la banda. Toda la informacion sobre la forma del perfil esta "
        f"modulada sobre esa amplificacion: sin superradiancia no hay senal.")
    L.append("")
    L.append(
        f"2. DONDE ESTA LA INFORMACION. La sensibilidad |d|R|^2/du| es maxima "
        f"en omega = {w_peak:.4f} ({w_peak/w_max:.2f} del umbral m*Omega_H). "
        f"Medir cerca del umbral o cerca de omega=0 aporta poco: alli las "
        f"derivadas se anulan. Esto es una prescripcion de diseno "
        f"experimental, no una observacion cualitativa.")
    L.append("")
    L.append(
        f"3. DEGENERACION. cond(F) = {res.condition:.1f}. La combinacion peor "
        f"determinada es {worst[0]:+.3f}*(eps/eps0) {worst[1]:+.3f}*"
        f"(sigma/sigma0); la mejor determinada es {best[0]:+.3f}*(eps/eps0) "
        f"{best[1]:+.3f}*(sigma/sigma0).")
    if res.condition > 30:
        L.append(
            f"   cond >> 1: el canal superradiante por si solo NO separa "
            f"amplitud de anchura de la deformacion. Restringe una "
            f"combinacion, no las dos por separado. Romper esa degeneracion "
            f"exige un segundo canal independiente (los QNM).")
    else:
        L.append(
            f"   cond moderado: el canal separa razonablemente ambos "
            f"parametros por si solo.")
    L.append("")
    L.append(
        f"4. UMBRAL DE RUIDO. Con un error relativo tolerable del "
        f"{100*rel_tol:.0f}% en eps, la identificabilidad se pierde cuando "
        f"sigma_n > {thr:.2e} en |R|^2. Contrastar con la amplificacion total "
        f"del fondo ({amp:.4f}): la relacion senal/ruido util es "
        f"amp/sigma_n ~ {amp/thr:.1f} en ese umbral.")
    L.append("")
    L.append(
        f"5. ALCANCE. Identificabilidad LOCAL (linealizada) del canal "
        f"superradiante con m=1, sobre la familia gaussiana (eps, sigma). No "
        f"incluye el canal QNM ni explora la posterior global: una inversion "
        f"bayesiana podria revelar multimodalidad que la Fisher no detecta.")
    return "\n".join(L)


def report(res: FisherResult, rel_tol: float = 0.30):
    ev, evec = res.eig
    print("\n" + "=" * 64)
    print("MATRIZ DE FISHER — canal superradiante (parametros fraccionarios)")
    print("=" * 64)
    print(f"\nfiducial: eps={res.theta0[0]:.3f}  sigma={res.theta0[1]:.3f}")
    print(f"frecuencias: {len(res.omegas)}   "
          f"residuo de flujo max: {res.flux_residual.max():.2e}")
    print(f"amplificacion maxima del fondo: {res.signal.max()-1:.4f}")
    print(f"\nsensibilidades (adimensionales):")
    print(f"  max |d|R|^2/d(eps/eps0)|     = "
          f"{np.abs(res.jac_u[:,0]).max():.4e}")
    print(f"  max |d|R|^2/d(sigma/sigma0)| = "
          f"{np.abs(res.jac_u[:,1]).max():.4e}")
    print(f"\nF (sigma_n={res.sigma_n:g}):")
    print(f"  [{res.F[0,0]:11.4e} {res.F[0,1]:11.4e}]")
    print(f"  [{res.F[1,0]:11.4e} {res.F[1,1]:11.4e}]")
    print(f"\nautovalores: {ev[0]:.4e}  {ev[1]:.4e}")
    print(f"cond(F) = {res.condition:.1f}")
    print(f"\ncotas de Cramer-Rao (sigma_n={res.sigma_n:g}):")
    print(f"  eps   : {res.abs_errors[0]:.4f} absoluto  "
          f"({100*res.rel_errors[0]:.1f}% relativo)")
    print(f"  sigma : {res.abs_errors[1]:.4f} absoluto  "
          f"({100*res.rel_errors[1]:.1f}% relativo)")
    print(f"\nbarrido en ruido:")
    print(f"  {'sigma_n':>9} {'err(eps)':>10} {'err(sigma)':>11} "
          f"{'cond':>8}  identificable")
    for s, ee, es, c in noise_scan(res):
        print(f"  {s:9.1e} {100*ee:9.1f}% {100*es:10.1f}% {c:8.1f}  "
              f"{'SI' if ee < rel_tol else 'NO'}")
    print(f"\numbral: sigma_n = "
          f"{identifiability_threshold(res, rel_tol):.3e}")
    print("=" * 64)
    print()
    print(interpret(res, rel_tol))
