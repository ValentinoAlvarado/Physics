"""
Superficie de verosimilitud chi^2(eps, sigma) — kernel CUDA.

POR QUE ESTA FIGURA
===================
La elipse de Fisher es la aproximacion CUADRATICA de chi^2 en torno al punto
fiducial. En los resultados obtenidos la elipse se extiende a sigma < 0, que
es no fisico (sigma es una anchura): senal de que a ese nivel de ruido la
linealizacion ya no vale.

La superficie chi^2 real muestra:
  - el valle de degeneracion VERDADERO, que puede ser curvo (la elipse solo
    puede ser recta);
  - si hay mas de un minimo (multimodalidad), que Fisher no puede ver;
  - hasta donde la aproximacion cuadratica es fiable.

Definicion (mismo observable y mismo ruido que fisher.py):

    chi^2(theta) = sum_k [ |R_k(theta)|^2 - |R_k(theta_0)|^2 ]^2 / sigma_n^2

con datos SIN ruido generados en theta_0 (chi^2 de Fisher-forecast: el minimo
vale exactamente 0 en theta_0). Eso aisla la geometria del problema inverso
del ruido de una realizacion concreta.

POR QUE UN KERNEL NUEVO
=======================
El kernel de cuda_radial.py batchea sobre MODOS m con omega fijo. Aqui hace
falta lo contrario: m fijo (=1) y batcheo sobre la malla (eps, sigma) x omega.
Son ~10^4 integraciones radiales; con scipy serian decenas de minutos. Este
kernel toma omega, eps y sigma como arrays por hilo y resuelve todas las
combinaciones en un lanzamiento.

La FISICA es identica a la de src/numerics/radial.py: mismas ecuaciones,
misma condicion entrante en el horizonte, misma extraccion de A_in/A_out.
`validate_against_scipy` lo comprueba punto por punto.

Requisito:  pip install cupy-cuda12x
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    cp = None
    HAVE_CUPY = False


_SRC = r'''
extern "C" {

__device__ __forceinline__ double g_metric(
    const double r, const double A, const double c,
    const double eps, const double sig, const double rh)
{
    double h = 0.0;
    if (eps != 0.0) {
        const double d = r - rh;
        const double dn = d / rh;
        h = eps * dn * dn * exp(-sig * d * d);
    }
    const double u = 1.0 + h;
    return 1.0 - (A * A / (c * c * r * r)) * u * u;
}

__device__ __forceinline__ double V_eff(
    const double r, const double m, const double omega,
    const double A, const double B, const double c,
    const double eps, const double sig, const double rh)
{
    double h = 0.0, dh = 0.0;
    if (eps != 0.0) {
        const double d = r - rh;
        const double E = exp(-sig * d * d);
        const double dn = d / rh;
        h  = eps * dn * dn * E;
        dh = eps / (rh * rh) * (2.0 * d - 2.0 * sig * d * d * d) * E;
    }
    const double k = A * A / (c * c);
    const double u = 1.0 + h;
    const double r2 = r * r;
    const double g  = 1.0 - (k / r2) * u * u;
    const double dg = 2.0 * k * u * u / (r2 * r) - 2.0 * k * u * dh / r2;
    const double dr_ = (omega - m * B / r2);
    return dr_ * dr_ / (c * c)
         - g * (m * m / r2 + dg / (2.0 * r) - g / (4.0 * r2));
}

__device__ __forceinline__ void deriv(
    const double r, const double Hr, const double Hi,
    const double Pr, const double Pi,
    const double m, const double omega,
    const double A, const double B, const double c,
    const double eps, const double sig, const double rh, double* o)
{
    const double V = V_eff(r, m, omega, A, B, c, eps, sig, rh);
    o[0] = g_metric(r, A, c, eps, sig, rh);
    o[1] = Pr;  o[2] = Pi;
    o[3] = -V * Hr;  o[4] = -V * Hi;
}

// Un hilo por combinacion (omega_i, eps_i, sigma_i). Devuelve |R|^2.
__global__ void reflect_kernel(
    const double* __restrict__ omega_a,
    const double* __restrict__ eps_a,
    const double* __restrict__ sig_a,
    const double m, const double A, const double B, const double c,
    const double r0, const double h_step, const int n_steps,
    const double r_extract,
    double* __restrict__ out_R2,
    double* __restrict__ out_flux,   // residuo de conservacion (diagnostico)
    const int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double omega = omega_a[i];
    const double eps   = eps_a[i];
    const double sig   = sig_a[i];
    const double rh    = fabs(A) / c;
    const double OmH   = B * c * c / (A * A);
    const double kH    = (omega - m * OmH) / c;

    double r = r0, Hr = 1.0, Hi = 0.0, Pr = 0.0, Pi = -kH, rs = 0.0;
    double k1[5], k2[5], k3[5], k4[5];
    bool got = false;
    double R2 = 0.0, flux = 0.0;

    for (int n = 0; n < n_steps; ++n) {
        deriv(r,Hr,Hi,Pr,Pi, m,omega,A,B,c,eps,sig,rh, k1);
        deriv(r+0.5*h_step*k1[0], Hr+0.5*h_step*k1[1], Hi+0.5*h_step*k1[2],
              Pr+0.5*h_step*k1[3], Pi+0.5*h_step*k1[4],
              m,omega,A,B,c,eps,sig,rh, k2);
        deriv(r+0.5*h_step*k2[0], Hr+0.5*h_step*k2[1], Hi+0.5*h_step*k2[2],
              Pr+0.5*h_step*k2[3], Pi+0.5*h_step*k2[4],
              m,omega,A,B,c,eps,sig,rh, k3);
        deriv(r+h_step*k3[0], Hr+h_step*k3[1], Hi+h_step*k3[2],
              Pr+h_step*k3[3], Pi+h_step*k3[4],
              m,omega,A,B,c,eps,sig,rh, k4);
        const double f = h_step / 6.0;
        r  += f*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);
        Hr += f*(k1[1]+2*k2[1]+2*k3[1]+k4[1]);
        Hi += f*(k1[2]+2*k2[2]+2*k3[2]+k4[2]);
        Pr += f*(k1[3]+2*k2[3]+2*k3[3]+k4[3]);
        Pi += f*(k1[4]+2*k2[4]+2*k3[4]+k4[4]);
        rs += h_step;

        if (!got && r >= r_extract) {
            const double kk = omega / c;
            // P/(i k) = (Pi - i Pr)/k
            const double qr = Pi / kk, qi = -Pr / kk;
            const double sr = 0.5*(Hr+qr), si = 0.5*(Hi+qi);   // A_out * e^{ikrs}
            const double tr = 0.5*(Hr-qr), ti = 0.5*(Hi-qi);   // A_in * e^{-ikrs}
            const double ao2 = sr*sr + si*si;
            const double ai2 = tr*tr + ti*ti;
            R2 = ao2 / ai2;
            // |A_in|^2 - |A_out|^2 = k_H/k   (residuo relativo)
            const double lhs = ai2 - ao2, rhsv = kH / kk;
            flux = fabs(lhs - rhsv) / fmax(fabs(rhsv), 1e-300);
            got = true;
            break;
        }
    }
    
    const double nan_val = __longlong_as_double(0x7ff8000000000000ULL);
    out_R2[i] = got ? R2 : nan_val;
    out_flux[i] = got ? flux : nan_val;
}

}  // extern "C"
'''

_MOD = None
_KERN = None


def _kernel():
    global _MOD, _KERN
    if _KERN is None:
        if not HAVE_CUPY:
            raise RuntimeError("CuPy no disponible: pip install cupy-cuda12x")
        _MOD = cp.RawModule(code=_SRC)
        _KERN = _MOD.get_function("reflect_kernel")
    return _KERN


def reflectivity_batch(bg, omegas, epss, sigs, m: int = 1,
                       delta: float = 1e-8, r_extract: float = 300.0,
                       n_steps: int = 160000, block: int = 128):
    """|R|^2 para N combinaciones (omega, eps, sigma) en un solo lanzamiento.

    Los tres arrays deben tener la misma longitud N.
    Devuelve (R2 (N,), flux_residual (N,)).
    """
    kern = _kernel()
    omegas = np.ascontiguousarray(omegas, dtype=np.float64)
    epss = np.ascontiguousarray(epss, dtype=np.float64)
    sigs = np.ascontiguousarray(sigs, dtype=np.float64)
    N = omegas.size
    assert epss.size == N and sigs.size == N

    r_h = abs(bg.A) / bg.c
    r0 = r_h * (1.0 + delta)
    rs_end = r_extract + 2.0 * r_h * abs(np.log(delta)) + 50.0
    h_step = rs_end / n_steps

    d_w = cp.asarray(omegas); d_e = cp.asarray(epss); d_s = cp.asarray(sigs)
    R2 = cp.empty(N, cp.float64)
    fl = cp.empty(N, cp.float64)

    kern(((N + block - 1) // block,), (block,), (
        d_w, d_e, d_s,
        np.float64(m), np.float64(bg.A), np.float64(bg.B), np.float64(bg.c),
        np.float64(r0), np.float64(h_step), np.int32(n_steps),
        np.float64(r_extract), R2, fl, np.int32(N)))
    return cp.asnumpy(R2), cp.asnumpy(fl)


@dataclass
class Chi2Surface:
    eps_grid: np.ndarray          # (n_e,)
    sig_grid: np.ndarray          # (n_s,)
    chi2: np.ndarray              # (n_s, n_e)
    omegas: np.ndarray
    theta0: np.ndarray
    sigma_n: float
    flux_max: float


def chi2_surface(bg, theta0=(0.30, 2.0), m: int = 1,
                 eps_range=(-0.1, 0.7), sig_range=(0.4, 4.0),
                 n_eps: int = 41, n_sig: int = 41, n_omega: int = 24,
                 band_lo: float = 0.06, band_hi: float = 0.96,
                 sigma_n: float = 1e-2, r_extract: float = 300.0,
                 n_steps: int = 160000) -> Chi2Surface:
    """chi^2 sobre una malla de (eps, sigma), con datos sinteticos en theta0."""
    w_max = m * bg.Omega_H
    omegas = np.linspace(band_lo * w_max, band_hi * w_max, n_omega)
    e_g = np.linspace(*eps_range, n_eps)
    s_g = np.linspace(*sig_range, n_sig)

    # datos de referencia (sin ruido) en theta0
    d0, f0 = reflectivity_batch(
        bg, omegas, np.full(n_omega, theta0[0]), np.full(n_omega, theta0[1]),
        m=m, r_extract=r_extract, n_steps=n_steps)

    # malla completa: (n_sig, n_eps, n_omega) aplanada
    S, E, W = np.meshgrid(s_g, e_g, omegas, indexing="ij")
    R2, fl = reflectivity_batch(bg, W.ravel(), E.ravel(), S.ravel(),
                                m=m, r_extract=r_extract, n_steps=n_steps)
    R2 = R2.reshape(n_sig, n_eps, n_omega)

    resid = R2 - d0[None, None, :]
    chi2 = np.nansum(resid ** 2, axis=2) / sigma_n ** 2

    return Chi2Surface(eps_grid=e_g, sig_grid=s_g, chi2=chi2, omegas=omegas,
                       theta0=np.asarray(theta0, float), sigma_n=sigma_n,
                       flux_max=float(np.nanmax(np.concatenate([f0, fl]))))


def validate_against_scipy(bg, m=1, eps=0.30, sig=2.0, n_omega=8,
                           r_extract=300.0, tol=1e-3):
    """Compara |R|^2 del kernel contra reflectivity_curve (scipy)."""
    import time
    from ..numerics.radial import reflectivity_curve
    from ..physics.deformation import GaussianDeformation

    w_max = m * bg.Omega_H
    omegas = np.linspace(0.1 * w_max, 0.9 * w_max, n_omega)

    t0 = time.perf_counter()
    R2, fl = reflectivity_batch(bg, omegas, np.full(n_omega, eps),
                                np.full(n_omega, sig), m=m,
                                r_extract=r_extract)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - t0

    defo = GaussianDeformation(r_h=bg.r_h, eps=eps, sigma=sig)
    t0 = time.perf_counter()
    ref, _ = reflectivity_curve(omegas, m, bg, defo, r_max=r_extract)
    t_cpu = time.perf_counter() - t0

    err = np.abs(R2 - np.asarray(ref)) / np.abs(ref)
    print(f"CUDA  : {t_gpu*1000:8.1f} ms  ({n_omega} frecuencias)")
    print(f"scipy : {t_cpu*1000:8.1f} ms")
    print(f"speedup: {t_cpu/max(t_gpu,1e-9):.1f}x")
    print(f"residuo de flujo max (kernel): {np.nanmax(fl):.2e}")
    print(f"error relativo max en |R|^2: {err.max():.3e}",
          "OK" if err.max() < tol else "FALLA")
    return err.max() < tol
