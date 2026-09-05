"""
Kernel CUDA para el solver radial: UN HILO POR MODO, bucle RK4 dentro.

Por que un kernel y no PyTorch vectorizado
-------------------------------------------
Medido: batchear los 41 modos en un bucle RK4 de Python/PyTorch da 36 s,
frente a 1.7 s de scipy (para 9 modos). El bucle a nivel de Python lanza
decenas de operaciones diminutas por paso; con 60000 pasos el coste es todo
overhead de lanzamiento, no calculo. Vectorizar sobre 41 elementos no ayuda.

Con un kernel, cada modo vive en un hilo y los 60000 pasos ocurren en
registros: un solo lanzamiento por resolucion completa.

Simplificacion clave
---------------------
Para omega REAL el potencial V_h(r; omega, m) es REAL. El sistema

    dr/dr_*   = g(r)
    dH/dr_*   = P
    dP/dr_*   = -V_h(r) H

se separa entonces en parte real e imaginaria sin acoplarlas:

    dHr = Pr ,  dHi = Pi ,  dPr = -V*Hr ,  dPi = -V*Hi

de modo que NO hace falta aritmetica compleja dentro del kernel.

Nota: r(r_*) no depende de m, pero cada hilo integra su propia copia. Es un
calculo redundante barato que evita sincronizacion entre hilos.

Requisito:  pip install cupy-cuda12x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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
        const double E = exp(-sig * d * d);
        const double dn = d / rh;
        h = eps * dn * dn * E;
    }
    const double k = A * A / (c * c);
    const double u = 1.0 + h;
    return 1.0 - (k / (r * r)) * u * u;
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

    const double drag = (omega - m * B / r2) * (omega - m * B / r2) / (c * c);
    const double barrier = g * (m * m / r2 + dg / (2.0 * r) - g / (4.0 * r2));
    return drag - barrier;
}

// estado por hilo: (r, Hr, Hi, Pr, Pi)
__device__ __forceinline__ void deriv(
    const double r, const double Hr, const double Hi,
    const double Pr, const double Pi,
    const double m, const double omega,
    const double A, const double B, const double c,
    const double eps, const double sig, const double rh,
    double* out)
{
    const double g = g_metric(r, A, c, eps, sig, rh);
    const double V = V_eff(r, m, omega, A, B, c, eps, sig, rh);
    out[0] = g;         // dr/dr_*
    out[1] = Pr;        // dHr
    out[2] = Pi;        // dHi
    out[3] = -V * Hr;   // dPr
    out[4] = -V * Hi;   // dPi
}

__global__ void radial_kernel(
    const double* __restrict__ m_arr,
    const double omega, const double A, const double B, const double c,
    const double eps, const double sig,
    const double r0, const double h_step, const int n_steps,
    const double r_extract, const int record_every, const int n_rec,
    double* __restrict__ rec_r,      // (n_rec,)   solo lo escribe el hilo 0
    double* __restrict__ rec_Hr,     // (n_rec, n_modes)
    double* __restrict__ rec_Hi,
    double* __restrict__ A_in_r,  double* __restrict__ A_in_i,
    double* __restrict__ A_out_r, double* __restrict__ A_out_i,
    const int n_modes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_modes) return;

    const double m  = m_arr[i];
    const double rh = fabs(A) / c;
    const double OmH = B * c * c / (A * A);
    const double kH = (omega - m * OmH) / c;

    // condicion puramente entrante en el horizonte:  H = 1, P = -i kH
    double r = r0, Hr = 1.0, Hi = 0.0, Pr = 0.0, Pi = -kH;
    double rs = 0.0;
    int j = 0;
    bool got = false;
    double ain_r = 0.0, ain_i = 0.0, aout_r = 0.0, aout_i = 0.0;

    if (j < n_rec) {
        if (i == 0) rec_r[j] = r;
        rec_Hr[j * n_modes + i] = Hr;
        rec_Hi[j * n_modes + i] = Hi;
        j++;
    }

    double k1[5], k2[5], k3[5], k4[5];
    for (int n = 0; n < n_steps; ++n) {
        deriv(r, Hr, Hi, Pr, Pi, m, omega, A,B,c, eps,sig,rh, k1);
        deriv(r + 0.5*h_step*k1[0], Hr + 0.5*h_step*k1[1],
              Hi + 0.5*h_step*k1[2], Pr + 0.5*h_step*k1[3],
              Pi + 0.5*h_step*k1[4], m, omega, A,B,c, eps,sig,rh, k2);
        deriv(r + 0.5*h_step*k2[0], Hr + 0.5*h_step*k2[1],
              Hi + 0.5*h_step*k2[2], Pr + 0.5*h_step*k2[3],
              Pi + 0.5*h_step*k2[4], m, omega, A,B,c, eps,sig,rh, k3);
        deriv(r + h_step*k3[0], Hr + h_step*k3[1], Hi + h_step*k3[2],
              Pr + h_step*k3[3], Pi + h_step*k3[4],
              m, omega, A,B,c, eps,sig,rh, k4);

        const double f = h_step / 6.0;
        r  += f * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]);
        Hr += f * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]);
        Hi += f * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]);
        Pr += f * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]);
        Pi += f * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4]);
        rs += h_step;

        if (((n + 1) % record_every) == 0 && j < n_rec) {
            if (i == 0) rec_r[j] = r;
            rec_Hr[j * n_modes + i] = Hr;
            rec_Hi[j * n_modes + i] = Hi;
            j++;
        }

        if (!got && r >= r_extract) {
            // A_out = 0.5 (H + P/(i k)) e^{-i k rs}
            // A_in  = 0.5 (H - P/(i k)) e^{+i k rs}
            const double kk = omega / c;
            // P/(i k) = -i P / k  ->  (Pr + i Pi) * (-i/k) = (Pi - i Pr)/k
            const double qr =  Pi / kk;
            const double qi = -Pr / kk;
            const double sr = 0.5 * (Hr + qr), si = 0.5 * (Hi + qi);
            const double tr = 0.5 * (Hr - qr), ti = 0.5 * (Hi - qi);
            const double cm = cos(kk * rs), sm = sin(kk * rs);
            // e^{-i k rs} = cm - i sm ;  e^{+i k rs} = cm + i sm
            aout_r = sr * cm + si * sm;
            aout_i = si * cm - sr * sm;
            ain_r  = tr * cm - ti * sm;
            ain_i  = ti * cm + tr * sm;
            got = true;
        }
    }

    if (!got) {
        const double kk = omega / c;
        const double qr =  Pi / kk, qi = -Pr / kk;
        const double sr = 0.5 * (Hr + qr), si = 0.5 * (Hi + qi);
        const double tr = 0.5 * (Hr - qr), ti = 0.5 * (Hi - qi);
        const double cm = cos(kk * rs), sm = sin(kk * rs);
        aout_r = sr * cm + si * sm;  aout_i = si * cm - sr * sm;
        ain_r  = tr * cm - ti * sm;  ain_i  = ti * cm + tr * sm;
    }

    A_in_r[i]  = ain_r;   A_in_i[i]  = ain_i;
    A_out_r[i] = aout_r;  A_out_i[i] = aout_i;
}

}  // extern "C"
'''

_MODULE = None
_KERNEL = None


def _kernel():
    global _MODULE, _KERNEL
    if _KERNEL is None:
        if not HAVE_CUPY:
            raise RuntimeError("CuPy no disponible: pip install cupy-cuda12x")
        _MODULE = cp.RawModule(code=_SRC)
        _KERNEL = _MODULE.get_function("radial_kernel")
    return _KERNEL


@dataclass
class BatchedRadial:
    m: np.ndarray
    r: np.ndarray             # (n_rec,)
    H: np.ndarray             # (n_rec, n_modes) complejo
    A_in: np.ndarray
    A_out: np.ndarray


def solve_modes_cuda(bg, omega: float, m_values: Sequence[int],
                     defo=None, delta: float = 1e-8,
                     r_profile: float = 12.0, r_extract: float = 400.0,
                     n_steps: int = 200000, record_every: int = 60,
                     block: int = 64) -> BatchedRadial:
    """Resuelve todos los modos con un unico lanzamiento de kernel."""
    kern = _kernel()
    eps = float(getattr(defo, "eps", 0.0)) if defo is not None else 0.0
    sig = 2.0
    if defo is not None:
        sig = float(getattr(defo, "sigma", getattr(defo, "sig", 2.0)))

    m_arr = np.asarray(m_values, dtype=np.float64)
    n_modes = m_arr.size
    r_h = abs(bg.A) / bg.c
    r0 = r_h * (1.0 + delta)
    rs_end = r_extract + 2.0 * r_h * abs(np.log(delta)) + 50.0
    h_step = rs_end / n_steps
    n_rec = n_steps // record_every + 2

    d_m = cp.asarray(m_arr)
    rec_r = cp.zeros(n_rec, dtype=cp.float64)
    rec_Hr = cp.zeros((n_rec, n_modes), dtype=cp.float64)
    rec_Hi = cp.zeros((n_rec, n_modes), dtype=cp.float64)
    ain_r = cp.zeros(n_modes, dtype=cp.float64)
    ain_i = cp.zeros(n_modes, dtype=cp.float64)
    aout_r = cp.zeros(n_modes, dtype=cp.float64)
    aout_i = cp.zeros(n_modes, dtype=cp.float64)

    grid = ((n_modes + block - 1) // block,)
    kern(grid, (block,), (
        d_m, np.float64(omega), np.float64(bg.A), np.float64(bg.B),
        np.float64(bg.c), np.float64(eps), np.float64(sig),
        np.float64(r0), np.float64(h_step), np.int32(n_steps),
        np.float64(r_extract), np.int32(record_every), np.int32(n_rec),
        rec_r, rec_Hr, rec_Hi, ain_r, ain_i, aout_r, aout_i,
        np.int32(n_modes),
    ))

    rr = cp.asnumpy(rec_r)
    HH = cp.asnumpy(rec_Hr) + 1j * cp.asnumpy(rec_Hi)
    keep = (rr > 0) & (rr <= r_profile)
    return BatchedRadial(
        m=m_arr.astype(int), r=rr[keep], H=HH[keep],
        A_in=cp.asnumpy(ain_r) + 1j * cp.asnumpy(ain_i),
        A_out=cp.asnumpy(aout_r) + 1j * cp.asnumpy(aout_i))


# ============================================================== validacion
def validate_against_scipy(bg, omega=1.6, m_values=range(-4, 5),
                           n_steps=200000, r_extract=400.0, tol=1e-3):
    """Compara A_in del kernel contra la version scipy existente."""
    from ..numerics.radial import integrate_radial
    from ..physics.deformation import NullDeformation
    import time

    m_values = list(m_values)
    defo = NullDeformation(r_h=bg.r_h)

    t0 = time.perf_counter()
    res = solve_modes_cuda(bg, omega, m_values, n_steps=n_steps,
                           r_extract=r_extract)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = np.array([integrate_radial(omega, m, bg, defo,
                                     r_max=r_extract).A_in for m in m_values])
    t_cpu = time.perf_counter() - t0

    err = np.abs(res.A_in - ref) / np.abs(ref)
    print(f"CUDA  : {t_gpu*1000:8.2f} ms   ({len(m_values)} modos)")
    print(f"scipy : {t_cpu*1000:8.2f} ms")
    print(f"speedup: {t_cpu/t_gpu:.1f}x")
    print(f"error relativo max en A_in: {err.max():.3e}",
          "OK" if err.max() < tol else "FALLA")
    return err.max() < tol
