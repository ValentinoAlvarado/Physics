"""
Cronometra el barrido COMPLETO de chi^2 (GPU vs scipy) sobre la misma malla
que usa chi2_surface_3d.py — no solo el paso de validacion de 8 frecuencias.

    python benchmark_chi2_full.py

Corre desde la raiz del proyecto (junto a check_cuda_field.py).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.field import chi2_cuda
from src.physics.background import DrainingBathtub
from src.physics.deformation import GaussianDeformation
from src.numerics.radial import reflectivity_curve

if not chi2_cuda.HAVE_CUPY:
    raise SystemExit("Falta CuPy: pip install cupy-cuda12x")

import cupy as cp

BG = DrainingBathtub(A=1.0, B=0.7, c=1.0)
M = 1
THETA0 = (0.30, 2.0)
EPS_RANGE = (-0.10, 0.70)
SIG_RANGE = (0.40, 4.00)
N_EPS, N_SIG, N_OMEGA = 41, 41, 24
R_EXTRACT = 300.0

w_max = M * BG.Omega_H
omegas = np.linspace(0.06 * w_max, 0.96 * w_max, N_OMEGA)
e_g = np.linspace(*EPS_RANGE, N_EPS)
s_g = np.linspace(*SIG_RANGE, N_SIG)
S, E, W = np.meshgrid(s_g, e_g, omegas, indexing="ij")
n_total = W.size
print(f"malla completa: {N_SIG}x{N_EPS}x{N_OMEGA} = {n_total} integraciones\n")

# ---------------------------------------------------------------- GPU
print("calentando kernel (compilacion JIT, no se cuenta)...")
chi2_cuda.reflectivity_batch(BG, omegas[:2], np.full(2, THETA0[0]),
                              np.full(2, THETA0[1]), m=M, r_extract=R_EXTRACT)
cp.cuda.Stream.null.synchronize()

t0 = time.perf_counter()
R2_gpu, _ = chi2_cuda.reflectivity_batch(BG, W.ravel(), E.ravel(), S.ravel(),
                                          m=M, r_extract=R_EXTRACT)
cp.cuda.Stream.null.synchronize()
t_gpu = time.perf_counter() - t0
print(f"GPU (kernel, malla completa): {t_gpu:8.3f} s")

# ---------------------------------------------------------------- CPU / scipy
# ADVERTENCIA: esto tarda. 40000+ integraciones en scipy, minutos a horas
# segun tu maquina. Si quieres solo una estimacion, corre con --subset.
N_SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else n_total

idx = np.random.default_rng(0).choice(n_total, size=min(N_SUBSET, n_total),
                                       replace=False)
e_flat, s_flat, w_flat = E.ravel()[idx], S.ravel()[idx], W.ravel()[idx]

t0 = time.perf_counter()
for e, s, w in zip(e_flat, s_flat, w_flat):
    defo = GaussianDeformation(r_h=BG.r_h, eps=float(e), sigma=float(s))
    reflectivity_curve(np.array([w]), M, BG, defo, r_max=R_EXTRACT)
t_cpu_subset = time.perf_counter() - t0

t_cpu_full = t_cpu_subset * (n_total / len(idx))
print(f"scipy (subset de {len(idx)}, extrapolado a {n_total}): "
      f"{t_cpu_full:8.3f} s  (medido en {t_cpu_subset:.3f} s)")

print(f"\nspeedup (malla completa, extrapolado): {t_cpu_full / t_gpu:.1f}x")
