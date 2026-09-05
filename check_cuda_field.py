"""Verifica el kernel CUDA del solver radial contra scipy. Corre esto PRIMERO.

    python check_cuda_field.py
"""
from src.physics.background import DrainingBathtub
from src.field import cuda_radial

if not cuda_radial.HAVE_CUPY:
    raise SystemExit("CuPy no instalado:  pip install cupy-cuda12x\n"
                     "(sin el, app_field.py cae a scipy y funciona igual, "
                     "solo que mas lento)")

import cupy as cp
props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
print(f"GPU: {props['name'].decode()}  SMs={props['multiProcessorCount']}\n")

bg = DrainingBathtub(A=1.0, B=0.7, c=1.0)
ok = cuda_radial.validate_against_scipy(bg, omega=1.6, m_values=range(-6, 7))
print("\n->", "kernel validado" if ok else
      "DISCREPANCIA: revisa signos en el kernel antes de usarlo")
