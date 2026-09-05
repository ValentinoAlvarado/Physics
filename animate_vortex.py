"""Anima un periodo completo de la perturbacion sobre el vortice y guarda GIF.

Usa el mismo render fisico que experiments/render_vortex.py, a menor resolucion.
"""
import numpy as np
from PIL import Image
from pathlib import Path

import src.experiments.render_vortex as rv
from src.physics.background import DrainingBathtub
from src.physics.deformation import NullDeformation

rv.W, rv.HPX = 900, 620
rv.N_R, rv.N_PHI = 950, 1500

bg = DrainingBathtub(A=1.0, B=0.85, c=1.0)
defo = NullDeformation(r_h=bg.r_h)
M, OM = 2, 2.8
ELEV, AZIM0 = 30.0, -58.0
N_FRAMES = 28

period = 2.0 * np.pi / OM
frames = []
for i in range(N_FRAMES):
    t = period * i / N_FRAMES
    # la camara gira un cuarto de vuelta a lo largo del ciclo
    azim = AZIM0 + 90.0 * i / N_FRAMES
    img, _ = rv.render(bg, defo, OM, M, t, ELEV, azim)
    frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    print(f"frame {i + 1}/{N_FRAMES}", flush=True)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

frames[0].save(OUT / "vortice.gif", save_all=True, append_images=frames[1:],
               duration=70, loop=0, optimize=True)
print(OUT / "vortice.gif")
