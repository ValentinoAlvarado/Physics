"""
Geometria de la malla polar del vortice. Separa la construccion de la
superficie (radios, angulos, embudo de Bernoulli, relieve de la onda) del
render, para poder cambiar una sin tocar la otra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.background import DrainingBathtub


@dataclass
class MeshConfig:
    n_r: int = 520
    n_phi: int = 900
    r_max: float = 12.0
    r_pad: float = 1.02          # arranca en r_h * r_pad
    radial_bias: float = 2.2     # >1 concentra puntos cerca del horizonte
    g_grav: float = 0.85         # gravedad efectiva (exageracion vertical)
    relief_amp: float = 0.30     # amplitud del relieve de la onda
    relief_envelope: float = 1.35  # ancho relativo de la envolvente


class VortexMesh:
    """Malla polar con el embudo de Bernoulli y el relieve de la onda.

    La forma del embudo sale de Bernoulli en la superficie libre de un flujo
    estacionario ideal, G*eta + |v|^2/2 = const, con |v|^2 = (A^2+B^2)/r^2:

        eta_0(r) = -(A^2 + B^2) / (2 G r^2)

    G es una gravedad EFECTIVA reducida (exageracion vertical): la profundidad
    fisica real es minuscula frente al radio del dominio. Debe declararse al
    presentar la figura.
    """

    def __init__(self, bg: DrainingBathtub, cfg: MeshConfig = None):
        self.bg = bg
        self.cfg = cfg or MeshConfig()
        self._build()

    def _build(self):
        cfg = self.cfg
        r0 = self.bg.r_h * cfg.r_pad
        s = np.linspace(0.0, 1.0, cfg.n_r)
        # malla concentrada cerca del horizonte: ahi la pared del embudo es
        # casi vertical y una malla uniforme deja huecos al rasterizar
        self.r = r0 + (cfg.r_max - r0) * s ** cfg.radial_bias
        self.phi = np.linspace(0.0, 2.0 * np.pi, cfg.n_phi)
        self.PHI, self.R = np.meshgrid(self.phi, self.r)
        self.X = self.R * np.cos(self.PHI)
        self.Y = self.R * np.sin(self.PHI)
        self.eta0 = -(self.bg.A ** 2 + self.bg.B ** 2) / (
            2.0 * cfg.g_grav * self.R ** 2)
        self.envelope = np.exp(
            -((self.R - self.bg.r_h) / (cfg.relief_envelope * cfg.r_max)) ** 2)

    # ------------------------------------------------------------ relieve
    def height(self, psi: np.ndarray, omega: float, t: float,
               normalize: bool = True) -> np.ndarray:
        """eta(r,phi,t) = eta_0(r) + amp * Re[Psi e^{-i omega t}] * envolvente.

        La ALTURA lleva la onda INSTANTANEA. El color (domain coloring) lleva
        fase y modulo. Las tres son informacion distinta y no se sustituyen.
        """
        inst = np.real(psi * np.exp(-1j * omega * t))
        inst = np.nan_to_num(inst, nan=0.0)
        if normalize:
            mx = np.nanmax(np.abs(inst))
            if mx > 0:
                inst = inst / mx
        return self.eta0 + self.cfg.relief_amp * inst * self.envelope

    # ------------------------------------------------------------ normales
    @staticmethod
    def normals(X, Y, Z):
        """Normales analiticas de la malla (producto de tangentes)."""
        Xu, Xv = np.gradient(X, axis=0), np.gradient(X, axis=1)
        Yu, Yv = np.gradient(Y, axis=0), np.gradient(Y, axis=1)
        Zu, Zv = np.gradient(Z, axis=0), np.gradient(Z, axis=1)
        nx = Yu * Zv - Zu * Yv
        ny = Zu * Xv - Xu * Zv
        nz = Xu * Yv - Yu * Xv
        n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-30
        nx, ny, nz = nx / n, ny / n, nz / n
        flip = np.where(nz < 0.0, -1.0, 1.0)
        return nx * flip, ny * flip, nz * flip

    def occlusion(self) -> np.ndarray:
        """Oclusion ambiental anclada al embudo SIN perturbar: la garganta
        recibe menos cielo. Se ancla a eta0 y no a Z para que una cresta de
        onda dentro de la garganta no se ilumine como si estuviera expuesta."""
        rng = np.ptp(self.eta0) + 1e-30
        return np.clip((self.eta0 - self.eta0.min()) / (0.40 * rng),
                       0.0, 1.0) ** 0.75
