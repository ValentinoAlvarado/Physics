"""
Orquestador: compone campo + coloreado + malla + renderer en una escena.

Es la unica clase que conoce a todas las demas. Anadir una representacion
nueva del campo (otra subclase de coloreado, otra capa) no obliga a tocar el
renderer ni el campo.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import numpy as np

from ..field.analysis import (find_phase_defects, phase_gradient_cartesian)
from ..field.coloring import ColorConfig, DomainColoring
from ..field.scattering import ScatteringField
from ..physics.background import DrainingBathtub
from .mesh import MeshConfig, VortexMesh
from .renderer import CameraConfig, Renderer, ShadingConfig


@dataclass
class LayerToggles:
    """Que capas se dibujan. Cada una es independiente."""
    complex_field: bool = True      # usar Psi (si False, color plano)
    domain_coloring: bool = True    # matiz=fase, luminancia=modulo
    phase_bands: bool = True        # isolineas arg(Psi)=const
    relief: bool = True             # altura = Re[Psi e^{-i omega t}]
    phase_gradient: bool = False    # streamlines de grad(arg Psi)
    horizon: bool = True
    ergosphere: bool = True
    zeros: bool = False             # marcar dislocaciones de fase
    mesh_wire: bool = False         # rejilla de la malla


@dataclass
class SceneConfig:
    omega: float = 1.6
    m_max: int = 20   # convergencia medida: M=15 da 5e-2, M=20 da 9.5e-4
    t: float = 0.0
    mesh: MeshConfig = dc_field(default_factory=MeshConfig)
    color: ColorConfig = dc_field(default_factory=ColorConfig)
    camera: CameraConfig = dc_field(default_factory=CameraConfig)
    shading: ShadingConfig = dc_field(default_factory=ShadingConfig)
    layers: LayerToggles = dc_field(default_factory=LayerToggles)
    flat_rgb: tuple = (0.075, 0.330, 0.360)   # color si complex_field=False


class FieldVisualizer:
    """Escena completa. Cachea Psi: cambiar camara o t no reintegra ODEs."""

    def __init__(self, bg: DrainingBathtub, cfg: SceneConfig = None,
                 defo=None):
        self.bg = bg
        self.cfg = cfg or SceneConfig()
        self.defo = defo
        self.mesh = VortexMesh(bg, self.cfg.mesh)
        self.coloring = DomainColoring(self.cfg.color)
        self.renderer = Renderer(self.cfg.camera, self.cfg.shading)
        self.sfield = ScatteringField(
            bg, omega=self.cfg.omega, defo=defo, m_max=self.cfg.m_max,
            r_max=self.cfg.mesh.r_max + 1.0,
            n_out=max(900, self.cfg.mesh.n_r * 2))
        self._psi: Optional[np.ndarray] = None

    # ------------------------------------------------------------ campo
    @property
    def psi(self) -> np.ndarray:
        if self._psi is None:
            self._psi = self.sfield.evaluate(self.mesh.R, self.mesh.PHI)
        return self._psi

    def invalidate_field(self):
        """Llamar tras cambiar omega, m_max, A, B o la deformacion."""
        self._psi = None

    def set_omega(self, omega):
        self.cfg.omega = float(omega)
        self.sfield.omega = float(omega)
        self.sfield.clear_cache()
        self.invalidate_field()

    def set_m_max(self, m_max):
        self.cfg.m_max = int(m_max)
        self.sfield.m_max = int(m_max)
        self.invalidate_field()

    # ------------------------------------------------------------ geometria
    def geometry(self):
        L = self.cfg.layers
        if L.relief and L.complex_field:
            Z = self.mesh.height(self.psi, self.cfg.omega, self.cfg.t)
        else:
            Z = self.mesh.eta0
        return self.mesh.X, self.mesh.Y, Z

    # ------------------------------------------------------------ color base
    def base_color(self) -> np.ndarray:
        L = self.cfg.layers
        shape = self.mesh.R.shape
        if not (L.complex_field and L.domain_coloring):
            return np.broadcast_to(np.asarray(self.cfg.flat_rgb),
                                   shape + (3,)).copy()
        # respeta el toggle de bandas sin mutar la config del usuario
        old = self.coloring.cfg.phase_bands
        self.coloring.cfg.phase_bands = L.phase_bands
        rgb = self.coloring.rgb(self.psi)
        self.coloring.cfg.phase_bands = old
        return rgb

    # ------------------------------------------------------------ render
    def render(self):
        X, Y, Z = self.geometry()
        nx, ny, nz = self.mesh.normals(X, Y, Z)
        occ = self.mesh.occlusion()
        rgb = self.base_color()
        img, half = self.renderer.render(X, Y, Z, rgb, nx, ny, nz, occ)
        return img, half, (X, Y, Z)

    # ------------------------------------------------------- capas extra
    def horizon_curve(self, n=700):
        th = np.linspace(0, 2 * np.pi, n)
        r = self.bg.r_h
        z = -(self.bg.A ** 2 + self.bg.B ** 2) / (
            2.0 * self.cfg.mesh.g_grav * r ** 2)
        return r * np.cos(th), r * np.sin(th), np.full(n, z)

    def ergosphere_curve(self, n=700):
        th = np.linspace(0, 2 * np.pi, n)
        r = self.bg.r_e
        z = -(self.bg.A ** 2 + self.bg.B ** 2) / (
            2.0 * self.cfg.mesh.g_grav * r ** 2)
        return r * np.cos(th), r * np.sin(th), np.full(n, z)

    def phase_defects(self):
        """Dislocaciones de fase de Psi sobre la malla."""
        return find_phase_defects(self.psi, self.mesh.X, self.mesh.Y)

    def phase_gradient(self):
        """grad(arg Psi) = vector de onda local (aproximacion eikonal)."""
        return phase_gradient_cartesian(self.psi, self.mesh.X, self.mesh.Y)
