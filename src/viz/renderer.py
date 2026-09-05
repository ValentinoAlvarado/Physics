"""
Renderizado: camara ortografica, rasterizador con z-buffer y sombreado.

Principio de diseno: el sombreado MODULA el color del domain coloring, no lo
reemplaza. Asi se conserva a la vez la percepcion 3D del relieve y la lectura
matematica del color (fase en el matiz, modulo en la luminancia).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    elev: float = 30.0
    azim: float = -58.0
    width: int = 1500
    height: int = 1000
    pad: float = 1.06


@dataclass
class ShadingConfig:
    light: tuple = (-0.50, -0.60, 0.62)
    ambient: float = 0.35        # cuanto color pasa sin iluminacion directa
    diffuse: float = 0.65
    specular: float = 0.30
    shininess: float = 60.0
    use_occlusion: bool = True
    background_top: tuple = (0.020, 0.028, 0.046)
    background_bottom: tuple = (0.005, 0.008, 0.016)
    tone_map: bool = True
    exposure: float = 1.55
    gamma_out: float = 1.9


def camera_basis(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return d, right, up


class Renderer:
    """Rasterizador con z-buffer. Recibe geometria + color base y devuelve
    la imagen final."""

    def __init__(self, cam: CameraConfig = None, shading: ShadingConfig = None):
        self.cam = cam or CameraConfig()
        self.sh = shading or ShadingConfig()

    # ------------------------------------------------------------ proyeccion
    def project(self, X, Y, Z):
        d, right, up = camera_basis(self.cam.elev, self.cam.azim)
        su = X * right[0] + Y * right[1] + Z * right[2]
        sv = X * up[0] + Y * up[1] + Z * up[2]
        dep = X * d[0] + Y * d[1] + Z * d[2]
        W, H = self.cam.width, self.cam.height
        half = self.cam.pad * max(float(np.abs(su).max()),
                                  float(np.abs(sv).max()) * W / H)
        return su, sv, dep, half

    def to_pixels(self, su, sv, half):
        W, H = self.cam.width, self.cam.height
        px = (su / half * 0.5 + 0.5) * (W - 1)
        py = (1.0 - (sv / (half * H / W) * 0.5 + 0.5)) * (H - 1)
        return px, py

    # ------------------------------------------------------------ sombreado
    def shade(self, base_rgb, nx, ny, nz, occ=None):
        """Modula el color base con iluminacion Blinn-Phong.

        base_rgb es el color del domain coloring: su MATIZ (la fase) se
        conserva exactamente; solo se escala su intensidad. Asi la lectura
        matematica del color sobrevive al sombreado.
        """
        sh = self.sh
        d, _, _ = camera_basis(self.cam.elev, self.cam.azim)
        L = np.asarray(sh.light, float)
        L = L / np.linalg.norm(L)
        V = d / np.linalg.norm(d)
        Hh = (L + V) / np.linalg.norm(L + V)

        diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
        spec = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2],
                       0.0, 1.0) ** sh.shininess

        lit = sh.ambient + sh.diffuse * diff
        if occ is not None and sh.use_occlusion:
            lit = lit * (0.30 + 0.70 * occ)
        rgb = base_rgb * lit[..., None]
        # el especular se anade en blanco: no altera el matiz de fondo, solo
        # marca la orientacion de la superficie
        rgb = rgb + (spec * sh.specular)[..., None]
        return np.clip(rgb, 0.0, 1.0)

    # ------------------------------------------------------------ rasterizado
    def rasterize(self, X, Y, Z, rgb):
        su, sv, dep, half = self.project(X, Y, Z)
        pxf, pyf = self.to_pixels(su, sv, half)
        W, H = self.cam.width, self.cam.height
        px, py = pxf.astype(np.int32), pyf.astype(np.int32)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        idx = py[ok].astype(np.int64) * W + px[ok].astype(np.int64)
        dz = dep[ok]
        cols = rgb[ok]

        zbuf = np.full(W * H, -np.inf)
        np.maximum.at(zbuf, idx, dz)
        win = dz >= zbuf[idx] - 1e-12

        img = np.zeros((W * H, 3))
        mask = np.zeros(W * H, bool)
        img[idx[win]] = cols[win]
        mask[idx[win]] = True
        return img.reshape(H, W, 3), mask.reshape(H, W), half

    @staticmethod
    def fill_holes(img, mask, passes=3):
        for _ in range(passes):
            if mask.all():
                break
            acc = np.zeros_like(img)
            cnt = np.zeros(mask.shape)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sm = np.roll(np.roll(mask, dy, 0), dx, 1)
                acc += np.roll(np.roll(img, dy, 0), dx, 1) * sm[..., None]
                cnt += sm
            hole = (~mask) & (cnt > 0)
            img[hole] = acc[hole] / cnt[hole][..., None]
            mask = mask | hole
        return img, mask

    def composite(self, img, mask):
        sh = self.sh
        H = self.cam.height
        grad = np.linspace(0.0, 1.0, H)[:, None, None]
        back = (np.array(sh.background_bottom) * grad
                + np.array(sh.background_top) * (1.0 - grad))
        out = np.where(mask[..., None], img, back)
        if sh.tone_map:
            out = out / (1.0 + out)
            out = np.clip(out * sh.exposure, 0.0, 1.0) ** (1.0 / sh.gamma_out)
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------ pipeline
    def render(self, X, Y, Z, base_rgb, nx, ny, nz, occ=None):
        rgb = self.shade(base_rgb, nx, ny, nz, occ)
        img, mask, half = self.rasterize(X, Y, Z, rgb)
        img, mask = self.fill_holes(img, mask)
        return self.composite(img, mask), half

    # -------------------------------------------------- overlays geometricos
    def project_curve(self, x, y, z, half):
        """Proyecta una curva 3D a pixeles (para horizonte, ergosfera, rayos)."""
        _, right, up = camera_basis(self.cam.elev, self.cam.azim)
        su = x * right[0] + y * right[1] + z * right[2]
        sv = x * up[0] + y * up[1] + z * up[2]
        return self.to_pixels(su, sv, half)
