"""
Aplicacion interactiva: campo complejo Psi sobre la geometria 3D del vortice.

    python app_field.py

CONTROLES
---------
  sliders    : M (modos), omega, contraste del modulo, separacion de bandas,
               tiempo t (fase instantanea del relieve)
  checkboxes : campo complejo / domain coloring / bandas de fase /
               gradiente de fase / relieve / horizonte / ergosfera / ceros
  botones    : B=0 vs B!=0 (comparacion de simetria), guardar PNG
  arrastrar  : orbitar la camara
  rueda      : acercar/alejar

La arquitectura es modular: este archivo solo cablea widgets con
FieldVisualizer. La fisica vive en src/field y src/physics, el render en
src/viz.
"""

from __future__ import annotations
from field_legend import add_field_legend
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

from src.field.analysis import symmetry_test
from src.physics.background import DrainingBathtub
from src.viz.mesh import MeshConfig
from src.viz.visualizer import FieldVisualizer, SceneConfig

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


class App:
    def __init__(self, A=1.0, B=0.7, omega=1.6, m_max=20):
        self.bg = DrainingBathtub(A=A, B=B, c=1.0)
        cfg = SceneConfig(omega=omega, m_max=m_max)
        cfg.mesh = MeshConfig(n_r=340, n_phi=620, r_max=11.0)
        cfg.camera.width, cfg.camera.height = 1100, 760
        self.vis = FieldVisualizer(self.bg, cfg)

        self.fig = plt.figure(figsize=(13.5, 8.6), facecolor="#08080c")
        self.ax = self.fig.add_axes([0.24, 0.10, 0.74, 0.86])
        self.ax.set_axis_off()
        self.im = self.ax.imshow(np.zeros((2, 2, 3)), interpolation="bilinear")
        self.txt = self.fig.text(0.24, 0.965, "", color="#9fb6d0", fontsize=9)

        # ---------------- checkboxes de capas ----------------
        labels = ["campo complejo", "domain coloring", "bandas de fase",
                  "relieve", "grad(arg Ψ)", "horizonte", "ergosfera", "ceros"]
        init = [True, True, True, True, False, True, True, False]
        axc = self.fig.add_axes([0.015, 0.60, 0.20, 0.34],
                                facecolor="#151820")
        self.chk = CheckButtons(axc, labels, init)
        for t in self.chk.labels:
            t.set_color("#c8d4e4")
            t.set_fontsize(9)
        self.chk.on_clicked(self._on_toggle)

        # ---------------- sliders ----------------
        def mk(y, label, lo, hi, val, step=None):
            a = self.fig.add_axes([0.06, y, 0.15, 0.022],
                                  facecolor="#20242c")
            s = Slider(a, label, lo, hi, valinit=val, valstep=step,
                       color="#4f8fc0")
            s.label.set_color("#c8d4e4")
            s.label.set_fontsize(8)
            s.valtext.set_color("#c8d4e4")
            s.valtext.set_fontsize(8)
            return s

        self.s_M = mk(0.52, "M (modos)", 3, 30, m_max, 1)
        self.s_w = mk(0.47, "ω", 0.4, 4.0, omega)
        self.s_B = mk(0.42, "B", 0.0, 2.0, B)
        self.s_ctr = mk(0.37, "contraste", 0.3, 2.5, 1.0)
        self.s_band = mk(0.32, "bandas (π/n)", 2, 24, 8, 1)
        self.s_t = mk(0.27, "t", 0.0, 6.28, 0.0)

        self.s_M.on_changed(self._on_M)
        self.s_w.on_changed(self._on_omega)
        self.s_B.on_changed(self._on_B)
        self.s_ctr.on_changed(self._on_light)
        self.s_band.on_changed(self._on_light)
        self.s_t.on_changed(self._on_time)

        # ---------------- botones ----------------
        ab = self.fig.add_axes([0.06, 0.19, 0.07, 0.04])
        self.b_sym = Button(ab, "test B=0", color="#2a3340",
                            hovercolor="#3d4a5c")
        self.b_sym.label.set_color("#c8d4e4")
        self.b_sym.label.set_fontsize(8)
        self.b_sym.on_clicked(self._on_symmetry)

        ab2 = self.fig.add_axes([0.14, 0.19, 0.07, 0.04])
        self.b_save = Button(ab2, "guardar", color="#2a3340",
                             hovercolor="#3d4a5c")
        self.b_save.label.set_color("#c8d4e4")
        self.b_save.label.set_fontsize(8)
        self.b_save.on_clicked(self._on_save)

        # ---------------- raton ----------------
        self.drag = False
        self.last = None
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._press)
        c.mpl_connect("button_release_event", self._release)
        c.mpl_connect("motion_notify_event", self._motion)
        c.mpl_connect("scroll_event", self._scroll)

        self.draw()

    # ------------------------------------------------------------ callbacks
    def _on_toggle(self, label):
        L = self.vis.cfg.layers
        m = {"campo complejo": "complex_field",
             "domain coloring": "domain_coloring",
             "bandas de fase": "phase_bands",
             "relieve": "relief",
             "grad(arg Ψ)": "phase_gradient",
             "horizonte": "horizon",
             "ergosfera": "ergosphere",
             "ceros": "zeros"}
        attr = m[label]
        setattr(L, attr, not getattr(L, attr))
        self.draw()

    def _on_M(self, v):
        self.vis.set_m_max(int(v))
        self.draw()

    def _on_omega(self, v):
        self.vis.set_omega(float(v))
        self.draw()

    def _on_B(self, v):
        self.bg = DrainingBathtub(A=self.bg.A, B=float(v), c=self.bg.c)
        cfg = self.vis.cfg
        self.vis = FieldVisualizer(self.bg, cfg)
        self.draw()

    def _on_light(self, _v):
        self.vis.cfg.color.contrast = float(self.s_ctr.val)
        self.vis.cfg.color.band_spacing = np.pi / float(self.s_band.val)
        self.draw()

    def _on_time(self, v):
        self.vis.cfg.t = float(v)
        self.draw()

    def _on_symmetry(self, _e):
        r = np.linspace(self.bg.r_h * 1.05, 11.0, 90)
        phi = np.linspace(-np.pi, np.pi, 120)
        R, PHI = np.meshgrid(r, phi, indexing="ij")
        res = symmetry_test(self.vis.sfield, R, PHI)
        print(f"[simetria] B={self.bg.B:.3f}  "
              f"asimetria = {res.asymmetry:.3e}  "
              + ("(simetrico: correcto para B=0)" if self.bg.B == 0 else
                 "(ruptura por arrastre de marco)"))

    def _on_save(self, _e):
        img, _, _ = self.vis.render()
        name = (f"campo_A{self.bg.A:.2f}_B{self.bg.B:.2f}"
                f"_w{self.vis.cfg.omega:.2f}_M{self.vis.cfg.m_max}.png")
        fig = plt.figure(figsize=(11, 7.6), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.set_axis_off()
        fig.savefig(OUT / name, dpi=150, facecolor="black")
        plt.close(fig)
        print("guardado:", OUT / name)

    def _press(self, e):
        if e.inaxes is self.ax and e.button == 1:
            self.drag = True
            self.last = (e.x, e.y)

    def _release(self, _e):
        self.drag = False

    def _motion(self, e):
        if not self.drag or self.last is None or e.x is None:
            return
        dx, dy = e.x - self.last[0], e.y - self.last[1]
        self.last = (e.x, e.y)
        cam = self.vis.cfg.camera
        cam.azim += dx * 0.4
        cam.elev = float(np.clip(cam.elev + dy * 0.4, 5.0, 88.0))
        self.draw()

    def _scroll(self, e):
        c = self.vis.cfg.camera
        c.pad = float(np.clip(c.pad * (0.92 if e.step > 0 else 1 / 0.92),
                              0.3, 4.0))
        self.draw()

    # ------------------------------------------------------------ dibujo
    def draw(self):
        img, half, (X, Y, Z) = self.vis.render()
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(img, interpolation="bilinear")
        L = self.vis.cfg.layers
        R = self.vis.renderer

        if L.horizon:
            x, y, z = self.vis.horizon_curve()
            px, py = R.project_curve(x, y, z, half)
            self.ax.plot(px, py, "-", color="#ff6a3d", lw=1.6)
        if L.ergosphere:
            x, y, z = self.vis.ergosphere_curve()
            px, py = R.project_curve(x, y, z, half)
            self.ax.plot(px, py, "--", color="#ffc04d", lw=1.4)
        if L.zeros and L.complex_field:
            zs = self.vis.phase_defects()
            if zs:
                zz = self.vis.mesh.eta0
                xs = np.array([d.x for d in zs])
                ys = np.array([d.y for d in zs])
                zv = np.array([zz[d.iy, d.ix] for d in zs])
                px, py = R.project_curve(xs, ys, zv, half)
                pos = np.array([d.charge > 0 for d in zs])
                self.ax.plot(px[pos], py[pos], "o", ms=3.5, mfc="none",
                             mec="#ffffff", mew=0.9)
                self.ax.plot(px[~pos], py[~pos], "s", ms=3.5, mfc="none",
                             mec="#7fd4ff", mew=0.9)
        if L.phase_gradient and L.complex_field:
            self._draw_phase_streamlines(half)

        self.ax.set_xlim(0, self.vis.cfg.camera.width)
        self.ax.set_ylim(self.vis.cfg.camera.height, 0)
        self.txt.set_text(
            f"A={self.bg.A:.2f}  B={self.bg.B:.2f}  ω={self.vis.cfg.omega:.2f}"
            f"  M={self.vis.cfg.m_max}  |  r_h={self.bg.r_h:.2f}"
            f"  r_e={self.bg.r_e:.2f}  Ω_H={self.bg.Omega_H:.2f}"
            f"  |  altura=Re[Ψe^{{-iωt}}]  matiz=arg Ψ  luminancia=|Ψ|")
        self.fig.canvas.draw_idle()

    def _draw_phase_streamlines(self, half, n_seed=26, n_step=140, ds=0.09):
        """Lineas de flujo de grad(arg Psi) = vector de onda local.

        En la aproximacion eikonal este k entra en omega = v.k + c|k|, la
        misma relacion de dispersion que integra el ray tracer: comparar
        estas lineas con las trayectorias RK4 conecta la descripcion
        ondulatoria con la geometrica.
        """
        pg = self.vis.phase_gradient()
        X, Y = self.vis.mesh.X, self.vis.mesh.Y
        eta = self.vis.mesh.eta0
        r_h, r_max = self.bg.r_h, self.vis.cfg.mesh.r_max

        kx = np.nan_to_num(pg.kx)
        ky = np.nan_to_num(pg.ky)

        def sample(field, xq, yq):
            rq = np.hypot(xq, yq)
            pq = np.mod(np.arctan2(yq, xq), 2 * np.pi)
            ir = np.clip(np.searchsorted(self.vis.mesh.r, rq) - 1,
                         0, field.shape[0] - 1)
            ip = np.clip(np.searchsorted(self.vis.mesh.phi, pq) - 1,
                         0, field.shape[1] - 1)
            return field[ir, ip]

        th0 = np.linspace(0, 2 * np.pi, n_seed, endpoint=False)
        r0 = r_max * 0.95
        for th in th0:
            x, y = r0 * np.cos(th), r0 * np.sin(th)
            xs, ys, zs = [], [], []
            for _ in range(n_step):
                r = np.hypot(x, y)
                if r < r_h * 1.05 or r > r_max:
                    break
                ux = float(sample(kx, x, y))
                uy = float(sample(ky, x, y))
                nrm = np.hypot(ux, uy)
                if not np.isfinite(nrm) or nrm < 1e-12:
                    break
                # se sigue -k: hacia donde converge la fase (hacia el vortice)
                x -= ds * ux / nrm
                y -= ds * uy / nrm
                xs.append(x); ys.append(y)
                zs.append(float(sample(eta, x, y)))
            if len(xs) > 4:
                px, py = self.vis.renderer.project_curve(
                    np.array(xs), np.array(ys), np.array(zs), half)
                self.ax.plot(px, py, "-", color="#ffffff", lw=0.7, alpha=0.5)


if __name__ == "__main__":
    print("Cargando campo (integra 2M+1 ODEs radiales)...")
    matplotlib.use("TkAgg")     # si falla: "QtAgg"
    app = App()
    plt.show()
