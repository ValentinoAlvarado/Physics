"""
Render de la superficie libre del vortice acustico rotante.

Base fisica
-----------
Forma del embudo, por Bernoulli en la superficie libre de un flujo estacionario
ideal:  G*eta + |v|^2/2 = const, con |v|^2 = (A^2+B^2)/r^2, de donde

    eta_0(r) = -(A^2+B^2)/(2 G r^2)

Perturbacion de la superficie, proporcional a -d_t(Phi)/G con
Phi = Psi(r,phi) e^{-i omega t} y Psi = H(r) r^{-1/2} e^{i m phi} obtenido por
integracion radial de d^2H/dr_*^2 + V_h H = 0:

    delta_eta(r,phi,t) ~ Re[ i omega Psi e^{-i omega t} ]

Render
------
Rasterizador propio con z-buffer y sombreado Blinn-Phong por pixel sobre las
normales analiticas de la malla, mas Fresnel de Schlick y atenuacion por
profundidad. No usa el sombreado por defecto de matplotlib.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

from src.numerics.radial import radial_profile
from src.physics.background import DrainingBathtub
from src.physics.deformation import Deformation, NullDeformation

# Escala vertical de render. La profundidad fisica del embudo es minuscula
# frente al radio del dominio, asi que se usa una gravedad efectiva reducida
# (exageracion vertical). No altera la fisica de la onda, solo la vista.
G_GRAV = 0.85
W, HPX = 1500, 1000
N_R, N_PHI = 1500, 2400


def build_field(bg, defo, omega, m, r_max, amp):
    r_s, H = radial_profile(omega, m, bg, defo, delta=2e-3,
                            r_max=r_max, n_out=12000)
    # malla concentrada cerca del horizonte: la pared del embudo es
    # casi vertical ahi y una malla uniforme deja huecos al rasterizar
    s_ = np.linspace(0.0, 1.0, N_R)
    r0_ = r_s.min()
    r_g = r0_ + (r_max - r0_) * s_**3.0
    Hg = np.interp(r_g, r_s, H.real) + 1j * np.interp(r_g, r_s, H.imag)
    Rr = Hg / np.sqrt(r_g)
    Rr = Rr / np.abs(Rr).max()

    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    PHI, RR = np.meshgrid(phi, r_g)
    PSI = Rr[:, None] * np.exp(1j * m * PHI)

    X = RR * np.cos(PHI)
    Y = RR * np.sin(PHI)
    eta0 = -(bg.A**2 + bg.B**2) / (2.0 * G_GRAV * RR**2)
    env = amp * np.exp(-((RR - bg.r_h) / (1.35 * r_max)) ** 2)
    return X, Y, eta0, PSI * env


def camera(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return d, right, up


def normals(X, Y, Z):
    Xu, Xv = np.gradient(X, axis=0), np.gradient(X, axis=1)
    Yu, Yv = np.gradient(Y, axis=0), np.gradient(Y, axis=1)
    Zu, Zv = np.gradient(Z, axis=0), np.gradient(Z, axis=1)
    nx = Yu * Zv - Zu * Yv
    ny = Zu * Xv - Xu * Zv
    nz = Xu * Yv - Yu * Xv
    n = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-30
    nx, ny, nz = nx / n, ny / n, nz / n
    flip = np.where(nz < 0.0, -1.0, 1.0)
    return nx * flip, ny * flip, nz * flip


def shade(nx, ny, nz, Z, occ, view, light=(-0.50, -0.60, 0.62)):
    L = np.asarray(light, float); L = L / np.linalg.norm(L)
    V = np.asarray(view, float); V = V / np.linalg.norm(V)
    Hh = (L + V) / np.linalg.norm(L + V)

    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
    spec = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0.0, 1.0) ** 60
    cosv = np.clip(nx * V[0] + ny * V[1] + nz * V[2], 0.0, 1.0)
    fres = 0.03 + 0.97 * (1.0 - cosv) ** 5

    rng = float(Z.max() - Z.min()) + 1e-30
    depth = np.clip((Z - Z.min()) / rng, 0.0, 1.0)
    deep = np.array([0.012, 0.045, 0.090])
    shal = np.array([0.070, 0.330, 0.360])
    base = deep + (shal - deep) * depth[..., None]

    sky = np.array([0.40, 0.58, 0.76])
    # oclusion ambiental aproximada: la garganta del embudo recibe menos
    # cielo, de modo que Fresnel y especular se atenuan con la profundidad
    rgb = base * (0.10 + 0.90 * diff[..., None]) * (0.25 + 0.75 * occ[..., None])
    rgb = rgb + sky * (fres * 0.55 * occ)[..., None]
    rgb = rgb + (spec * 1.05 * occ)[..., None] * np.array([1.0, 0.98, 0.93])
    return np.clip(rgb, 0.0, 1.0)


def project(X, Y, Z, elev, azim, pad=1.06):
    d, right, up = camera(elev, azim)
    su = X * right[0] + Y * right[1] + Z * right[2]
    sv = X * up[0] + Y * up[1] + Z * up[2]
    dep = X * d[0] + Y * d[1] + Z * d[2]
    half = pad * max(float(np.abs(su).max()),
                     float(np.abs(sv).max()) * W / HPX)
    return su, sv, dep, half


def to_pixels(su, sv, half):
    px = ((su / half * 0.5 + 0.5) * (W - 1))
    py = ((1.0 - (sv / (half * HPX / W) * 0.5 + 0.5)) * (HPX - 1))
    return px, py


def rasterize(X, Y, Z, rgb, elev, azim):
    su, sv, dep, half = project(X, Y, Z, elev, azim)
    pxf, pyf = to_pixels(su, sv, half)
    px, py = pxf.astype(np.int32), pyf.astype(np.int32)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < HPX)

    idx = (py[ok].astype(np.int64) * W + px[ok].astype(np.int64))
    dz = dep[ok]
    cols = rgb[ok]

    zbuf = np.full(W * HPX, -np.inf)
    np.maximum.at(zbuf, idx, dz)
    win = dz >= zbuf[idx] - 1e-12

    img = np.zeros((W * HPX, 3))
    mask = np.zeros(W * HPX, bool)
    img[idx[win]] = cols[win]
    mask[idx[win]] = True
    return img.reshape(HPX, W, 3), mask.reshape(HPX, W), half


def fill_holes(img, mask, passes=3):
    for _ in range(passes):
        if mask.all():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros(mask.shape)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            acc += np.roll(np.roll(img, dy, 0), dx, 1) * \
                np.roll(np.roll(mask, dy, 0), dx, 1)[..., None]
            cnt += np.roll(np.roll(mask, dy, 0), dx, 1)
        hole = (~mask) & (cnt > 0)
        img[hole] = acc[hole] / cnt[hole][..., None]
        mask = mask | hole
    return img, mask


def composite(img, mask):
    grad = np.linspace(0.0, 1.0, HPX)[:, None, None]
    back = np.array([0.005, 0.008, 0.016]) * grad + \
        np.array([0.020, 0.028, 0.046]) * (1.0 - grad)
    out = np.where(mask[..., None], img, back)
    out = out / (1.0 + out)
    return np.clip(out * 1.80, 0.0, 1.0) ** (1.0 / 1.9)



def render(bg, defo: Deformation, omega, m, t, elev, azim,
           r_max=7.0, amp=0.34):
    X, Y, eta0, PSIe = build_field(bg, defo, omega, m, r_max, amp)
    Z = eta0 + np.real(1j * omega * PSIe * np.exp(-1j * omega * t))
    nx, ny, nz = normals(X, Y, Z)
    d, _, _ = camera(elev, azim)
    # la oclusion se ancla al embudo SIN perturbar: si se anclase a Z, la
    # cresta de una onda dentro de la garganta se iluminaria como si
    # estuviera expuesta al cielo
    occ = np.clip((eta0 - eta0.min()) / (0.40 * (np.ptp(eta0) + 1e-30)),
                  0.0, 1.0) ** 0.75
    rgb = shade(nx, ny, nz, Z, occ, view=d)
    img, mask, half = rasterize(X, Y, Z, rgb, elev, azim)
    img, mask = fill_holes(img, mask)
    return composite(img, mask), half


def ring(ax, bg, radius, half, elev, azim, style, color, label):
    _, right, up = camera(elev, azim)
    th = np.linspace(0.0, 2.0 * np.pi, 700)
    z = -(bg.A**2 + bg.B**2) / (2.0 * G_GRAV * radius**2)
    x, y = radius * np.cos(th), radius * np.sin(th)
    su = x * right[0] + y * right[1] + z * right[2]
    sv = x * up[0] + y * up[1] + z * up[2]
    px, py = to_pixels(su, sv, half)
    ax.plot(px, py, style, color=color, lw=1.9, label=label, zorder=5)


def main():
    bg = DrainingBathtub(A=1.0, B=0.85, c=1.0)
    defo = NullDeformation(r_h=bg.r_h)
    m, omega = 2, 2.8
    elev, azim = 30.0, -58.0

    img, half = render(bg, defo, omega, m, 0.0, elev, azim)

    fig = plt.figure(figsize=(W / 150, HPX / 150), facecolor="#04060a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="bilinear")
    ring(ax, bg, bg.r_h, half, elev, azim, "-", "#ff6a3d", "horizonte acústico")
    ring(ax, bg, bg.r_e, half, elev, azim, "--", "#ffc04d", "ergosfera")
    ax.set_xlim(0, W); ax.set_ylim(HPX, 0); ax.set_axis_off()

    leg = ax.legend(loc="lower right", frameon=False, fontsize=12)
    for t in leg.get_texts():
        t.set_color("#c8d4e4")
    ax.text(0.030, 0.955,
            "Vórtice acústico rotante — superficie libre y perturbación",
            transform=ax.transAxes, color="#eaf0f8", fontsize=15, va="top")
    ax.text(0.030, 0.905,
            rf"$m={m}$,  $\omega={omega}$  ($m\Omega_H={m*bg.Omega_H:.2f}$),  "
            rf"$r_h={bg.r_h:.2f}$,  $r_e={bg.r_e:.2f}$",
            transform=ax.transAxes, color="#8fa3bd", fontsize=11, va="top")

    fig.savefig(OUT / "render_vortice.png", dpi=150, facecolor="#04060a")
    print("results/render_vortice.png")


if __name__ == "__main__":
    main()
