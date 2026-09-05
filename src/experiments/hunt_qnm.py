"""
Retrato de fase (domain coloring) riguroso de log_match(omega) sobre el plano
complejo, en PyTorch/CUDA.

Convencion: coloreado de dominio MEJORADO (Wegert, "Visual Complex Functions",
2012) -- el estandar en analisis complejo y en la comunidad de QNM/pseudo-
espectros para visualizar funciones meromorfas de una variable compleja. No es
"colores bonitos": cada elemento tiene significado analitico preciso.

- Matiz (hue)      : arg(M) en [-pi, pi). Constante a lo largo de rayos que
                     salen de un cero o polo -- por eso un cero real se ve
                     como TODO el disco de color confluyendo en un punto.
- Anillos          : bandas de log2|M| -- cada anillo es una octava del
                     modulo. La DENSIDAD de anillos alrededor de un punto da
                     directamente el ORDEN del cero o del polo.
- Lineas radiales  : bandas de arg(M) cada pi/8 -- el NUMERO de sectores de
                     color que confluyen en un punto da el INDICE topologico
                     (principio del argumento, contado a simple vista).
- Normalizacion    : con np.nanpercentile (no np.percentile) -- un solo pixel
                     con NaN cerca de una singularidad de la serie asintotica
                     (omega=0 exactamente) ya NO contamina toda la imagen.

CORRECCION DE ESTA VERSION respecto a la anterior
--------------------------------------------------
1. `_asymptotic_series` diverge en omega=0 exacto (k=omega/c en el
   denominador). Un solo pixel con Re(omega)=0 en la rejilla producia NaN,
   y como `phase_portrait` usaba `np.percentile` (no NaN-safe), UN pixel
   invalido volvia blanca la imagen COMPLETA. Corregido:
   (a) `phase_portrait` ahora usa `np.nanpercentile` en todos los calculos
       de normalizacion, y hace `np.nan_to_num` sobre el HSV final.
   (b) `plot_hunt` evita omega=0 por construccion: si `re_range` incluye 0,
       se desplaza el primer punto de la rejilla lejos de el.

Backend: PyTorch, complex128, en GPU si hay CUDA disponible. Toda la rejilla
se integra en paralelo (RK4 de paso fijo vectorizado sobre el batch), sin
bucle Python por pixel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128
RDTYPE = torch.float64


# ------------------------------------------------------------ fondo y potencial
class Background:
    """Espejo minimo de src.physics.background.DrainingBathtub en torch."""

    def __init__(self, A, B, c=1.0):
        self.A, self.B, self.c = float(A), float(B), float(c)
        self.r_h = abs(self.A) / self.c
        self.Omega_H = self.B * self.c**2 / self.A**2


def g_metric(r, bg: Background):
    """g(r) = 1 - A^2/(c^2 r^2), fondo sin deformar."""
    return 1.0 - (bg.A**2 / (bg.c**2 * r**2))


def dg_metric(r, bg: Background):
    return 2.0 * bg.A**2 / (bg.c**2 * r**3)


def V_effective(r, omega, m, bg: Background):
    """V_h(r;omega,m) para el fondo NO deformado (h=0), forma cerrada
    verificada (reduce identicamente a la expresion estandar del draining
    bathtub -- ver src/mathematics/potential.py, test_reduccion_potencial).
    """
    g = g_metric(r, bg)
    dg = dg_metric(r, bg)
    drag = (omega - m * bg.B / r**2) ** 2 / bg.c**2
    barrier = g * (m**2 / r**2 + dg / (2.0 * r) - g / (4.0 * r**2))
    return drag - barrier


def asymptotic_series(omega, m, bg: Background):
    """Coeficientes b1..b4 de H_out=e^{ikr}(1+b1/r+...+b4/r^4).

    Derivados por sympy (ver src/numerics/qnm.py, _asymptotic_series):
        L  = 2 m B omega / c^2 + m^2 - 1/4
        c4 = (A^2 (m^2 - 3/2) + B^2 m^2) / c^2
    c3 = 0 exactamente: este sistema no tiene cola tipo Coulomb (~1/r).

    ADVERTENCIA: diverge en omega=0 (k=omega/c en el denominador). No evaluar
    en omega=0 exacto; `plot_hunt` ya evita ese punto en la rejilla.
    """
    c = bg.c
    A, B = bg.A, bg.B
    k = omega / c
    L = 2.0 * m * B * omega / c**2 + (m**2 - 0.25)
    c4 = (A**2 * (m**2 - 1.5) + B**2 * m**2) / c**2

    b1 = 1j * L / (2.0 * k)
    b2 = (-L**2 + 2.0 * L) / (8.0 * k**2)
    b3 = (-1j * L**3 + 8j * L**2 - 12j * L + 8j * c4 * k**2) / (48.0 * k**3)
    b4 = (L**4 - 20 * L**3 + 108 * L**2 - 144 * L
          - 32 * L * c4 * k**2 + 96 * c4 * k**2) / (384.0 * k**4)
    return b1, b2, b3, b4


def series_value_deriv(r, k, b):
    b1, b2, b3, b4 = b
    f = 1.0 + b1 / r + b2 / r**2 + b3 / r**3 + b4 / r**4
    df = -b1 / r**2 - 2 * b2 / r**3 - 3 * b3 / r**4 - 4 * b4 / r**5
    phase = torch.exp(1j * k * r)
    H = phase * f
    dH = phase * (1j * k * f + df)
    return H, dH


# ------------------------------------------------------------- RK4 vectorizado
def rk4_batch(rhs, y0, s0, s1, n_steps):
    """y0: tensor complejo (3, N). Devuelve y(s1), misma forma."""
    h = (s1 - s0) / n_steps
    y = y0.clone()
    s = s0
    for _ in range(n_steps):
        k1 = rhs(s, y)
        k2 = rhs(s + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(s + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(s + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s = s + h
    return y


def log_match_grid(omega_grid, m, bg: Background,
                   r_match_factor=6.0, r_inf=120.0,
                   n_out=1600, n_in=1600, delta=1e-8):
    """log_match(omega) = H1'/H1 - H2'/H2 en r_match, sobre TODA la rejilla.

    omega_grid: tensor complejo (Ny, Nx) en DEVICE. Devuelve tensor (Ny, Nx).
    No pasar ningun omega con Re=Im=0 exacto (ver `asymptotic_series`).
    """
    shape = omega_grid.shape
    W = omega_grid.reshape(-1).to(CDTYPE)
    N = W.numel()
    r_h = bg.r_h
    r_match = r_match_factor * r_h

    # --- disparo desde el horizonte, hacia afuera (condicion entrante) ---
    r0 = r_h * (1.0 + delta)
    kH = (W - m * bg.Omega_H) / bg.c
    y0 = torch.stack([
        torch.full((N,), r0, dtype=CDTYPE, device=DEVICE),
        torch.ones(N, dtype=CDTYPE, device=DEVICE),
        -1j * kH,
    ])

    def rhs_out(_s, y):
        r = y[0].real
        g = g_metric(r, bg).to(CDTYPE)
        V = V_effective(r, W, m, bg)
        return torch.stack([g, y[2], -V * y[1]])

    s1 = 2.0 * r_h * abs(np.log(delta)) + 2.0 * r_match + 30.0
    y_h = rk4_batch(rhs_out, y0, 0.0, s1, n_out)
    H1, P1 = y_h[1], y_h[2]
    g1 = g_metric(torch.tensor(r_match, dtype=RDTYPE), bg)
    dH1 = P1 / g1

    # --- disparo desde infinito, hacia adentro (condicion saliente pura) ---
    k = W / bg.c
    b_out = asymptotic_series(W, m, bg)
    H0, dHdr0 = series_value_deriv(
        torch.tensor(r_inf, dtype=CDTYPE, device=DEVICE), k, b_out)
    P0 = g_metric(torch.tensor(r_inf, dtype=RDTYPE), bg) * dHdr0
    y0b = torch.stack([
        torch.full((N,), r_inf, dtype=CDTYPE, device=DEVICE), H0, P0,
    ])

    def rhs_in(_s, y):
        r = y[0].real
        g = g_metric(r, bg).to(CDTYPE)
        V = V_effective(r, W, m, bg)
        return torch.stack([-g, -y[2], V * y[1]])

    s2 = 2.0 * (r_inf - r_match) + 30.0
    y_i = rk4_batch(rhs_in, y0b, 0.0, s2, n_in)
    H2, P2 = y_i[1], y_i[2]
    g2 = g_metric(torch.tensor(r_match, dtype=RDTYPE), bg)
    dH2 = P2 / g2

    M = dH1 / H1 - dH2 / H2
    return M.reshape(shape)


# ---------------------------------------------------- retrato de fase (Wegert)
def phase_portrait(M, mod_octave_bands=True, phase_bands=16,
                   band_contrast=0.28, p_lo=2.0, p_hi=98.0):
    """Retrato de fase mejorado (Wegert). M: array complejo de NumPy.

    Devuelve RGB en [0,1]^3, listo para imshow. NaN-safe: un pixel invalido
    (p.ej. cerca de omega=0) ya no arruina la normalizacion de toda la
    imagen -- se usa nanpercentile y se limpia el resultado al final.
    """
    mag = np.abs(M)
    mag = np.where(np.isfinite(mag), mag, np.nan)
    mag_safe = np.clip(np.where(np.isnan(mag), np.nanmin(mag[mag > 0]), mag),
                       1e-300, None)
    hue = (np.angle(M) + np.pi) / (2.0 * np.pi)
    hue = np.nan_to_num(hue, nan=0.0)

    val = np.ones_like(mag_safe)
    if mod_octave_bands:
        log2mag = np.log2(mag_safe)
        lo, hi = np.nanpercentile(log2mag, [p_lo, p_hi])
        log2m = np.clip((log2mag - lo) / max(hi - lo, 1e-12), 0, 1)
        frac = np.mod(log2m * 6.0, 1.0)
        ring = 0.5 + 0.5 * np.cos(2 * np.pi * frac)
        val = val * (1.0 - band_contrast + band_contrast * ring)

    if phase_bands:
        frac_p = np.mod(np.angle(M) / (2 * np.pi / phase_bands), 1.0)
        frac_p = np.nan_to_num(frac_p, nan=0.0)
        sector = 0.5 + 0.5 * np.cos(2 * np.pi * frac_p)
        val = val * (1.0 - band_contrast + band_contrast * sector)

    floor = np.nanpercentile(mag_safe, 1) + 1e-300
    near_zero = np.clip(1.0 - np.log2(mag_safe / floor) / 6.0, 0, 1)
    val = np.clip(val * (0.15 + 0.85 * (1 - 0.6 * near_zero)), 0, 1)

    sat = np.full_like(hue, 0.90)
    hsv = np.stack([hue, sat, val], axis=-1)
    hsv = np.nan_to_num(hsv, nan=0.0, posinf=1.0, neginf=0.0)
    hsv = np.clip(hsv, 0.0, 1.0)
    return hsv_to_rgb(hsv)


def plot_hunt(bg: Background, m, re_range, im_range, n=700, save=None, **kw):
    """Domain-coloring de log_match sobre [re_range] x [im_range].

    Evita automaticamente omega=0 exacto (singularidad de la serie
    asintotica) desplazando el primer punto de la rejilla si re_range o
    im_range incluyen el cero.
    """
    re0, re1 = re_range
    im0, im1 = im_range
    if re0 == 0.0:
        re0 = (re1 - re_range[0]) / (n * 4)  # pequeno desplazamiento
    if im0 == 0.0 and im1 == 0.0:
        raise ValueError("im_range no puede ser (0,0)")

    re = torch.linspace(re0, re1, n, dtype=RDTYPE, device=DEVICE)
    im = torch.linspace(im0, im1, n, dtype=RDTYPE, device=DEVICE)
    RE, IM = torch.meshgrid(re, im, indexing="xy")
    omega_grid = (RE + 1j * IM).to(CDTYPE)

    # blindaje adicional: si algun punto cayo exactamente en 0+0j (rejillas
    # que cruzan el origen en ambos ejes a la vez), lo desplazamos un epsilon
    zero_mask = (omega_grid.real == 0) & (omega_grid.imag == 0)
    if zero_mask.any():
        eps = (re1 - re0) / (n * 100)
        omega_grid = torch.where(zero_mask, omega_grid + eps, omega_grid)

    with torch.no_grad():
        M = log_match_grid(omega_grid, m, bg, **kw)
    M_np = M.detach().cpu().numpy()

    rgb = phase_portrait(M_np)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(rgb, origin="lower",
             extent=[re_range[0], re_range[1], im_range[0], im_range[1]],
             aspect="auto", interpolation="bilinear")
    ax.axhline(0, color="w", lw=0.4, alpha=0.35)
    ax.set_xlabel(r"$\mathrm{Re}(\omega)$")
    ax.set_ylabel(r"$\mathrm{Im}(\omega)$")
    ax.set_title(
        rf"Retrato de fase de $\log$-match$(\omega)$, $m={m}$"
        "\nun cero: sectores de matiz + anillos convergiendo en un punto;"
        " el nº de sectores = índice topológico"
    )
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180)
        print(save)
    return fig, ax, M_np


if __name__ == "__main__":
    print(f"device: {DEVICE}"
         + (f"  ({torch.cuda.get_device_name(0)})"
            if DEVICE.type == "cuda" else ""))

    bg = Background(A=1.0, B=0.7, c=1.0)

    # Rejilla amplia y exploratoria primero.
    plot_hunt(bg, m=1, re_range=(0.05, 3.0), im_range=(-1.5, -0.02),
             n=900, save=str(OUT / "qnm_phase_portrait_m1.png"))

    # Zoom de ejemplo (ajusta segun lo que veas en la corrida de arriba).
    # Nota: re_range=(0.0, ...) ya no revienta la imagen (nanpercentile +
    # desplazamiento automatico del origen), pero omega=0 exacto sigue
    # siendo una singularidad FISICA (frecuencia cero), no solo numerica --
    # el punto en si nunca sera un QNM valido.
    plot_hunt(bg, m=1, re_range=(0.0, 0.35), im_range=(-0.08, 0.0),
             n=900, save=str(OUT / "qnm_zoom_m1.png"))