"""Leyenda del campo complejo: rueda de fase (angulo=arg, radio=|Psi|)."""
import numpy as np
from matplotlib.colors import hsv_to_rgb


def phase_wheel_rgb(n=320, band_spacing=np.pi/8, band_strength=0.30,
                    band_sharpness=6.0, gamma=0.6, saturation=0.90,
                    show_bands=True):
    """RGB (n,n,4) con alfa: rueda de dominio.  angulo=arg, radio=|Psi| norm."""
    y, x = np.mgrid[-1:1:n*1j, -1:1:n*1j]
    r = np.hypot(x, y)
    ph = np.arctan2(y, x)
    inside = r <= 1.0

    hue = (ph + np.pi) / (2*np.pi)
    val = np.clip(r, 0, 1) ** gamma
    if show_bands:
        u = np.mod(ph, band_spacing) / band_spacing
        band = (0.5 + 0.5*np.cos(2*np.pi*u)) ** band_sharpness
        val = val * (1.0 - band_strength + band_strength*band)
    val = np.clip(val * (0.12 + 0.88*np.clip(r, 0, 1)), 0, 1)

    hsv = np.stack([hue, np.full_like(hue, saturation), val], axis=-1)
    rgb = hsv_to_rgb(np.clip(hsv, 0, 1))
    rgba = np.dstack([rgb, inside.astype(float)])
    return rgba


def add_field_legend(fig, rect=(0.845, 0.055, 0.115, 0.155),
                     band_spacing=np.pi/8, show_bands=True,
                     text_color="#c8d4e4", label_size=7.5):
    """Anade la rueda de fase como inset. Devuelve el axes creado."""
    ax = fig.add_axes(rect)
    ax.imshow(phase_wheel_rgb(band_spacing=band_spacing,
                              show_bands=show_bands),
              extent=[-1, 1, -1, 1], interpolation="bilinear", zorder=2)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.95, 1.95)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    for ang, lab, ha, va in ((0, r"$0$", "left", "center"),
                             (np.pi/2, r"$\pi/2$", "center", "bottom"),
                             (np.pi, r"$\pm\pi$", "right", "center"),
                             (-np.pi/2, r"$-\pi/2$", "center", "top")):
        ax.text(1.13*np.cos(ang), 1.13*np.sin(ang), lab,
                color=text_color, fontsize=label_size, ha=ha, va=va)
    ax.annotate("", xy=(0.92, 0), xytext=(0.05, 0),
                arrowprops=dict(arrowstyle="->", color=text_color, lw=0.9))
    ax.text(0.48, -0.20, r"$|\Psi|$", color=text_color,
            fontsize=label_size, ha="center")
    ax.text(0, 1.72, r"matiz $=\arg\Psi$", color=text_color,
            fontsize=label_size, ha="center")
    ax.text(0, -1.78, r"radio $=|\Psi|$ (norm.)", color=text_color,
            fontsize=label_size, ha="center")
    return ax


def add_phase_colorbar(fig, rect=(0.905, 0.30, 0.018, 0.42),
                       band_spacing=np.pi/8, show_bands=True,
                       band_strength=0.30, band_sharpness=6.0,
                       saturation=0.90, text_color="#c8d4e4",
                       label_size=8, n=512):
    """Barra vertical de fase: gradiente ciclico de -pi a pi.

    La fase es CICLICA: el color de arriba y el de abajo son el mismo
    (ambos +-pi). Eso es correcto y hay que leerlo asi, no como una escala
    lineal con extremos distintos.

    Muestra el matiz a luminancia plena (|Psi| maximo). La luminancia del
    campo codifica |Psi| aparte; esta barra solo mapea el matiz.
    """
    ax = fig.add_axes(rect)
    ph = np.linspace(-np.pi, np.pi, n)
    hue = (ph + np.pi) / (2*np.pi)
    val = np.ones_like(hue)
    if show_bands:
        u = np.mod(ph, band_spacing) / band_spacing
        band = (0.5 + 0.5*np.cos(2*np.pi*u)) ** band_sharpness
        val = val * (1.0 - band_strength + band_strength*band)
    hsv = np.stack([hue, np.full_like(hue, saturation), val], axis=-1)
    rgb = hsv_to_rgb(np.clip(hsv, 0, 1))[:, None, :]
    rgb = np.repeat(rgb, 8, axis=1)

    ax.imshow(rgb, origin="lower", aspect="auto",
              extent=[0, 1, -np.pi, np.pi], interpolation="bilinear")
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$",
                        r"$\pi/2$", r"$\pi$"])
    ax.tick_params(colors=text_color, labelsize=label_size,
                   length=2.5, pad=2)
    for sp in ax.spines.values():
        sp.set_color(text_color)
        sp.set_linewidth(0.6)
    ax.set_title(r"$\arg\Psi$", color=text_color, fontsize=label_size+1,
                 pad=6)
    return ax


def add_modulus_colorbar(fig, rect=(0.955, 0.30, 0.014, 0.42),
                         gamma=0.6, text_color="#c8d4e4", label_size=8,
                         n=256):
    """Barra vertical de |Psi| normalizado: rampa de luminancia en gris.

    Complementa a add_phase_colorbar. La normalizacion es por PERCENTILES
    (2-98) de la propia malla, de ahi el 'norm.' en la etiqueta: no es una
    escala absoluta.
    """
    ax = fig.add_axes(rect)
    v = np.linspace(0, 1, n) ** gamma
    v = v * (0.12 + 0.88*np.linspace(0, 1, n))
    rgb = np.repeat(np.clip(v, 0, 1)[:, None, None], 3, axis=2)
    rgb = np.repeat(rgb, 6, axis=1)
    ax.imshow(rgb, origin="lower", aspect="auto",
              extent=[0, 1, 0, 1], interpolation="bilinear")
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "", "max"])
    ax.tick_params(colors=text_color, labelsize=label_size,
                   length=2.5, pad=2)
    for sp in ax.spines.values():
        sp.set_color(text_color)
        sp.set_linewidth(0.6)
    ax.set_title(r"$|\Psi|$", color=text_color, fontsize=label_size+1, pad=6)
    return ax