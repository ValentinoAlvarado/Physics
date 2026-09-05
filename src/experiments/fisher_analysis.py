"""
Analisis de Fisher completo + figuras para el poster.

    python experiments/fisher_analysis.py

Produce en results/:
    fisher_sensibilidad.png   donde vive la informacion (frecuencias utiles)
    fisher_elipse.png         degeneracion en el plano (eps, sigma)
    fisher_ruido.png          errores vs nivel de ruido, con umbral
    fisher_panel.png          los tres juntos, listo para el poster

y en consola el informe numerico con la interpretacion.

TIEMPO: cada punto de la curva |R|^2 es una integracion radial. Con
n_omega=36 y 5 configuraciones son 180 integraciones; cuenta unos minutos.
Baja n_omega a 20 para una pasada rapida.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.field.fisher import (derivative_convergence, fisher_superradiance,
                              identifiability_threshold, interpret,
                              noise_scan, report)
from src.physics.background import DrainingBathtub

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})

# ---------------------------------------------------------------- parametros
BG = DrainingBathtub(A=1.0, B=0.7, c=1.0)
M = 1
THETA0 = (0.30, 2.0)          # (eps, sigma) fiducial
N_OMEGA = 36
R_MAX = 300.0
REL_TOL = 0.30                # error relativo tolerable en eps
SIGMA_REF = 1e-2              # nivel de ruido de referencia


def main():
    print("=" * 64)
    print("CONVERGENCIA DE LAS DERIVADAS (control previo obligatorio)")
    print("=" * 64)
    print("Si el jacobiano no esta convergido, la Fisher no significa nada.")
    conv = derivative_convergence(BG, m=M, theta0=THETA0, r_max=R_MAX)
    for h, d in conv:
        print(f"  paso fraccionario {h:.3f} -> cambio relativo {d:.3e}")
    if conv[-1][1] > 5e-2:
        print("  ADVERTENCIA: el jacobiano NO esta convergido. "
              "Reduce frac_step o sube la precision del integrador.")
    else:
        print("  -> convergido")

    res = fisher_superradiance(BG, m=M, theta0=THETA0, n_omega=N_OMEGA,
                               sigma_n=SIGMA_REF, r_max=R_MAX)
    report(res, rel_tol=REL_TOL)

    # ------------------------------------------------------------ figura 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    w = res.omegas / (M * BG.Omega_H)
    ax[0].plot(w, res.signal, lw=1.8, color="#1f6fb2")
    ax[0].axhline(1.0, ls=":", c="k", lw=1)
    ax[0].set_xlabel(r"$\omega/(m\Omega_H)$")
    ax[0].set_ylabel(r"$|R(\omega)|^2$")
    ax[0].set_title("señal: amplificación superradiante")
    ax[1].plot(w, res.jac_u[:, 0], lw=1.8, label=r"$\partial/\partial(\epsilon/\epsilon_0)$")
    ax[1].plot(w, res.jac_u[:, 1], lw=1.8, label=r"$\partial/\partial(\sigma/\sigma_0)$")
    ax[1].axhline(0, ls=":", c="k", lw=1)
    ax[1].set_xlabel(r"$\omega/(m\Omega_H)$")
    ax[1].set_ylabel(r"$\partial|R|^2/\partial u$")
    ax[1].set_title("dónde vive la información")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fisher_sensibilidad.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "fisher_sensibilidad.png")

    # ------------------------------------------------------------ figura 2
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for ns, style, lab in ((1, "-", r"$1\sigma$"), (2, "--", r"$2\sigma$")):
        x, y = res.error_ellipse(n_sigma=ns)
        ax.plot(x, y, style, lw=1.7, label=lab)
    ax.plot(*res.theta0, "o", color="k", ms=6, label="fiducial")
    ev, evec = res.eig
    for k, (col, lab) in enumerate((("#d1495b", "peor determinada"),
                                    ("#2a9d8f", "mejor determinada"))):
        v = evec[:, k] * res.theta0
        v = v / np.linalg.norm(v) * 0.35 * np.linalg.norm(res.theta0)
        ax.annotate("", xy=res.theta0 + v, xytext=res.theta0,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.plot([], [], color=col, lw=2, label=lab)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title(rf"elipse de error, $\sigma_n={SIGMA_REF:g}$"
                 f"\ncond(F) = {res.condition:.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fisher_elipse.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "fisher_elipse.png")

    # ------------------------------------------------------------ figura 3
    sig = np.logspace(-3.5, -0.5, 40)
    base = res.with_noise(1.0).rel_errors
    err_e = base[0] * sig
    err_s = base[1] * sig
    thr = identifiability_threshold(res, REL_TOL)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(sig, 100 * err_e, lw=1.8, label=r"error rel. en $\epsilon$")
    ax.loglog(sig, 100 * err_s, lw=1.8, label=r"error rel. en $\sigma$")
    ax.axhline(100 * REL_TOL, ls="--", c="k", lw=1,
               label=f"tolerancia {100*REL_TOL:.0f}%")
    ax.axvline(thr, ls=":", c="#d1495b", lw=1.6,
               label=rf"umbral $\sigma_n={thr:.1e}$")
    ax.set_xlabel(r"ruido asumido $\sigma_n$ en $|R|^2$")
    ax.set_ylabel("error relativo (%)")
    ax.set_title("pérdida de identificabilidad con el ruido")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fisher_ruido.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "fisher_ruido.png")

    # ------------------------------------------------------------ panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(w, res.jac_u[:, 0], lw=1.8, label=r"$\epsilon$")
    axes[0].plot(w, res.jac_u[:, 1], lw=1.8, label=r"$\sigma$")
    axes[0].axhline(0, ls=":", c="k", lw=1)
    axes[0].set_xlabel(r"$\omega/(m\Omega_H)$")
    axes[0].set_ylabel(r"$\partial|R|^2/\partial u$")
    axes[0].set_title("(a) sensibilidad")
    axes[0].legend(fontsize=8)

    for ns, style in ((1, "-"), (2, "--")):
        x, y = res.error_ellipse(n_sigma=ns)
        axes[1].plot(x, y, style, lw=1.7)
    axes[1].plot(*res.theta0, "o", color="k", ms=6)
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_ylabel(r"$\sigma$")
    axes[1].set_title(f"(b) degeneración, cond(F)={res.condition:.1f}")

    axes[2].loglog(sig, 100 * err_e, lw=1.8, label=r"$\epsilon$")
    axes[2].loglog(sig, 100 * err_s, lw=1.8, label=r"$\sigma$")
    axes[2].axhline(100 * REL_TOL, ls="--", c="k", lw=1)
    axes[2].axvline(thr, ls=":", c="#d1495b", lw=1.6)
    axes[2].set_xlabel(r"$\sigma_n$")
    axes[2].set_ylabel("error relativo (%)")
    axes[2].set_title(rf"(c) umbral $\sigma_n={thr:.1e}$")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fisher_panel.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "fisher_panel.png")

    # ------------------------------------------------------ texto exportado
    txt = OUT / "fisher_interpretacion.txt"
    txt.write_text(interpret(res, REL_TOL), encoding="utf-8")
    print("guardado:", txt)


if __name__ == "__main__":
    main()
