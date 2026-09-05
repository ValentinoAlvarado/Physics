"""
Superficie 3D de verosimilitud chi^2(eps, sigma) vs. la elipse de Fisher.

    python experiments/chi2_surface_3d.py

Produce en results/:
    chi2_superficie3d.png   valle de degeneracion real, en 3D
    chi2_fisher_vs_real.png contornos reales vs. aproximacion cuadratica
    chi2_valle.png          perfil a lo largo del valle

QUE RESPONDE ESTA FIGURA
========================
La elipse de Fisher es la aproximacion cuadratica de chi^2 en torno al punto
fiducial. En los resultados obtenidos la elipse de 1-sigma se extiende a
sigma < 0, que es NO FISICO (sigma es una anchura). Eso indica que la
linealizacion deja de valer antes de llegar al borde de la elipse.

La superficie real muestra:
  (a) si el valle de degeneracion es recto (Fisher vale) o curvo (no vale);
  (b) si hay mas de un minimo, cosa que Fisher no puede detectar;
  (c) hasta que distancia del fiducial la elipse es una descripcion honesta.

VALIDACION OBLIGATORIA: el script comprueba primero el kernel contra scipy y
el residuo de conservacion de flujo. Si eso falla, la superficie no vale nada.

COSTE: n_eps x n_sig x n_omega integraciones radiales en un lanzamiento.
Con 41x41x24 son ~40000; en GPU son segundos. Baja n_eps/n_sig si hace falta.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.field import chi2_cuda
from src.field.fisher import fisher_superradiance
from src.physics.background import DrainingBathtub

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

BG = DrainingBathtub(A=1.0, B=0.7, c=1.0)
M = 1
THETA0 = (0.30, 2.0)
SIGMA_N = 1e-2
EPS_RANGE = (-0.10, 0.70)
SIG_RANGE = (0.40, 4.00)
N_EPS, N_SIG, N_OMEGA = 41, 41, 24
R_EXTRACT = 300.0


def main():
    if not chi2_cuda.HAVE_CUPY:
        raise SystemExit("Falta CuPy:  pip install cupy-cuda12x")

    print("=" * 62)
    print("VALIDACION DEL KERNEL (obligatoria antes de la superficie)")
    print("=" * 62)
    ok = chi2_cuda.validate_against_scipy(BG, m=M, eps=THETA0[0],
                                          sig=THETA0[1], n_omega=8,
                                          r_extract=R_EXTRACT)
    if not ok:
        raise SystemExit("El kernel NO reproduce scipy. No sigas: la "
                         "superficie chi^2 seria basura.")

    print("\ncalculando superficie chi^2 "
          f"({N_SIG}x{N_EPS}x{N_OMEGA} integraciones)...")
    surf = chi2_cuda.chi2_surface(
        BG, theta0=THETA0, m=M, eps_range=EPS_RANGE, sig_range=SIG_RANGE,
        n_eps=N_EPS, n_sig=N_SIG, n_omega=N_OMEGA, sigma_n=SIGMA_N,
        r_extract=R_EXTRACT)
    print(f"  residuo de flujo max en toda la malla: {surf.flux_max:.2e}")
    print(f"  chi^2 min = {np.nanmin(surf.chi2):.3e}  "
          f"(debe ser ~0 en el fiducial)")

    E, S = np.meshgrid(surf.eps_grid, surf.sig_grid)
    Z = surf.chi2
    # escala logaritmica: el rango dinamico de chi^2 es enorme
    Zl = np.log10(np.maximum(Z, 1e-6))

    # ------------------------------------------------------- Fisher de ref.
    fis = fisher_superradiance(BG, m=M, theta0=THETA0, n_omega=N_OMEGA,
                               sigma_n=SIGMA_N, r_max=R_EXTRACT,
                               verbose=False)
    Finv = fis.cov_u
    Fm = fis.F
    du_e = (E - THETA0[0]) / THETA0[0]
    du_s = (S - THETA0[1]) / THETA0[1]
    # chi^2 cuadratico = du^T F du
    Zq = (Fm[0, 0]*du_e**2 + 2*Fm[0, 1]*du_e*du_s + Fm[1, 1]*du_s**2)
    Zql = np.log10(np.maximum(Zq, 1e-6))

    # ============================================================ figura 3D
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(E, S, Zl, cmap=cm.viridis, linewidth=0,
                    antialiased=True, alpha=0.92, rstride=1, cstride=1)
    ax.contour(E, S, Zl, levels=12, zdir="z",
               offset=float(np.nanmin(Zl)), cmap=cm.viridis, linewidths=0.6)
    ax.scatter([THETA0[0]], [THETA0[1]], [float(np.nanmin(Zl))],
               color="red", s=40, depthshade=False)
    ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel(r"$\sigma$")
    ax.set_zlabel(r"$\log_{10}\chi^2$")
    ax.set_title("superficie de verosimilitud REAL")
    ax.view_init(elev=38, azim=-125)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(E, S, Zql, cmap=cm.magma, linewidth=0,
                     antialiased=True, alpha=0.92, rstride=1, cstride=1)
    ax2.scatter([THETA0[0]], [THETA0[1]], [float(np.nanmin(Zql))],
                color="red", s=40, depthshade=False)
    ax2.set_xlabel(r"$\epsilon$"); ax2.set_ylabel(r"$\sigma$")
    ax2.set_zlabel(r"$\log_{10}\chi^2$")
    ax2.set_title("aproximación cuadrática (Fisher)")
    ax2.view_init(elev=38, azim=-125)
    fig.suptitle(r"$\chi^2(\epsilon,\sigma)$ — canal superradiante, "
                 rf"$\sigma_n={SIGMA_N:g}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "chi2_superficie3d.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "chi2_superficie3d.png")

    # ================================================== contornos comparados
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    # niveles de confianza para 2 parametros: Delta chi^2 = 2.30, 6.17, 11.8
    lv = [2.30, 6.17, 11.83]
    cs1 = ax.contour(E, S, Z, levels=lv, colors="#1f6fb2", linewidths=1.9)
    cs2 = ax.contour(E, S, Zq, levels=lv, colors="#d1495b",
                     linewidths=1.6, linestyles="--")
    ax.clabel(cs1, fmt={2.30: "68%", 6.17: "95%", 11.83: "99.7%"},
              fontsize=7)
    ax.plot(*THETA0, "o", color="k", ms=7)
    ax.axhline(0, color="k", lw=1.2, ls=":")
    ax.text(EPS_RANGE[0] + 0.02, 0.08, r"$\sigma=0$ (no físico)",
            fontsize=8, color="k")
    ax.plot([], [], color="#1f6fb2", lw=1.9, label="χ² real")
    ax.plot([], [], color="#d1495b", lw=1.6, ls="--", label="Fisher")
    ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel(r"$\sigma$")
    ax.set_title("¿hasta dónde vale la elipse de Fisher?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "chi2_fisher_vs_real.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "chi2_fisher_vs_real.png")

    # ========================================================= perfil valle
    # direccion del autovalor menor de F = valle de degeneracion
    ev, evec = fis.eig
    d = evec[:, 0] * fis.theta0          # a unidades fisicas
    d = d / np.linalg.norm(d)
    t = np.linspace(-1.6, 1.6, 60)
    pe = THETA0[0] + t * d[0]
    ps = THETA0[1] + t * d[1]
    inside = ((pe >= EPS_RANGE[0]) & (pe <= EPS_RANGE[1])
              & (ps >= SIG_RANGE[0]) & (ps <= SIG_RANGE[1]))
    # muestreo bilineal de la superficie a lo largo del valle
    ie = np.interp(pe[inside], surf.eps_grid, np.arange(N_EPS))
    isx = np.interp(ps[inside], surf.sig_grid, np.arange(N_SIG))
    prof = Z[np.clip(isx.astype(int), 0, N_SIG-1),
             np.clip(ie.astype(int), 0, N_EPS-1)]
    profq = Zq[np.clip(isx.astype(int), 0, N_SIG-1),
               np.clip(ie.astype(int), 0, N_EPS-1)]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t[inside], prof, lw=1.9, color="#1f6fb2", label="χ² real")
    ax.plot(t[inside], profq, "--", lw=1.6, color="#d1495b", label="Fisher")
    for L, lab in ((2.30, "68%"), (6.17, "95%")):
        ax.axhline(L, ls=":", c="k", lw=0.9)
        ax.text(t[inside].min(), L*1.1, lab, fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("distancia a lo largo del valle de degeneración")
    ax.set_ylabel(r"$\chi^2$")
    ax.set_title("perfil en la dirección peor determinada")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "chi2_valle.png", bbox_inches="tight")
    plt.close(fig)
    print("guardado:", OUT / "chi2_valle.png")

    # ============================================== lectura automatica
    print("\n" + "=" * 62)
    print("LECTURA")
    print("=" * 62)
    dev = np.abs(prof - profq) / np.maximum(prof, 1e-12)
    j = np.argmax(dev > 0.5) if np.any(dev > 0.5) else None
    if j:
        print(f"La aproximación de Fisher se desvía >50% del chi^2 real a "
              f"partir de |t| ~ {abs(t[inside][j]):.2f} en (eps,sigma).")
        print("  -> la elipse SOLO es honesta dentro de esa distancia.")
    else:
        print("Fisher sigue al chi^2 real en toda la región explorada:")
        print("  -> la elipse es una descripción fiable aquí.")
    n_min = int(np.sum(Z < np.nanmin(Z) + 2.30))
    print(f"Puntos de la malla dentro del contorno del 68%: {n_min}")
    print("Si la región del 68% toca sigma <= 0, la anchura NO está acotada")
    print("por abajo con este canal a ese nivel de ruido.")


if __name__ == "__main__":
    main()
