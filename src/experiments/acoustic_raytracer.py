"""
Trazador de rayos acusticos 3D — "cuerda negra" acustica (vortice cilindrico)
con fondo de estrellas, camara controlable y parametros variables.

=============================================================================
FISICA — que es esto exactamente, y que NO es
=============================================================================
El campo de fondo es el mismo draining bathtub de tu proyecto, extendido como
vortice CILINDRICO invariante en z:

    v_r(r)   = -(A/r) [1 + h(r)]        r = sqrt(x^2 + y^2)  (radio cilindrico)
    v_phi(r) =  B/r
    v_z      =  0

La continuidad para un sumidero cilindrico, rho*v_r*(2 pi r L) = const, vuelve
a forzar v_r ~ 1/r con rho constante: son los MISMOS A, B, el mismo horizonte
r_h = |A|/c y la misma ergosfera r_e = sqrt(A^2+B^2)/c que en el caso 2+1D.

El horizonte es un CILINDRO, no una esfera: esto es el analogo acustico de una
"cuerda negra" (black string), no de Schwarzschild ni de Kerr.

ADVERTENCIA IMPORTANTE PARA EL POSTER
-------------------------------------
Al pasar a 3D los rayos adquieren k_z, que se conserva (el flujo no depende de
z). Para el campo de ondas, ese k_z actua como un TERMINO DE MASA EFECTIVA en
el problema 2+1D reducido. Los QNM que analizas en el poster corresponden a
k_z = 0. Por tanto esta visualizacion es legitima como retrato de la GEOMETRIA
EFECTIVA (mismo fondo, mismo horizonte, mismo arrastre de marco), pero NO
representa el mismo sector de modos que tu analisis espectral. Presentala como
tal.

Ecuaciones de rayos (exactas para c homogenea)
----------------------------------------------
Relacion de dispersion:      omega = v.k + c|k|   =: H(x,k)
Ecuaciones de Hamilton:      dx/dt = dH/dk = v + c*khat
                             dk_i/dt = -dH/dx_i = -k_j d(v_j)/d(x_i)
omega se conserva a lo largo del rayo -> test de validacion incorporado.
k_z se conserva exactamente (v no depende de z, v_z = 0).

Trazado INVERSO: para saber que ve la camara, se integra hacia atras en el
tiempo desde cada pixel. Con s = -t:
                             dx/ds = -(v + c*khat)
                             dk_i/ds = +k_j d(v_j)/d(x_i)

=============================================================================
USO
=============================================================================
1) Corre primero self_test() -- valida el jacobiano analitico contra autograd
   y la conservacion de omega. Es barato y detecta de inmediato errores de
   derivadas.
2) Ajusta CONFIG y corre render_image().
3) Para barrer parametros, usa sweep() (genera una serie de imagenes).

Todo el trazado es un batch de tensores: un rayo por elemento, sin bucles
Python por pixel. Corre en CUDA automaticamente si esta disponible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32          # float32 basta para imagen; usa float64 en tests


# =============================================================== parametros
@dataclass
class Config:
    # --- fisica del fondo ---
    A: float = 1.0             # sumidero  [L^2/T]
    B: float = 0.7             # circulacion [L^2/T]
    c: float = 1.0             # velocidad del sonido
    eps: float = 0.0           # amplitud de la deformacion h(r) (0 = sin deformar)
    sig: float = 2.0           # ancho de la deformacion

    # --- camara ---
    cam_pos: tuple = (0.0, -18.0, 4.0)   # posicion del observador
    look_at: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 0.0, 1.0)
    fov_deg: float = 42.0                # campo de vision vertical
    width: int = 1280
    height: int = 720

    # --- integracion ---
    dt: float = 0.02
    n_steps: int = 4000
    r_sky: float = 60.0        # radio donde se considera que el rayo escapo
    capture_pad: float = 1.002 # captura si r < r_h * capture_pad
    z_sky: float = 60.0        # |z| donde tambien se considera escape

    # --- fondo estelar ---
    n_stars: int = 9000
    star_seed: int = 7
    sky_tex_h: int = 1024      # resolucion de la textura equirectangular
    grid_overlay: bool = True  # rejilla de coordenadas para ver el lensing
    grid_strength: float = 0.35


# ================================================== campo de velocidad y grad
def velocity(x, y, cfg: Config):
    """v = (vx, vy, vz=0). Solo depende de x,y (invariante en z)."""
    r2 = x * x + y * y
    r = torch.sqrt(r2)
    A, B = cfg.A, cfg.B

    if cfg.eps != 0.0:
        r_h = abs(cfg.A) / cfg.c
        d = r - r_h
        E = torch.exp(-cfg.sig * d * d)
        h = cfg.eps * (d / r_h) ** 2 * E
    else:
        h = torch.zeros_like(r)

    vr = -(A / r) * (1.0 + h)
    vph = B / r
    ux, uy = x / r, y / r          # rhat
    px, py = -y / r, x / r         # phihat
    vx = vr * ux + vph * px
    vy = vr * uy + vph * py
    return vx, vy


def velocity_grad(x, y, cfg: Config):
    """Jacobiano analitico dv[i][j] = d v_i / d x_j, para i,j en {x,y}.

    Derivado a mano; validado contra autograd en self_test().
    Para h = 0 se reduce a:
        vx = (-A x - B y)/r^2 ,  vy = (-A y + B x)/r^2
    """
    r2 = x * x + y * y
    r = torch.sqrt(r2)
    A, B = cfg.A, cfg.B

    if cfg.eps != 0.0:
        r_h = abs(cfg.A) / cfg.c
        d = r - r_h
        E = torch.exp(-cfg.sig * d * d)
        h = cfg.eps * (d / r_h) ** 2 * E
        dh_dr = cfg.eps / r_h**2 * (2 * d - 2 * cfg.sig * d**3) * E
    else:
        h = torch.zeros_like(r)
        dh_dr = torch.zeros_like(r)

    vr = -(A / r) * (1.0 + h)
    dvr_dr = (A / r2) * (1.0 + h) - (A / r) * dh_dr
    vph = B / r
    dvph_dr = -B / r2

    ux, uy = x / r, y / r
    px, py = -y / r, x / r

    # d rhat_i / d x_j = (delta_ij - u_i u_j) / r
    # phihat = R90 . rhat  =>  d phihat_0/dx_j = -d rhat_1/dx_j
    #                          d phihat_1/dx_j = +d rhat_0/dx_j
    dv = [[None, None], [None, None]]
    for j, uj in enumerate((ux, uy)):
        d0j = 1.0 if j == 0 else 0.0
        d1j = 1.0 if j == 1 else 0.0
        dr0j = (d0j - ux * uj) / r
        dr1j = (d1j - uy * uj) / r
        dp0j = -dr1j
        dp1j = dr0j
        dv[0][j] = dvr_dr * uj * ux + vr * dr0j + dvph_dr * uj * px + vph * dp0j
        dv[1][j] = dvr_dr * uj * uy + vr * dr1j + dvph_dr * uj * py + vph * dp1j
    return dv


# ========================================================= ecuaciones de rayo
def ray_rhs(state, cfg: Config, backward: bool):
    """state = (x, y, z, kx, ky, kz). Devuelve d(state)/ds.

    backward=True integra hacia atras en el tiempo (para trazado inverso
    desde la camara).
    """
    x, y, z, kx, ky, kz = state
    vx, vy = velocity(x, y, cfg)
    dv = velocity_grad(x, y, cfg)

    kn = torch.sqrt(kx * kx + ky * ky + kz * kz) + 1e-30
    dx = vx + cfg.c * kx / kn
    dy = vy + cfg.c * ky / kn
    dz = cfg.c * kz / kn                        # v_z = 0

    # dk_i/dt = -k_j d v_j / d x_i
    dkx = -(dv[0][0] * kx + dv[1][0] * ky)
    dky = -(dv[0][1] * kx + dv[1][1] * ky)
    dkz = torch.zeros_like(kz)                  # k_z se conserva exactamente

    out = torch.stack([dx, dy, dz, dkx, dky, dkz])
    return -out if backward else out


def omega_of(state, cfg: Config):
    """omega = v.k + c|k|. Debe conservarse a lo largo del rayo."""
    x, y, _z, kx, ky, kz = state
    vx, vy = velocity(x, y, cfg)
    return vx * kx + vy * ky + cfg.c * torch.sqrt(kx * kx + ky * ky + kz * kz)


def trace_rays(state, cfg: Config, backward=True):
    """RK4 de paso fijo, vectorizado. Devuelve (estado_final, capturado, escapado)."""
    r_h = abs(cfg.A) / cfg.c
    r_cap = r_h * cfg.capture_pad
    s = state.clone()
    N = s.shape[1]
    captured = torch.zeros(N, dtype=torch.bool, device=s.device)
    escaped = torch.zeros(N, dtype=torch.bool, device=s.device)

    for _ in range(cfg.n_steps):
        k1 = ray_rhs(s, cfg, backward)
        k2 = ray_rhs(s + 0.5 * cfg.dt * k1, cfg, backward)
        k3 = ray_rhs(s + 0.5 * cfg.dt * k2, cfg, backward)
        k4 = ray_rhs(s + cfg.dt * k3, cfg, backward)
        s_new = s + (cfg.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        r_cyl = torch.sqrt(s_new[0] ** 2 + s_new[1] ** 2)
        far = torch.sqrt(s_new[0] ** 2 + s_new[1] ** 2 + s_new[2] ** 2)
        active = ~(captured | escaped)
        captured |= active & (r_cyl < r_cap)
        escaped |= active & ((far > cfg.r_sky) | (s_new[2].abs() > cfg.z_sky))

        done = captured | escaped
        s = torch.where(done.unsqueeze(0), s, s_new)   # congela los terminados
        if bool(done.all()):
            break
    return s, captured, escaped


# ============================================================== fondo estelar
def make_sky_texture(cfg: Config):
    """Textura equirectangular (H, 2H, 3) con estrellas + rejilla opcional."""
    rng = np.random.default_rng(cfg.star_seed)
    H = cfg.sky_tex_h
    W = 2 * H
    tex = np.zeros((H, W, 3), dtype=np.float32)

    # estrellas: distribucion uniforme en la esfera -> uniforme en (phi, cos th)
    n = cfg.n_stars
    u = rng.uniform(-1.0, 1.0, n)          # cos(theta)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    th = np.arccos(u)
    py = ((th / np.pi) * (H - 1)).astype(int)
    px = ((ph / (2 * np.pi)) * (W - 1)).astype(int)

    # magnitudes: ley de potencias -> pocas brillantes, muchas debiles
    mag = rng.pareto(1.7, n) + 0.12
    mag = np.clip(mag / mag.max(), 0.02, 1.0)
    # tinte estelar (azuladas / amarillentas)
    temp = rng.uniform(0.0, 1.0, n)
    col = np.stack([0.72 + 0.28 * temp,
                    0.80 + 0.14 * np.abs(temp - 0.5),
                    1.0 - 0.25 * temp], axis=1)

    for k in range(n):
        tex[py[k], px[k]] += (mag[k] * col[k]).astype(np.float32)

    # pequeno halo gaussiano para que no sean pixeles duros
    try:
        from scipy.ndimage import gaussian_filter
        tex = gaussian_filter(tex, sigma=(0.7, 0.7, 0))
        tex += 0.55 * gaussian_filter(tex, sigma=(2.2, 2.2, 0))
    except Exception:
        pass  # sin scipy quedan estrellas puntuales; no es critico

    if cfg.grid_overlay:
        th_lines = (np.arange(H) * np.pi / H)
        ph_lines = (np.arange(W) * 2 * np.pi / W)
        g_th = (np.abs(np.sin(th_lines[:, None] * 12.0)) > 0.995)
        g_ph = (np.abs(np.sin(ph_lines[None, :] * 12.0)) > 0.995)
        grid = (g_th | g_ph).astype(np.float32) * cfg.grid_strength
        tex[..., 0] += grid * 0.25
        tex[..., 1] += grid * 0.55
        tex[..., 2] += grid * 0.75

    return np.clip(tex, 0.0, 4.0)


def sample_sky(dirs, tex_t):
    """Muestrea la textura equirectangular en las direcciones dadas (3,N)."""
    dx, dy, dz = dirs
    n = torch.sqrt(dx * dx + dy * dy + dz * dz) + 1e-30
    dx, dy, dz = dx / n, dy / n, dz / n
    theta = torch.acos(torch.clamp(dz, -1.0, 1.0))
    phi = torch.atan2(dy, dx) % (2 * np.pi)

    H, W, _ = tex_t.shape
    fy = (theta / np.pi) * (H - 1)
    fx = (phi / (2 * np.pi)) * (W - 1)
    y0 = torch.clamp(fy.floor().long(), 0, H - 1)
    x0 = torch.clamp(fx.floor().long(), 0, W - 1)
    y1 = torch.clamp(y0 + 1, 0, H - 1)
    x1 = (x0 + 1) % W
    wy = (fy - y0.to(fy.dtype)).unsqueeze(-1)
    wx = (fx - x0.to(fx.dtype)).unsqueeze(-1)

    c00 = tex_t[y0, x0]
    c01 = tex_t[y0, x1]
    c10 = tex_t[y1, x0]
    c11 = tex_t[y1, x1]
    top = c00 * (1 - wx) + c01 * wx
    bot = c10 * (1 - wx) + c11 * wx
    return top * (1 - wy) + bot * wy


# ================================================================== camara
def camera_rays(cfg: Config):
    """Genera (origen, direccion) por pixel. Devuelve dirs (3, N)."""
    cam = torch.tensor(cfg.cam_pos, dtype=DTYPE, device=DEVICE)
    tgt = torch.tensor(cfg.look_at, dtype=DTYPE, device=DEVICE)
    up = torch.tensor(cfg.up, dtype=DTYPE, device=DEVICE)

    fwd = tgt - cam
    fwd = fwd / torch.norm(fwd)
    right = torch.linalg.cross(fwd, up)
    right = right / torch.norm(right)
    true_up = torch.linalg.cross(right, fwd)

    aspect = cfg.width / cfg.height
    half_h = np.tan(np.radians(cfg.fov_deg) / 2.0)
    half_w = half_h * aspect

    ys = torch.linspace(half_h, -half_h, cfg.height, dtype=DTYPE, device=DEVICE)
    xs = torch.linspace(-half_w, half_w, cfg.width, dtype=DTYPE, device=DEVICE)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    d = (fwd.view(3, 1, 1)
         + right.view(3, 1, 1) * gx.unsqueeze(0)
         + true_up.view(3, 1, 1) * gy.unsqueeze(0))
    d = d / torch.norm(d, dim=0, keepdim=True)
    return cam, d.reshape(3, -1)


# ================================================================== render
def render_image(cfg: Config, save_name="acoustic_blackstring.png"):
    """Traza la imagen completa y la guarda. Devuelve el array RGB (H,W,3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam, dirs = camera_rays(cfg)
    N = dirs.shape[1]

    # trazado INVERSO: el rayo que llega a la camara desde la direccion d
    # se integra hacia atras en el tiempo partiendo de la camara con k ~ d.
    x0 = torch.full((N,), float(cam[0]), dtype=DTYPE, device=DEVICE)
    y0 = torch.full((N,), float(cam[1]), dtype=DTYPE, device=DEVICE)
    z0 = torch.full((N,), float(cam[2]), dtype=DTYPE, device=DEVICE)
    state = torch.stack([x0, y0, z0, dirs[0], dirs[1], dirs[2]])

    with torch.no_grad():
        final, captured, escaped = trace_rays(state, cfg, backward=True)

    tex = make_sky_texture(cfg)
    tex_t = torch.tensor(tex, dtype=DTYPE, device=DEVICE)

    # color del cielo segun la direccion final del rayo
    col = sample_sky(final[3:6], tex_t)          # (N,3)
    col[captured] = 0.0                          # sombra acustica: negro
    # rayos que no terminaron (ni captura ni escape) -> oscuros, marcados
    stuck = ~(captured | escaped)
    col[stuck] = torch.tensor([0.05, 0.0, 0.06], dtype=DTYPE, device=DEVICE)

    img = col.reshape(cfg.height, cfg.width, 3).detach().cpu().numpy()
    img = img / (1.0 + img)                      # tone mapping
    img = np.clip(img * 1.9, 0, 1) ** (1 / 2.2)  # gamma

    n_cap = int(captured.sum())
    n_stuck = int(stuck.sum())
    print(f"capturados: {n_cap}/{N} ({100*n_cap/N:.1f}%)   "
          f"sin terminar: {n_stuck} (si es alto, sube n_steps)")

    fig = plt.figure(figsize=(cfg.width / 140, cfg.height / 140),
                     facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="bilinear")
    ax.set_axis_off()
    r_h = abs(cfg.A) / cfg.c
    r_e = np.hypot(cfg.A, cfg.B) / cfg.c
    ax.text(0.012, 0.975,
            f"cuerda negra acústica  |  A={cfg.A}  B={cfg.B}  "
            f"$r_h$={r_h:.2f}  $r_e$={r_e:.2f}",
            transform=ax.transAxes, color="#9fb6d0", fontsize=9, va="top")
    path = OUT / save_name
    fig.savefig(path, dpi=140, facecolor="black")
    plt.close(fig)
    print(path)
    return img


def sweep(param, values, base: Config = None, prefix="sweep"):
    """Barre un parametro y guarda una imagen por valor. Ej:
        sweep("B", [0.0, 0.3, 0.6, 0.9])
        sweep("eps", [-0.4, 0.0, 0.4])
    """
    base = base or Config()
    for v in values:
        cfg = Config(**{**base.__dict__, param: v})
        render_image(cfg, save_name=f"{prefix}_{param}_{v}.png")


# =================================================================== tests
def self_test():
    """Valida (a) el jacobiano analitico contra autograd y (b) la
    conservacion de omega. Corre esto ANTES de renderizar."""
    torch.set_default_dtype(torch.float64)
    cfg = Config(A=1.0, B=0.7, c=1.0, eps=0.3, sig=2.0)

    # --- (a) jacobiano vs autograd ---
    xs = torch.tensor([2.0, -3.5, 1.2, 5.0], dtype=torch.float64,
                      requires_grad=True)
    ys = torch.tensor([1.0, 2.5, -4.0, 0.3], dtype=torch.float64,
                      requires_grad=True)
    vx, vy = velocity(xs, ys, cfg)
    gx_x = torch.autograd.grad(vx.sum(), xs, retain_graph=True)[0]
    gx_y = torch.autograd.grad(vx.sum(), ys, retain_graph=True)[0]
    gy_x = torch.autograd.grad(vy.sum(), xs, retain_graph=True)[0]
    gy_y = torch.autograd.grad(vy.sum(), ys)[0]

    dv = velocity_grad(xs.detach(), ys.detach(), cfg)
    errs = [
        (dv[0][0] - gx_x).abs().max().item(),
        (dv[0][1] - gx_y).abs().max().item(),
        (dv[1][0] - gy_x).abs().max().item(),
        (dv[1][1] - gy_y).abs().max().item(),
    ]
    print("jacobiano analitico vs autograd, error max por componente:")
    print(f"  dvx/dx {errs[0]:.3e}   dvx/dy {errs[1]:.3e}")
    print(f"  dvy/dx {errs[2]:.3e}   dvy/dy {errs[3]:.3e}")
    ok_jac = max(errs) < 1e-9
    print("  ->", "OK" if ok_jac else "FALLA: las derivadas no coinciden")

    # --- (b) conservacion de omega ---
    cfg2 = Config(A=1.0, B=0.7, c=1.0, eps=0.0,
                  dt=0.004, n_steps=8000, r_sky=40.0)
    n = 7
    b = torch.linspace(-6, 6, n, dtype=torch.float64)
    st = torch.stack([
        torch.full((n,), 15.0, dtype=torch.float64), b,
        torch.zeros(n, dtype=torch.float64),
        torch.full((n,), -1.0, dtype=torch.float64),
        torch.zeros(n, dtype=torch.float64),
        torch.zeros(n, dtype=torch.float64),
    ])
    w0 = omega_of(st, cfg2)
    sf, cap, esc = trace_rays(st, cfg2, backward=False)
    wf = omega_of(sf, cfg2)
    drift = ((wf - w0).abs() / w0.abs()).max().item()
    print(f"conservacion de omega: deriva relativa maxima = {drift:.3e}")
    print("  ->", "OK" if drift < 1e-8 else "FALLA: revisa dt / las ecuaciones")
    print(f"capturados por parametro de impacto b: "
          f"{[bool(v) for v in cap.tolist()]}")
    print("  (la ASIMETRIA en b es fisica: arrastre de marco. "
          "Con B=0 debe ser simetrica.)")

    torch.set_default_dtype(torch.float32)
    return ok_jac and drift < 1e-8


if __name__ == "__main__":
    print(f"device: {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})"
             if DEVICE.type == "cuda" else ""))
    print("\n=== self_test ===")
    self_test()
    print("\n=== render ===")
    render_image(Config())
