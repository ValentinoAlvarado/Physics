"""
VISOR UNIFICADO — lente acustica + campo complejo Psi en una sola escena.

Fusiona los dos programas que antes estaban separados:
  - el ray tracer CUDA (fondo de estrellas lenteado, sombra del horizonte)
  - el campo complejo Psi (matiz = arg Psi, luminancia = |Psi|)

    pip install cupy-cuda12x
    python viewer_unified.py

=============================================================================
COMO SE UNIFICAN (no es pegar codigo: hay una decision geometrica)
=============================================================================
El ray tracer modela un vortice CILINDRICO invariante en z, que NO tiene
superficie libre. El campo Psi vive sobre el plano 2D del vortice. Para verlos
juntos se introduce explicitamente un DISCO en z = 0, r_h < r < r_disk, que
representa la superficie del fluido donde vive el campo.

Cada rayo de camara, trazado hacia atras en el tiempo, termina en uno de
cuatro estados:

    capturado por el horizonte      -> negro (sombra acustica)
    cruza el disco                  -> color del domain coloring de Psi(x,y)
                                       en el punto de cruce
    escapa al cielo                 -> estrellas, ya lenteadas por el trazado
    sin terminar                    -> marcado aparte (subir n_steps)

Asi la MISMA imagen contiene la lente gravitacional analoga (deformacion de
la rejilla del cielo) y la estructura del campo de ondas, con el horizonte y
la ergosfera en su sitio geometrico correcto.

Fisica sin cambios: mismas ecuaciones de rayos (omega = v.k + c|k|), mismo
V_h, misma construccion de ondas parciales con c_m = (-1)^m y normalizacion
por A_in. El disco es una superficie de visualizacion, no fisica nueva.

CONTROLES
---------
  arrastrar    orbitar          rueda    zoom
  sliders      A, B, omega, M, distancia, contraste
  checkboxes   campo Psi / estrellas / disco / bandas de fase / horizonte
  botones      alta calidad, B=0, guardar PNG
  teclas       r reset,  v validar kernel radial contra scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from field_legend import add_phase_colorbar, add_modulus_colorbar
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

try:
    import cupy as cp
except ImportError as e:
    raise SystemExit("Falta CuPy:  pip install cupy-cuda12x\n"
                     f"(detalle: {e})")

from src.field.coloring import ColorConfig, DomainColoring
from src.field.scattering import ScatteringField
from src.physics.background import DrainingBathtub
from src.physics.deformation import GaussianDeformation, NullDeformation

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# =============================================================================
# KERNEL: trazado de rayos + interseccion con el disco del campo
# =============================================================================
_SRC = r'''
extern "C" {

__device__ __forceinline__ void rhs(
    const float x, const float y, const float z,
    const float kx, const float ky, const float kz,
    const float A, const float B, const float c,
    const float eps, const float sig, const float rh,
    float* out)
{
    const float r2 = fmaf(x, x, y * y);
    const float r  = sqrtf(r2);
    const float ir = 1.0f / r, ir2 = 1.0f / r2;

    float h = 0.0f, dh = 0.0f;
    if (eps != 0.0f) {
        const float d = r - rh;
        const float E = __expf(-sig * d * d);
        const float dn = d / rh;
        h  = eps * dn * dn * E;
        dh = eps / (rh * rh) * (2.0f * d - 2.0f * sig * d * d * d) * E;
    }
    const float vr   = -(A * ir) * (1.0f + h);
    const float dvr  =  (A * ir2) * (1.0f + h) - (A * ir) * dh;
    const float vph  =  B * ir;
    const float dvph = -B * ir2;

    const float ux =  x * ir, uy = y * ir;
    const float px = -y * ir, py = x * ir;
    const float vx = vr * ux + vph * px;
    const float vy = vr * uy + vph * py;

    const float dr00 = (1.0f - ux*ux) * ir, dr10 = (-uy*ux) * ir;
    const float dr01 = (-ux*uy) * ir,       dr11 = (1.0f - uy*uy) * ir;
    const float dv00 = dvr*ux*ux + vr*dr00 + dvph*ux*px + vph*(-dr10);
    const float dv10 = dvr*ux*uy + vr*dr10 + dvph*ux*py + vph*( dr00);
    const float dv01 = dvr*uy*ux + vr*dr01 + dvph*uy*px + vph*(-dr11);
    const float dv11 = dvr*uy*uy + vr*dr11 + dvph*uy*py + vph*( dr01);

    const float ikn = rsqrtf(kx*kx + ky*ky + kz*kz + 1e-30f);
    out[0] = -(vx + c * kx * ikn);
    out[1] = -(vy + c * ky * ikn);
    out[2] = -(     c * kz * ikn);
    out[3] =  (dv00 * kx + dv10 * ky);
    out[4] =  (dv01 * kx + dv11 * ky);
    out[5] =  0.0f;
}

__global__ void trace_unified(
    const float* __restrict__ dirx,
    const float* __restrict__ diry,
    const float* __restrict__ dirz,
    const float camx, const float camy, const float camz,
    const float A, const float B, const float c,
    const float eps, const float sig,
    const float dt, const int n_steps, const float adapt_max,
    const float r_cap, const float r_sky, const float z_sky,
    const float r_disk_in, const float r_disk_out, const int disk_on,
    float* __restrict__ out_kx, float* __restrict__ out_ky,
    float* __restrict__ out_kz,
    float* __restrict__ hit_x,  float* __restrict__ hit_y,
    unsigned char* __restrict__ status,  // 0 sin acabar 1 horizonte
                                         // 2 cielo      3 disco
    const int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float rh = fabsf(A) / c;
    const float irh = 1.0f / rh;

    float s[6];
    s[0] = camx; s[1] = camy; s[2] = camz;
    s[3] = -dirx[i]; s[4] = -diry[i]; s[5] = -dirz[i];

    float acc[6], k[6], tmp[6];
    unsigned char st = 0;
    float hx = 0.0f, hy = 0.0f;

    for (int n = 0; n < n_steps; ++n) {
        const float rr0 = sqrtf(s[0]*s[0] + s[1]*s[1]);
        const float sc = fminf(fmaxf(rr0 * irh, 1.0f), adapt_max);
        const float hs = dt * sc;
        const float xp = s[0], yp = s[1], zp = s[2];   // posicion previa

        rhs(s[0],s[1],s[2],s[3],s[4],s[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j){ acc[j]=k[j]; tmp[j]=s[j]+0.5f*hs*k[j]; }
        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j){ acc[j]+=2.0f*k[j]; tmp[j]=s[j]+0.5f*hs*k[j]; }
        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j){ acc[j]+=2.0f*k[j]; tmp[j]=s[j]+hs*k[j]; }
        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j) s[j] += (hs/6.0f)*(acc[j]+k[j]);

        // --- interseccion con el disco z = 0 (superficie del fluido) ---
        // Se detecta el cambio de signo de z entre el paso previo y el actual
        // y se interpola linealmente el punto de cruce. El disco es opaco:
        // el primer cruce dentro del anillo [r_disk_in, r_disk_out] termina
        // el rayo.
        if (disk_on && (zp * s[2] <= 0.0f) && (zp != s[2])) {
            const float f  = zp / (zp - s[2]);          // en [0,1]
            const float xc = xp + f * (s[0] - xp);
            const float yc = yp + f * (s[1] - yp);
            const float rc2 = sqrtf(xc*xc + yc*yc);
            if (rc2 >= r_disk_in && rc2 <= r_disk_out) {
                hx = xc; hy = yc; st = 3; break;
            }
        }

        const float rc  = sqrtf(s[0]*s[0] + s[1]*s[1]);
        const float far = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
        if (rc < r_cap)                         { st = 1; break; }
        if (far > r_sky || fabsf(s[2]) > z_sky) { st = 2; break; }
    }

    out_kx[i] = s[3]; out_ky[i] = s[4]; out_kz[i] = s[5];
    hit_x[i] = hx;    hit_y[i] = hy;
    status[i] = st;
}

}  // extern "C"
'''

_MOD = cp.RawModule(code=_SRC, options=("-use_fast_math",))
_KERN = _MOD.get_function("trace_unified")


# =============================================================================
# PARAMETROS
# =============================================================================
@dataclass
class Params:
    A: float = 1.0
    B: float = 0.7
    c: float = 1.0
    eps: float = 0.0
    sig: float = 2.0
    omega: float = 1.6
    m_max: int = 20
    dt: float = 0.03
    n_steps: int = 1600
    adapt_max: float = 8.0
    capture_pad: float = 1.002
    r_sky: float = 40.0
    z_sky: float = 40.0
    r_disk_out: float = 11.0

    @property
    def r_h(self):
        return abs(self.A) / self.c

    @property
    def r_e(self):
        return float(np.hypot(self.A, self.B) / self.c)


@dataclass
class Camera:
    dist: float = 20.0
    azim: float = -90.0
    elev: float = 22.0
    fov: float = 45.0

    def position(self):
        a, e = np.radians(self.azim), np.radians(self.elev)
        return np.array([self.dist*np.cos(e)*np.cos(a),
                         self.dist*np.cos(e)*np.sin(a),
                         self.dist*np.sin(e)], dtype=np.float32)


@dataclass
class Layers:
    field: bool = True
    stars: bool = True
    disk: bool = True
    phase_bands: bool = True
    horizon: bool = True


# =============================================================================
# TEXTURAS
# =============================================================================
def make_sky(H=768, n_stars=8000, seed=7, grid=True, gstr=0.5):
    rng = np.random.default_rng(seed)
    W = 2*H
    tex = np.zeros((H, W, 3), dtype=np.float32)
    u = rng.uniform(-1, 1, n_stars); ph = rng.uniform(0, 2*np.pi, n_stars)
    py = ((np.arccos(u)/np.pi)*(H-1)).astype(int)
    px = ((ph/(2*np.pi))*(W-1)).astype(int)
    mag = rng.pareto(1.7, n_stars) + 0.12
    mag = np.clip(mag/mag.max(), 0.02, 1.0)
    t = rng.uniform(0, 1, n_stars)
    col = np.stack([0.72+0.28*t, 0.80+0.14*np.abs(t-0.5), 1.0-0.25*t], axis=1)
    np.add.at(tex, (py, px), (mag[:, None]*col).astype(np.float32))
    try:
        from scipy.ndimage import gaussian_filter
        tex = gaussian_filter(tex, sigma=(0.7, 0.7, 0))
        tex += 0.55*gaussian_filter(tex, sigma=(2.2, 2.2, 0))
    except Exception:
        pass
    if grid:
        th = np.arange(H)*np.pi/H
        pl = np.arange(W)*2*np.pi/W
        g = ((np.abs(np.sin(th[:, None]*40.0)) > 0.995)
             | (np.abs(np.sin(pl[None, :]*40.0)) > 0.995)).astype(np.float32)*gstr
        tex[..., 0] += g*0.25; tex[..., 1] += g*0.55; tex[..., 2] += g*0.75
    return cp.asarray(np.clip(tex, 0, 4))


class FieldTexture:
    """Textura cartesiana del domain coloring de Psi, para muestrear por (x,y).

    Se construye una vez por (omega, M, A, B, eps) y se reutiliza en cada
    frame: mover la camara NO recalcula el campo.
    """

    def __init__(self, p: Params, n: int = 700, bands: bool = True):
        self.n = n
        self.r_out = p.r_disk_out
        defo = (GaussianDeformation(r_h=p.r_h, eps=p.eps, sigma=p.sig)
                if p.eps != 0.0 else NullDeformation(r_h=p.r_h))
        bg = DrainingBathtub(A=p.A, B=p.B, c=p.c)
        sf = ScatteringField(bg, omega=p.omega, defo=defo, m_max=p.m_max,
                             r_max=p.r_disk_out + 1.0, n_out=1400)
        g = np.linspace(-self.r_out, self.r_out, n)
        X, Y = np.meshgrid(g, g)
        psi = sf.evaluate_cartesian(X, Y)
        cfg = ColorConfig(phase_bands=bands)
        rgb = DomainColoring(cfg).rgb(psi)
        # marca de fuera de dominio -> transparente (se usa el cielo)
        self.valid = cp.asarray(np.isfinite(psi).astype(np.float32))
        self.tex = cp.asarray(rgb.astype(np.float32))
        self.sfield = sf

    def sample(self, x, y):
        """Bilineal sobre la textura cartesiana. x,y: cupy (N,)."""
        f = (x + self.r_out) / (2*self.r_out) * (self.n - 1)
        g = (y + self.r_out) / (2*self.r_out) * (self.n - 1)
        i0 = cp.clip(cp.floor(g).astype(cp.int32), 0, self.n-1)
        j0 = cp.clip(cp.floor(f).astype(cp.int32), 0, self.n-1)
        i1 = cp.clip(i0+1, 0, self.n-1); j1 = cp.clip(j0+1, 0, self.n-1)
        wy = (g - i0)[:, None]; wx = (f - j0)[:, None]
        top = self.tex[i0, j0]*(1-wx) + self.tex[i0, j1]*wx
        bot = self.tex[i1, j0]*(1-wx) + self.tex[i1, j1]*wx
        col = top*(1-wy) + bot*wy
        v = (self.valid[i0, j0] * self.valid[i1, j1])[:, None]
        return col, v


def sample_sky(k, sky):
    d = -k
    n = cp.sqrt((d*d).sum(axis=0)) + 1e-30
    dx, dy, dz = d[0]/n, d[1]/n, d[2]/n
    th = cp.arccos(cp.clip(dz, -1, 1)); ph = cp.mod(cp.arctan2(dy, dx), 2*np.pi)
    H, W, _ = sky.shape
    fy = (th/np.pi)*(H-1); fx = (ph/(2*np.pi))*(W-1)
    y0 = cp.clip(cp.floor(fy).astype(cp.int32), 0, H-1)
    x0 = cp.clip(cp.floor(fx).astype(cp.int32), 0, W-1)
    y1 = cp.clip(y0+1, 0, H-1); x1 = (x0+1) % W
    wy = (fy-y0)[:, None]; wx = (fx-x0)[:, None]
    top = sky[y0, x0]*(1-wx) + sky[y0, x1]*wx
    bot = sky[y1, x0]*(1-wx) + sky[y1, x1]*wx
    return top*(1-wy) + bot*wy


# =============================================================================
# RENDER
# =============================================================================
def camera_rays(cam: Camera, w, h):
    pos = cam.position()
    fwd = -pos/np.linalg.norm(pos)
    up = np.array([0, 0, 1.0], dtype=np.float32)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    tup = np.cross(right, fwd)
    hh = np.tan(np.radians(cam.fov)/2); hw = hh*(w/h)
    ys = cp.linspace(hh, -hh, h, dtype=cp.float32)
    xs = cp.linspace(-hw, hw, w, dtype=cp.float32)
    gy, gx = cp.meshgrid(ys, xs, indexing="ij")
    d = (cp.asarray(fwd).reshape(3,1,1) + cp.asarray(right).reshape(3,1,1)*gx[None]
         + cp.asarray(tup).reshape(3,1,1)*gy[None])
    d = d/cp.linalg.norm(d, axis=0, keepdims=True)
    return pos, d.reshape(3, -1)


def render(p: Params, cam: Camera, w, h, sky, ftex: FieldTexture, L: Layers,
           block=128):
    pos, dirs = camera_rays(cam, w, h)
    dirs = cp.ascontiguousarray(dirs, dtype=cp.float32)
    N = dirs.shape[1]
    okx = cp.empty(N, cp.float32); oky = cp.empty(N, cp.float32)
    okz = cp.empty(N, cp.float32)
    hx = cp.empty(N, cp.float32); hy = cp.empty(N, cp.float32)
    st = cp.empty(N, cp.uint8)

    _KERN(((N+block-1)//block,), (block,), (
        dirs[0], dirs[1], dirs[2],
        np.float32(pos[0]), np.float32(pos[1]), np.float32(pos[2]),
        np.float32(p.A), np.float32(p.B), np.float32(p.c),
        np.float32(p.eps), np.float32(p.sig),
        np.float32(p.dt), np.int32(p.n_steps), np.float32(p.adapt_max),
        np.float32(p.r_h*p.capture_pad), np.float32(p.r_sky),
        np.float32(p.z_sky),
        np.float32(p.r_h*1.001), np.float32(p.r_disk_out),
        np.int32(1 if (L.disk and L.field) else 0),
        okx, oky, okz, hx, hy, st, np.int32(N)))

    kf = cp.stack([okx, oky, okz])
    col = cp.zeros((N, 3), cp.float32)
    if L.stars:
        col = sample_sky(kf, sky)
    hit = (st == 3)
    if bool(hit.any()) and L.field and L.disk:
        fc, v = ftex.sample(hx[hit], hy[hit])
        base = col[hit]
        col[hit] = fc*v + base*(1.0-v)
    col[st == 1] = 0.0
    col[st == 0] = cp.asarray([0.05, 0.0, 0.06], cp.float32)

    img = col.reshape(h, w, 3)
    img = img/(1.0+img)
    img = cp.clip(img*1.9, 0, 1)**(1/2.2)
    stats = cp.stack([(st == 1).mean(), (st == 3).mean(), (st == 0).mean()])
    return cp.asnumpy(img), cp.asnumpy(stats)


# =============================================================================
# APLICACION
# =============================================================================
class App:
    LOW = (420, 250); MID = (820, 490); HIGH = (1600, 950)

    def __init__(self):
        self.p = Params(); self.cam = Camera(); self.L = Layers()
        self.sky = make_sky()
        print("construyendo campo Psi (kernel CUDA radial)...", flush=True)
        t0 = time.time()
        self.ftex = FieldTexture(self.p)
        print(f"  listo en {time.time()-t0:.1f}s")

        self.fig = plt.figure(figsize=(13.5, 8.8), facecolor="#08080c")
        self.ax = self.fig.add_axes([0.23, 0.24, 0.63, 0.72])
        self.ax.set_axis_off()
        self.im = self.ax.imshow(np.zeros((2,2,3)), interpolation="bilinear")
        self.title = self.fig.text(0.23, 0.965, "", color="#9fb6d0", fontsize=9)

        axc = self.fig.add_axes([0.015, 0.63, 0.185, 0.30], facecolor="#151820")
        self.chk = CheckButtons(
            axc, ["campo Ψ", "estrellas", "disco", "bandas fase", "horizonte"],
            [True, True, True, True, True])
        for t in self.chk.labels:
            t.set_color("#c8d4e4"); t.set_fontsize(9)
        self.chk.on_clicked(self._toggle)

        def mk(y, lab, lo, hi, v, step=None):
            a = self.fig.add_axes([0.07, y, 0.14, 0.022], facecolor="#20242c")
            s = Slider(a, lab, lo, hi, valinit=v, valstep=step, color="#4f8fc0")
            s.label.set_color("#c8d4e4"); s.label.set_fontsize(8)
            s.valtext.set_color("#c8d4e4"); s.valtext.set_fontsize(8)
            return s

        self.s_A = mk(0.56, "A", 0.3, 2.5, self.p.A)
        self.s_B = mk(0.51, "B", 0.0, 2.0, self.p.B)
        self.s_w = mk(0.46, "ω", 0.5, 3.5, self.p.omega)
        self.s_M = mk(0.41, "M", 4, 30, self.p.m_max, 1)
        self.s_d = mk(0.36, "distancia", 5.0, 60.0, self.cam.dist)
        self.s_e = mk(0.31, "ε", -0.6, 0.6, self.p.eps)
        for s in (self.s_A, self.s_B, self.s_w, self.s_M, self.s_e):
            s.on_changed(self._field_changed)
        self.s_d.on_changed(lambda v: self.draw(self.MID))

        def mkb(x, y, lab, cb):
            a = self.fig.add_axes([x, y, 0.085, 0.04])
            b = Button(a, lab, color="#2a3340", hovercolor="#3d4a5c")
            b.label.set_color("#c8d4e4"); b.label.set_fontsize(8)
            b.on_clicked(cb); return b
        self.b1 = mkb(0.30, 0.11, "Alta calidad", lambda e: self.draw(self.HIGH))
        self.b2 = mkb(0.42, 0.11, "B = 0", lambda e: self.s_B.set_val(0.0))
        self.b3 = mkb(0.54, 0.11, "Guardar PNG", self._save)

        c = self.fig.canvas
        self.drag = False; self.last = None
        c.mpl_connect("button_press_event", self._press)
        c.mpl_connect("button_release_event", self._release)
        c.mpl_connect("motion_notify_event", self._motion)
        c.mpl_connect("scroll_event", self._scroll)
        c.mpl_connect("key_press_event", self._key)
        self.legend_ax = None
        self.cb_phase = add_phase_colorbar(
            self.fig, rect=(0.885, 0.32, 0.018, 0.42),
            band_spacing=np.pi / 8, show_bands=True)
        self.cb_mod = add_modulus_colorbar(
            self.fig, rect=(0.945, 0.32, 0.014, 0.42))
        self.draw(self.MID)
        def _dbg(e):
            print(f"[evt] {e.name}  inaxes={e.inaxes}  "
                  f"es_self.ax={e.inaxes is self.ax}  button={getattr(e,'button',None)}")
        c.mpl_connect("button_press_event", _dbg)
        c.mpl_connect("scroll_event", _dbg)

    def _toggle(self, label):
        m = {"campo Ψ": "field", "estrellas": "stars", "disco": "disk",
             "bandas fase": "phase_bands", "horizonte": "horizon"}
        a = m[label]; setattr(self.L, a, not getattr(self.L, a))
        if a == "phase_bands":
            self._rebuild_field()
        self.draw(self.MID)

    def _sync(self):
        self.p.A = self.s_A.val; self.p.B = self.s_B.val
        self.p.omega = self.s_w.val; self.p.m_max = int(self.s_M.val)
        self.p.eps = self.s_e.val; self.cam.dist = self.s_d.val

    def _rebuild_field(self):
        t0 = time.time()
        self.ftex = FieldTexture(self.p, bands=self.L.phase_bands)
        print(f"campo recalculado en {time.time()-t0:.1f}s")

    def _field_changed(self, _v):
        self._sync(); self._rebuild_field(); self.draw(self.MID)

    def _save(self, _e):
        self._sync()
        img, _ = render(self.p, self.cam, *self.HIGH, self.sky, self.ftex, self.L)
        n = (f"unificado_A{self.p.A:.2f}_B{self.p.B:.2f}"
             f"_w{self.p.omega:.2f}_M{self.p.m_max}.png")
        f = plt.figure(figsize=(16, 9.5), facecolor="black")
        a = f.add_axes([0,0,1,1]); a.imshow(img); a.set_axis_off()
        f.savefig(OUT/n, dpi=110, facecolor="black"); plt.close(f)
        print("guardado:", OUT/n)

    def _press(self, e):
        if e.inaxes is self.ax and e.button == 1:
            self.drag = True; self.last = (e.x, e.y)

    def _release(self, _e):
        if self.drag: self.drag = False; self.draw(self.MID)

    def _motion(self, e):
        if not self.drag or self.last is None or e.x is None: return
        dx, dy = e.x-self.last[0], e.y-self.last[1]; self.last = (e.x, e.y)
        self.cam.azim += dx*0.4
        self.cam.elev = float(np.clip(self.cam.elev + dy*0.4, -85, 85))
        self.draw(self.LOW)

    def _scroll(self, e):
        self.cam.fov = float(np.clip(
            self.cam.fov*(0.9 if e.step > 0 else 1/0.9), 4.0, 100.0))
        self.draw(self.LOW if self.drag else self.MID)

    def _key(self, e):
        if e.key == "r":
            self.cam = Camera(); self.s_d.set_val(self.cam.dist)
        elif e.key == "v":
            from src.field import cuda_radial
            bg = DrainingBathtub(A=self.p.A, B=self.p.B, c=self.p.c)
            cuda_radial.validate_against_scipy(bg, omega=self.p.omega,
                                               m_values=range(-4, 5))

    def draw(self, size):
        self._sync()
        w, h = size
        old = self.p.n_steps
        if size == self.LOW: self.p.n_steps = 900
        t0 = time.perf_counter()
        img, stats = render(self.p, self.cam, w, h, self.sky, self.ftex, self.L)
        ms = (time.perf_counter()-t0)*1000
        self.p.n_steps = old
        self.im.set_data(img); self.im.set_extent([0, w, h, 0])
        self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0)
        warn = "  ⚠ sube n_steps" if stats[2] > 0.02 else ""
        self.title.set_text(
            f"A={self.p.A:.2f} B={self.p.B:.2f} ω={self.p.omega:.2f} "
            f"M={self.p.m_max} ε={self.p.eps:.2f} | r_h={self.p.r_h:.2f} "
            f"r_e={self.p.r_e:.2f} | sombra={100*stats[0]:.1f}% "
            f"disco={100*stats[1]:.1f}% | {ms:.0f} ms{warn}\n"
            f"matiz=arg Ψ   luminancia=|Ψ|   fondo=estrellas lenteadas   "
            f"negro=horizonte")
        vis = self.L.field and self.L.disk
        self.cb_phase.set_visible(vis)
        self.cb_mod.set_visible(vis)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(f"GPU: {props['name'].decode()}  SMs={props['multiProcessorCount']}")
    matplotlib.use("TkAgg")     # si falla: "QtAgg"
    app = App()
    plt.show()
