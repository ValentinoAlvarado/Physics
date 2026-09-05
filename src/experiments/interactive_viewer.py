"""
VISOR INTERACTIVO CUDA — cuerda negra acustica (vortice cilindrico rotante)

Archivo unico y autocontenido. Kernel CUDA propio via CuPy (NVRTC, no requiere
Visual Studio). Un hilo por rayo, bucle RK4 completo dentro del kernel.

    pip install cupy-cuda12x

CONTROLES
---------
  arrastrar (bot. izq.) -> orbitar camara
  rueda                 -> zoom (campo de vision)
  sliders               -> A, B, eps, sigma, distancia
  boton "Alta calidad"  -> re-render a 1920x1080
  boton "B = 0"         -> comparacion sin rotacion (sombra simetrica)
  boton "Guardar PNG"   -> guarda la vista actual en results/
  tecla 'r'             -> reinicia camara
  tecla 'v'             -> verifica paso adaptativo contra paso fijo
  tecla 'b'             -> benchmark en consola

=============================================================================
FISICA
=============================================================================
Fondo: draining bathtub extendido como vortice CILINDRICO invariante en z.

    v_r(r) = -(A/r)[1 + h(r)] ,  v_phi(r) = B/r ,  v_z = 0
    r = sqrt(x^2+y^2)  (radio CILINDRICO)
    horizonte r_h = |A|/c   (un CILINDRO: analogo de "cuerda negra")
    ergosfera r_e = sqrt(A^2+B^2)/c

Rayos (exacto para c homogenea):
    omega = v.k + c|k|                    (se conserva)
    dx/dt = v + c*khat
    dk_i/dt = -k_j d(v_j)/d(x_i)
    k_z se conserva exactamente (v no depende de z, v_z=0)

Trazado INVERSO desde la camara: el sonido que LLEGA desde la direccion de
pantalla dhat viaja con k propto -dhat; se integra hacia atras en el tiempo y
la fuente del cielo se muestrea en -k_final.

ADVERTENCIA PARA EL POSTER: en 3D los rayos tienen k_z, que actua como MASA
EFECTIVA en el problema 2+1D reducido. Los QNM de tu analisis espectral son
k_z = 0. Esta vista es legitima como retrato de la GEOMETRIA EFECTIVA (mismo
fondo, mismo horizonte, mismo arrastre de marco), no del mismo sector de modos.

=============================================================================
PASO ADAPTATIVO
=============================================================================
Los gradientes de velocidad caen como A/r^2, asi que lejos del vortice el rayo
es casi recto. El kernel escala el paso como

    h_local = dt * clamp(r/r_h, 1, adapt_max)

Esto reduce mucho los pasos gastados en el trayecto largo hacia el cielo. ES
UNA APROXIMACION: verifica con la tecla 'v' o con verify_adaptive(), que
compara contra paso fijo (adapt_max=1) y reporta la discrepancia de estado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    import cupy as cp
except ImportError as e:
    raise SystemExit("Falta CuPy:  pip install cupy-cuda12x\n"
                     f"(detalle: {e})")

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# =============================================================================
# KERNEL CUDA
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
    const float inv_r  = 1.0f / r;
    const float inv_r2 = 1.0f / r2;

    float h = 0.0f, dh = 0.0f;
    if (eps != 0.0f) {
        const float d  = r - rh;
        const float E  = __expf(-sig * d * d);
        const float dn = d / rh;
        h  = eps * dn * dn * E;
        dh = eps / (rh * rh) * (2.0f * d - 2.0f * sig * d * d * d) * E;
    }

    const float vr   = -(A * inv_r) * (1.0f + h);
    const float dvr  =  (A * inv_r2) * (1.0f + h) - (A * inv_r) * dh;
    const float vph  =  B * inv_r;
    const float dvph = -B * inv_r2;

    const float ux =  x * inv_r,  uy = y * inv_r;
    const float px = -y * inv_r,  py = x * inv_r;

    const float vx = vr * ux + vph * px;
    const float vy = vr * uy + vph * py;

    const float dr00 = (1.0f - ux * ux) * inv_r;
    const float dr10 = (     - uy * ux) * inv_r;
    const float dr01 = (     - ux * uy) * inv_r;
    const float dr11 = (1.0f - uy * uy) * inv_r;

    const float dv00 = dvr*ux*ux + vr*dr00 + dvph*ux*px + vph*(-dr10);
    const float dv10 = dvr*ux*uy + vr*dr10 + dvph*ux*py + vph*( dr00);
    const float dv01 = dvr*uy*ux + vr*dr01 + dvph*uy*px + vph*(-dr11);
    const float dv11 = dvr*uy*uy + vr*dr11 + dvph*uy*py + vph*( dr01);

    const float ikn = rsqrtf(kx*kx + ky*ky + kz*kz + 1e-30f);

    // signo global negativo = integracion HACIA ATRAS en el tiempo
    out[0] = -(vx + c * kx * ikn);
    out[1] = -(vy + c * ky * ikn);
    out[2] = -(     c * kz * ikn);
    out[3] =  (dv00 * kx + dv10 * ky);
    out[4] =  (dv01 * kx + dv11 * ky);
    out[5] =  0.0f;                      // k_z se conserva
}

__global__ void trace_kernel(
    const float* __restrict__ dirx,
    const float* __restrict__ diry,
    const float* __restrict__ dirz,
    const float camx, const float camy, const float camz,
    const float A, const float B, const float c,
    const float eps, const float sig,
    const float dt, const int n_steps, const float adapt_max,
    const float r_cap, const float r_sky, const float z_sky,
    float* __restrict__ out_kx,
    float* __restrict__ out_ky,
    float* __restrict__ out_kz,
    unsigned char* __restrict__ status,   // 0=sin acabar 1=capturado 2=escapado
    const int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float rh = fabsf(A) / c;
    const float inv_rh = 1.0f / rh;

    // estado en REGISTROS
    float s[6];
    s[0] = camx;  s[1] = camy;  s[2] = camz;
    s[3] = -dirx[i];  s[4] = -diry[i];  s[5] = -dirz[i];

    // RK4 acumulando en sitio: 18 floats en vez de 36 -> mejor ocupacion
    float acc[6], k[6], tmp[6];
    unsigned char st = 0;

    for (int n = 0; n < n_steps; ++n) {
        // paso adaptativo: los gradientes caen como A/r^2, asi que lejos del
        // vortice se puede avanzar mas por paso sin perder precision
        const float rr = sqrtf(s[0]*s[0] + s[1]*s[1]);
        const float sc = fminf(fmaxf(rr * inv_rh, 1.0f), adapt_max);
        const float hstep = dt * sc;

        rhs(s[0],s[1],s[2],s[3],s[4],s[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j) { acc[j] = k[j]; tmp[j] = s[j] + 0.5f*hstep*k[j]; }

        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j) { acc[j] += 2.0f*k[j]; tmp[j] = s[j] + 0.5f*hstep*k[j]; }

        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j) { acc[j] += 2.0f*k[j]; tmp[j] = s[j] + hstep*k[j]; }

        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k);
        #pragma unroll
        for (int j=0;j<6;++j) s[j] += (hstep/6.0f) * (acc[j] + k[j]);

        const float rc  = sqrtf(s[0]*s[0] + s[1]*s[1]);
        const float far = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
        if (rc < r_cap)                         { st = 1; break; }
        if (far > r_sky || fabsf(s[2]) > z_sky) { st = 2; break; }
    }

    out_kx[i] = s[3];
    out_ky[i] = s[4];
    out_kz[i] = s[5];
    status[i] = st;
}

}  // extern "C"
'''

_MODULE = cp.RawModule(code=_SRC, options=("-use_fast_math",))
_KERNEL = _MODULE.get_function("trace_kernel")


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
    dt: float = 0.03
    n_steps: int = 1400
    adapt_max: float = 8.0     # 1.0 = paso fijo (referencia de verificacion)
    capture_pad: float = 1.002
    r_sky: float = 38.0
    z_sky: float = 38.0

    @property
    def r_h(self):
        return abs(self.A) / self.c

    @property
    def r_e(self):
        return float(np.hypot(self.A, self.B) / self.c)


@dataclass
class Camera:
    dist: float = 18.0
    azim: float = -90.0
    elev: float = 12.0
    fov: float = 42.0

    def position(self):
        a, e = np.radians(self.azim), np.radians(self.elev)
        return np.array([self.dist*np.cos(e)*np.cos(a),
                         self.dist*np.cos(e)*np.sin(a),
                         self.dist*np.sin(e)], dtype=np.float32)


# =============================================================================
# TRAZADO
# =============================================================================
def trace_rays_cuda(dirs, cam_pos, p: Params, block=128):
    """Un solo lanzamiento de kernel. Devuelve (k_final (3,N), status (N,))."""
    dirs = cp.ascontiguousarray(dirs, dtype=cp.float32)
    N = dirs.shape[1]
    okx = cp.empty(N, dtype=cp.float32)
    oky = cp.empty(N, dtype=cp.float32)
    okz = cp.empty(N, dtype=cp.float32)
    status = cp.empty(N, dtype=cp.uint8)
    grid = ((N + block - 1) // block,)
    _KERNEL(grid, (block,), (
        dirs[0], dirs[1], dirs[2],
        np.float32(cam_pos[0]), np.float32(cam_pos[1]), np.float32(cam_pos[2]),
        np.float32(p.A), np.float32(p.B), np.float32(p.c),
        np.float32(p.eps), np.float32(p.sig),
        np.float32(p.dt), np.int32(p.n_steps), np.float32(p.adapt_max),
        np.float32(p.r_h * p.capture_pad),
        np.float32(p.r_sky), np.float32(p.z_sky),
        okx, oky, okz, status, np.int32(N),
    ))
    return cp.stack([okx, oky, okz]), status


def camera_rays(cam: Camera, width, height):
    pos = cam.position()
    fwd = -pos / np.linalg.norm(pos)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    half_h = np.tan(np.radians(cam.fov) / 2.0)
    half_w = half_h * (width / height)
    ys = cp.linspace(half_h, -half_h, height, dtype=cp.float32)
    xs = cp.linspace(-half_w, half_w, width, dtype=cp.float32)
    gy, gx = cp.meshgrid(ys, xs, indexing="ij")
    f = cp.asarray(fwd).reshape(3, 1, 1)
    rr = cp.asarray(right).reshape(3, 1, 1)
    uu = cp.asarray(true_up).reshape(3, 1, 1)
    d = f + rr * gx[None] + uu * gy[None]
    d = d / cp.linalg.norm(d, axis=0, keepdims=True)
    return pos, d.reshape(3, -1)


# =============================================================================
# CIELO
# =============================================================================
def make_sky_texture(H=1024, n_stars=9000, seed=7, grid=True, gstr=0.35):
    rng = np.random.default_rng(seed)
    W = 2 * H
    tex = np.zeros((H, W, 3), dtype=np.float32)
    u = rng.uniform(-1.0, 1.0, n_stars)
    ph = rng.uniform(0.0, 2*np.pi, n_stars)
    th = np.arccos(u)
    py = ((th/np.pi)*(H-1)).astype(int)
    px = ((ph/(2*np.pi))*(W-1)).astype(int)
    mag = rng.pareto(1.7, n_stars) + 0.12
    mag = np.clip(mag/mag.max(), 0.02, 1.0)
    t = rng.uniform(0.0, 1.0, n_stars)
    col = np.stack([0.72+0.28*t, 0.80+0.14*np.abs(t-0.5), 1.0-0.25*t], axis=1)
    np.add.at(tex, (py, px), (mag[:, None]*col).astype(np.float32))
    try:
        from scipy.ndimage import gaussian_filter
        tex = gaussian_filter(tex, sigma=(0.7, 0.7, 0))
        tex += 0.55*gaussian_filter(tex, sigma=(2.2, 2.2, 0))
    except Exception:
        pass
    if grid:
        th_l = np.arange(H)*np.pi/H
        ph_l = np.arange(W)*2*np.pi/W
        g = ((np.abs(np.sin(th_l[:, None]*40.0)) > 0.995)
             | (np.abs(np.sin(ph_l[None, :]*40.0)) > 0.995)).astype(np.float32)*gstr
        tex[..., 0] += g*0.25; tex[..., 1] += g*0.55; tex[..., 2] += g*0.75
    return cp.asarray(np.clip(tex, 0.0, 4.0))


def shade(k_final, status, sky):
    d = -k_final
    n = cp.sqrt((d*d).sum(axis=0)) + 1e-30
    dx, dy, dz = d[0]/n, d[1]/n, d[2]/n
    theta = cp.arccos(cp.clip(dz, -1.0, 1.0))
    phi = cp.mod(cp.arctan2(dy, dx), 2*np.pi)
    H, W, _ = sky.shape
    fy = (theta/np.pi)*(H-1)
    fx = (phi/(2*np.pi))*(W-1)
    y0 = cp.clip(cp.floor(fy).astype(cp.int32), 0, H-1)
    x0 = cp.clip(cp.floor(fx).astype(cp.int32), 0, W-1)
    y1 = cp.clip(y0+1, 0, H-1)
    x1 = (x0+1) % W
    wy = (fy-y0)[:, None]; wx = (fx-x0)[:, None]
    top = sky[y0, x0]*(1-wx) + sky[y0, x1]*wx
    bot = sky[y1, x0]*(1-wx) + sky[y1, x1]*wx
    col = top*(1-wy) + bot*wy
    col[status == 1] = 0.0
    col[status == 0] = cp.asarray([0.05, 0.0, 0.06], dtype=cp.float32)
    return col


def render(p: Params, cam: Camera, width, height, sky):
    pos, dirs = camera_rays(cam, width, height)
    kf, st = trace_rays_cuda(dirs, pos, p)
    col = shade(kf, st, sky)
    img = col.reshape(height, width, 3)
    img = img/(1.0+img)
    img = cp.clip(img*1.9, 0, 1)**(1/2.2)
    stats = cp.stack([(st == 1).mean(), (st == 0).mean()])
    img_np = cp.asnumpy(img)
    s_np = cp.asnumpy(stats)
    return img_np, float(s_np[0]), float(s_np[1])


# =============================================================================
# VERIFICACION Y BENCHMARK
# =============================================================================
def verify_adaptive(p: Params = None, n=200000):
    """Compara paso adaptativo contra paso FIJO sobre rayos aleatorios.

    El paso adaptativo es una aproximacion; esto mide cuanto se aparta. Lo que
    importa es la coincidencia de ESTADO (capturado/escapado) y el error
    angular de la direccion final, que es lo que colorea el pixel.
    """
    p = p or Params()
    rng = np.random.default_rng(1)
    d = rng.normal(size=(3, n)).astype(np.float32)
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    d = cp.asarray(d)
    cam = Camera().position()

    p_fix = Params(**{**p.__dict__, "adapt_max": 1.0, "n_steps": 6000})
    k_a, st_a = trace_rays_cuda(d, cam, p)
    k_f, st_f = trace_rays_cuda(d, cam, p_fix)

    same = float((st_a == st_f).mean())
    both_esc = (st_a == 2) & (st_f == 2)
    if int(both_esc.sum()) > 0:
        a = k_a[:, both_esc]; f = k_f[:, both_esc]
        a = a/cp.linalg.norm(a, axis=0, keepdims=True)
        f = f/cp.linalg.norm(f, axis=0, keepdims=True)
        ang = cp.arccos(cp.clip((a*f).sum(axis=0), -1, 1))
        ang_deg = cp.asnumpy(ang)*180/np.pi
        print(f"error angular en escapados: mediana={np.median(ang_deg):.4f}° "
              f"p99={np.percentile(ang_deg,99):.4f}°  max={ang_deg.max():.4f}°")
    print(f"coincidencia de estado adaptativo vs fijo: {100*same:.3f}%")
    print(f"  sin acabar: adaptativo={int((st_a==0).sum())} "
          f"fijo={int((st_f==0).sum())}  (de {n})")
    if same > 0.995:
        print("  -> OK: el paso adaptativo es fiable para esta vista")
    else:
        print("  -> baja adapt_max (p.ej. 4.0) o sube n_steps")


def benchmark(p: Params = None, sizes=((960, 540), (1920, 1080), (3840, 2160))):
    p = p or Params()
    cam = Camera()
    for (w, h) in sizes:
        pos, dirs = camera_rays(cam, w, h)
        trace_rays_cuda(dirs, pos, p)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter(); reps = 5
        for _ in range(reps):
            trace_rays_cuda(dirs, pos, p)
        cp.cuda.Stream.null.synchronize()
        dt = (time.perf_counter()-t0)/reps
        print(f"{w}x{h}: {dt*1000:8.2f} ms/frame  {w*h/dt/1e6:7.1f} Mrayos/s"
              f"  ({1/dt:6.1f} FPS)")


# =============================================================================
# VISOR
# =============================================================================
class Viewer:
    LOW = (480, 270)
    MID = (960, 540)
    HIGH = (1920, 1080)

    def __init__(self):
        self.p = Params()
        self.cam = Camera()
        self.sky = make_sky_texture()
        self.dragging = False
        self.last = None

        self.fig = plt.figure(figsize=(12.5, 8.4), facecolor="#08080c")
        self.ax = self.fig.add_axes([0.02, 0.28, 0.96, 0.70])
        self.ax.set_axis_off()
        self.im = self.ax.imshow(np.zeros((2, 2, 3)), interpolation="bilinear")
        self.title = self.ax.set_title("", color="#9fb6d0", fontsize=10)

        def mk(y, label, lo, hi, val):
            ax = self.fig.add_axes([0.10, y, 0.32, 0.028], facecolor="#20242c")
            s = Slider(ax, label, lo, hi, valinit=val, color="#4f8fc0")
            s.label.set_color("#c8d4e4"); s.valtext.set_color("#c8d4e4")
            return s

        self.s_A = mk(0.20, "A (sumidero)", 0.2, 3.0, self.p.A)
        self.s_B = mk(0.155, "B (circulación)", 0.0, 3.0, self.p.B)
        self.s_eps = mk(0.11, "ε (deformación)", -0.8, 0.8, self.p.eps)
        self.s_sig = mk(0.065, "σ (ancho)", 0.2, 6.0, self.p.sig)
        self.s_dist = mk(0.02, "distancia", 4.0, 60.0, self.cam.dist)

        def mkb(x, label):
            ax = self.fig.add_axes([x, 0.02, 0.10, 0.045])
            b = Button(ax, label, color="#2a3340", hovercolor="#3d4a5c")
            b.label.set_color("#c8d4e4")
            return b

        self.b_hq = mkb(0.55, "Alta calidad")
        self.b_b0 = mkb(0.67, "B = 0")
        self.b_sv = mkb(0.79, "Guardar PNG")

        for s in (self.s_A, self.s_B, self.s_eps, self.s_sig, self.s_dist):
            s.on_changed(self.on_slider)
        self.b_hq.on_clicked(lambda _e: self.draw(self.HIGH))
        self.b_b0.on_clicked(lambda _e: self.s_B.set_val(0.0))
        self.b_sv.on_clicked(self.on_save)

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("scroll_event", self.on_scroll)
        c.mpl_connect("key_press_event", self.on_key)

        self.draw(self.MID)

    def sync(self):
        self.p.A = self.s_A.val
        self.p.B = self.s_B.val
        self.p.eps = self.s_eps.val
        self.p.sig = self.s_sig.val
        self.cam.dist = self.s_dist.val

    def on_slider(self, _v):
        self.draw(self.MID)

    def on_save(self, _e):
        self.sync()
        img, fc, _ = render(self.p, self.cam, *self.HIGH, self.sky)
        name = (f"blackstring_A{self.p.A:.2f}_B{self.p.B:.2f}"
                f"_eps{self.p.eps:.2f}_az{self.cam.azim:.0f}"
                f"_el{self.cam.elev:.0f}_fov{self.cam.fov:.0f}.png")
        fig = plt.figure(figsize=(19.2, 10.8), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.set_axis_off()
        ax.text(0.012, 0.975,
                f"cuerda negra acústica  |  A={self.p.A:.2f}  B={self.p.B:.2f}"
                f"  ε={self.p.eps:.2f}  $r_h$={self.p.r_h:.2f}"
                f"  $r_e$={self.p.r_e:.2f}",
                transform=ax.transAxes, color="#9fb6d0", fontsize=11, va="top")
        fig.savefig(OUT / name, dpi=100, facecolor="black")
        plt.close(fig)
        print("guardado:", OUT / name)

    def on_press(self, e):
        if e.inaxes is self.ax and e.button == 1:
            self.dragging = True; self.last = (e.x, e.y)

    def on_release(self, _e):
        if self.dragging:
            self.dragging = False; self.draw(self.MID)

    def on_motion(self, e):
        if not self.dragging or self.last is None or e.x is None:
            return
        dx, dy = e.x - self.last[0], e.y - self.last[1]
        self.last = (e.x, e.y)
        self.cam.azim += dx*0.4
        self.cam.elev = float(np.clip(self.cam.elev + dy*0.4, -85.0, 85.0))
        self.draw(self.LOW)

    def on_scroll(self, e):
        f = 0.9 if e.step > 0 else 1/0.9
        self.cam.fov = float(np.clip(self.cam.fov*f, 3.0, 100.0))
        self.draw(self.LOW if self.dragging else self.MID)

    def on_key(self, e):
        if e.key == "r":
            self.cam = Camera(); self.s_dist.set_val(self.cam.dist)
        elif e.key == "v":
            print("\n=== verificación paso adaptativo ==="); verify_adaptive(self.p)
        elif e.key == "b":
            print("\n=== benchmark ==="); benchmark(self.p)

    def draw(self, size):
        self.sync()
        w, h = size
        t0 = time.perf_counter()
        img, fc, fs = render(self.p, self.cam, w, h, self.sky)
        ms = (time.perf_counter()-t0)*1000
        self.im.set_data(img)
        self.im.set_extent([0, w, h, 0])
        self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0)
        warn = "  ⚠ sube n_steps" if fs > 0.02 else ""
        self.title.set_text(
            f"A={self.p.A:.2f}  B={self.p.B:.2f}  ε={self.p.eps:.2f}  |  "
            f"$r_h$={self.p.r_h:.2f}  $r_e$={self.p.r_e:.2f}  |  "
            f"az={self.cam.azim:.0f}° el={self.cam.elev:.0f}° "
            f"fov={self.cam.fov:.0f}°  |  sombra={100*fc:.1f}%  "
            f"|  {ms:.0f} ms{warn}")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(f"GPU: {props['name'].decode()}  SMs={props['multiProcessorCount']}")
    print("\n=== benchmark ==="); benchmark()
    print("\n=== verificación paso adaptativo ==="); verify_adaptive()
    print("\n=== visor ===")
    print("arrastra=orbitar | rueda=zoom | r=reset | v=verificar | b=benchmark")
    matplotlib.use("TkAgg")     # si falla, prueba "QtAgg"
    v = Viewer()
    plt.show()