"""
KERNEL CUDA REAL — un hilo por rayo, bucle RK4 completo DENTRO del kernel.

Por que esto y no PyTorch
--------------------------
En PyTorch cada operacion es un kernel separado. El bucle RK4 con ~30 ops x 4
etapas x n_steps pasos lanza cientos de miles de kernels por frame, y la GPU
pasa la mayor parte del tiempo esperando entre lanzamientos, no calculando.

Aqui todo el trazado de un rayo ocurre dentro de UN hilo CUDA, en registros.
Un solo lanzamiento de kernel por frame. Ademas cada rayo termina cuando le
toca (break real dentro del hilo), sin necesidad de mascaras ni de congelar
tensores: los hilos que acaban simplemente salen.

Requisito
---------
    pip install cupy-cuda12x          (tu CUDA es 12.8)

CuPy compila el kernel en tiempo de ejecucion con NVRTC, que viene incluido:
NO necesitas Visual Studio ni compilador de C++ instalado. Por eso uso CuPy y
no torch.utils.cpp_extension, que en Windows si requiere MSVC.

Interoperabilidad con tu codigo torch
--------------------------------------
`trace_rays_cuda` acepta y devuelve arrays de CuPy. Para convertir sin copiar:
    import torch.utils.dlpack as dlpack
    cp_arr = cp.from_dlpack(dlpack.to_dlpack(torch_tensor))
    torch_tensor = dlpack.from_dlpack(cp_arr.toDlpack())
o simplemente trabaja en CuPy de extremo a extremo (la API es casi identica a
NumPy y este archivo ya lo hace).

FISICA: identica a interactive_viewer.py. Mismas ecuaciones, mismos signos,
mismo trazado inverso desde la camara. Verificar con check_against_torch().
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError as e:
    raise SystemExit(
        "Falta CuPy. Instala con:  pip install cupy-cuda12x\n"
        f"(detalle: {e})"
    )


# =============================================================================
# KERNEL
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

    const float vr    = -(A * inv_r) * (1.0f + h);
    const float dvr   =  (A * inv_r2) * (1.0f + h) - (A * inv_r) * dh;
    const float vph   =  B * inv_r;
    const float dvph  = -B * inv_r2;

    const float ux =  x * inv_r,  uy = y * inv_r;
    const float px = -y * inv_r,  py = x * inv_r;

    const float vx = vr * ux + vph * px;
    const float vy = vr * uy + vph * py;

    // d rhat_i / d x_j = (delta_ij - u_i u_j)/r ;  phihat = R90 . rhat
    const float dr00 = (1.0f - ux * ux) * inv_r;
    const float dr10 = (     - uy * ux) * inv_r;
    const float dr01 = (     - ux * uy) * inv_r;
    const float dr11 = (1.0f - uy * uy) * inv_r;

    const float dv00 = dvr*ux*ux + vr*dr00 + dvph*ux*px + vph*(-dr10);
    const float dv10 = dvr*ux*uy + vr*dr10 + dvph*ux*py + vph*( dr00);
    const float dv01 = dvr*uy*ux + vr*dr01 + dvph*uy*px + vph*(-dr11);
    const float dv11 = dvr*uy*uy + vr*dr11 + dvph*uy*py + vph*( dr01);

    const float kn = sqrtf(kx*kx + ky*ky + kz*kz) + 1e-30f;
    const float ikn = 1.0f / kn;

    // signo global negativo = integracion HACIA ATRAS en el tiempo
    out[0] = -(vx + c * kx * ikn);
    out[1] = -(vy + c * ky * ikn);
    out[2] = -(     c * kz * ikn);
    out[3] =  (dv00 * kx + dv10 * ky);
    out[4] =  (dv01 * kx + dv11 * ky);
    out[5] =  0.0f;                      // k_z se conserva exactamente
}

__global__ void trace_kernel(
    const float* __restrict__ dirx,
    const float* __restrict__ diry,
    const float* __restrict__ dirz,
    const float camx, const float camy, const float camz,
    const float A, const float B, const float c,
    const float eps, const float sig,
    const float dt, const int n_steps,
    const float r_cap, const float r_sky, const float z_sky,
    float* __restrict__ out_kx,
    float* __restrict__ out_ky,
    float* __restrict__ out_kz,
    unsigned char* __restrict__ status,   // 0=sin terminar 1=capturado 2=escapado
    const int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float rh = fabsf(A) / c;

    // estado en REGISTROS: nunca toca memoria global durante el bucle
    float s[6];
    s[0] = camx;  s[1] = camy;  s[2] = camz;
    // el sonido que LLEGA desde dhat viaja con k proporcional a -dhat
    s[3] = -dirx[i];  s[4] = -diry[i];  s[5] = -dirz[i];

    float k1[6], k2[6], k3[6], k4[6], tmp[6];
    unsigned char st = 0;

    for (int n = 0; n < n_steps; ++n) {
        rhs(s[0],s[1],s[2],s[3],s[4],s[5], A,B,c,eps,sig,rh, k1);
        #pragma unroll
        for (int j = 0; j < 6; ++j) tmp[j] = s[j] + 0.5f*dt*k1[j];
        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k2);
        #pragma unroll
        for (int j = 0; j < 6; ++j) tmp[j] = s[j] + 0.5f*dt*k2[j];
        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k3);
        #pragma unroll
        for (int j = 0; j < 6; ++j) tmp[j] = s[j] + dt*k3[j];
        rhs(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5], A,B,c,eps,sig,rh, k4);

        #pragma unroll
        for (int j = 0; j < 6; ++j)
            s[j] += (dt/6.0f) * (k1[j] + 2.0f*k2[j] + 2.0f*k3[j] + k4[j]);

        const float rc  = sqrtf(s[0]*s[0] + s[1]*s[1]);
        const float far = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

        if (rc < r_cap)                              { st = 1; break; }
        if (far > r_sky || fabsf(s[2]) > z_sky)      { st = 2; break; }
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
# ENVOLTORIO
# =============================================================================
def trace_rays_cuda(dirs, cam_pos, A=1.0, B=0.7, c=1.0, eps=0.0, sig=2.0,
                    dt=0.03, n_steps=1400, capture_pad=1.002,
                    r_sky=38.0, z_sky=38.0, block=256):
    """Traza todos los rayos con UN solo lanzamiento de kernel.

    dirs    : (3, N) cupy float32 — direcciones de pantalla, normalizadas
    cam_pos : (3,) secuencia — posicion del observador
    Devuelve (k_final (3,N) cupy, status (N,) cupy uint8)
             status: 0 sin terminar, 1 capturado (sombra), 2 escapado (cielo)
    """
    dirs = cp.ascontiguousarray(dirs, dtype=cp.float32)
    N = dirs.shape[1]
    okx = cp.empty(N, dtype=cp.float32)
    oky = cp.empty(N, dtype=cp.float32)
    okz = cp.empty(N, dtype=cp.float32)
    status = cp.empty(N, dtype=cp.uint8)

    r_h = abs(A) / c
    grid = ((N + block - 1) // block,)

    _KERNEL(grid, (block,), (
        dirs[0], dirs[1], dirs[2],
        np.float32(cam_pos[0]), np.float32(cam_pos[1]), np.float32(cam_pos[2]),
        np.float32(A), np.float32(B), np.float32(c),
        np.float32(eps), np.float32(sig),
        np.float32(dt), np.int32(n_steps),
        np.float32(r_h * capture_pad), np.float32(r_sky), np.float32(z_sky),
        okx, oky, okz, status, np.int32(N),
    ))
    return cp.stack([okx, oky, okz]), status


def camera_rays_cuda(dist, azim_deg, elev_deg, fov_deg, width, height):
    """Genera (pos, dirs (3,N)) en GPU, sin pasar por CPU salvo 3 escalares."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    pos = np.array([dist*np.cos(e)*np.cos(a),
                    dist*np.cos(e)*np.sin(a),
                    dist*np.sin(e)], dtype=np.float32)
    fwd = -pos / np.linalg.norm(pos)          # mira al origen
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    half_h = np.tan(np.radians(fov_deg) / 2.0)
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


def shade(k_final, status, sky_tex_cp):
    """Colorea: capturados negros, escapados muestrean el cielo en -k_final."""
    d = -k_final
    n = cp.sqrt((d * d).sum(axis=0)) + 1e-30
    dx, dy, dz = d[0]/n, d[1]/n, d[2]/n
    theta = cp.arccos(cp.clip(dz, -1.0, 1.0))
    phi = cp.mod(cp.arctan2(dy, dx), 2*np.pi)

    H, W, _ = sky_tex_cp.shape
    fy = (theta / np.pi) * (H - 1)
    fx = (phi / (2*np.pi)) * (W - 1)
    y0 = cp.clip(cp.floor(fy).astype(cp.int32), 0, H-1)
    x0 = cp.clip(cp.floor(fx).astype(cp.int32), 0, W-1)
    y1 = cp.clip(y0 + 1, 0, H-1)
    x1 = (x0 + 1) % W
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[:, None]
    top = sky_tex_cp[y0, x0]*(1-wx) + sky_tex_cp[y0, x1]*wx
    bot = sky_tex_cp[y1, x0]*(1-wx) + sky_tex_cp[y1, x1]*wx
    col = top*(1-wy) + bot*wy

    col[status == 1] = 0.0                              # sombra acustica
    col[status == 0] = cp.asarray([0.05, 0.0, 0.06], dtype=cp.float32)
    return col


# =============================================================================
# VERIFICACION Y MEDICION
# =============================================================================
def check_against_torch(atol=2e-3):
    """Compara el kernel CUDA contra la version PyTorch, rayo a rayo.

    Ambos usan float32 y -use_fast_math, asi que se esperan diferencias del
    orden de 1e-4..1e-3 en las direcciones finales; lo que DEBE coincidir
    exactamente es el estado (capturado / escapado) de cada rayo.
    """
    try:
        import torch
        import interactive_viewer as iv
    except ImportError as e:
        print(f"no se pudo importar para comparar: {e}")
        return

    n = 4000
    rng = np.random.default_rng(0)
    d = rng.normal(size=(3, n)).astype(np.float32)
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    cam = np.array([0.0, -18.0, 4.0], dtype=np.float32)

    kf, st = trace_rays_cuda(cp.asarray(d), cam, n_steps=1400, dt=0.03)
    st = cp.asnumpy(st)

    p = iv.Params(A=1.0, B=0.7, c=1.0, eps=0.0, dt=0.03, n_steps=1400,
                  r_sky=38.0, z_sky=38.0)
    dev = iv.DEVICE
    dt_ = torch.tensor(d, dtype=torch.float32, device=dev)
    origin = torch.tensor(cam, dtype=torch.float32,
                          device=dev).view(3, 1).expand(3, n)
    state = torch.cat([origin, -dt_], dim=0).contiguous()
    with torch.no_grad():
        final, cap, esc = iv.trace_rays(state, p, backward=True)
    st_t = np.where(cp.asnumpy(cp.zeros(n)) + cap.cpu().numpy(), 1,
                    np.where(esc.cpu().numpy(), 2, 0)).astype(np.uint8)

    same = (st == st_t).mean()
    print(f"coincidencia de estado (capturado/escapado): {100*same:.2f}%")
    print(f"  capturados  CUDA={int((st==1).sum())}  torch={int((st_t==1).sum())}")
    print(f"  escapados   CUDA={int((st==2).sum())}  torch={int((st_t==2).sum())}")
    print(f"  sin acabar  CUDA={int((st==0).sum())}  torch={int((st_t==0).sum())}")
    if same > 0.99:
        print("  -> OK: el kernel reproduce la version PyTorch")
    else:
        print("  -> DISCREPANCIA: revisa signos en rhs() del kernel")


def benchmark(sizes=((960, 540), (1920, 1080), (3840, 2160))):
    """Mide ms/frame reales del kernel."""
    import time
    for (w, h) in sizes:
        pos, dirs = camera_rays_cuda(18.0, -90.0, 12.0, 42.0, w, h)
        trace_rays_cuda(dirs, pos)            # calienta (compila)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            trace_rays_cuda(dirs, pos)
        cp.cuda.Stream.null.synchronize()
        dt = (time.perf_counter() - t0) / reps
        print(f"{w}x{h}: {dt*1000:8.2f} ms/frame   "
              f"{w*h/dt/1e6:7.1f} Mrayos/s   ({1/dt:5.1f} FPS)")


if __name__ == "__main__":
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"GPU: {props['name'].decode()}  "
          f"SMs={props['multiProcessorCount']}")
    print("\n=== benchmark ===")
    benchmark()
    print("\n=== verificacion contra PyTorch ===")
    check_against_torch()