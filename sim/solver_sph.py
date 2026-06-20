"""
WCSPH solver — migrated from warp_water_fluid.py (Blender addon).
Kept for benchmarking against PBF.  No Blender dependency.

Original: Müller 2003 WCSPH, kernels borrowed from NVIDIA Warp example_sph.py.
"""

import math
import numpy as np
import warp as wp

from .scene import SimConfig, Domain, Collider, Emitter


# ---------------------------------------------------------------------------
# Warp kernel functions
# ---------------------------------------------------------------------------

@wp.func
def _sq(x: float): return x * x
@wp.func
def _cube(x: float): return x * x * x

@wp.func
def _density_kernel(xyz: wp.vec3, h: float):
    d = wp.dot(xyz, xyz)
    return wp.max(_cube(_sq(h) - d), 0.0)

@wp.func
def _diff_pressure(xyz: wp.vec3, p: float, np_: float, nrho: float, h: float):
    dist = wp.sqrt(wp.dot(xyz, xyz))
    if dist < h and dist > 1.0e-6:
        t1 = -xyz / dist
        t2 = (np_ + p) / (2.0 * nrho)
        t3 = _sq(h - dist)
        return t1 * t2 * t3
    return wp.vec3()

@wp.func
def _diff_visc(xyz: wp.vec3, v: wp.vec3, nv: wp.vec3, nrho: float, h: float):
    dist = wp.sqrt(wp.dot(xyz, xyz))
    if dist < h:
        return (nv - v) / nrho * (h - dist)
    return wp.vec3()


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _k_density(
    grid: wp.uint64,
    px: wp.array(dtype=wp.vec3),
    prho: wp.array(dtype=float),
    flag: wp.array(dtype=int),
    dnorm: float,
    h: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if flag[i] == 0:
        prho[i] = 1.0
        return
    x = px[i]
    rho = float(0.0)
    for idx in wp.hash_grid_query(grid, x, h):
        rho += _density_kernel(x - px[idx], h)
    prho[i] = dnorm * rho


@wp.kernel
def _k_accel(
    grid: wp.uint64,
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    prho: wp.array(dtype=float),
    pa: wp.array(dtype=wp.vec3),
    flag: wp.array(dtype=int),
    iso_exp: float,
    base_rho: float,
    gravity: float,
    pnorm: float,
    vnorm: float,
    h: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if flag[i] == 0:
        return
    x = px[i]
    v = pv[i]
    rho = prho[i]
    pressure = iso_exp * (rho - base_rho)
    pf = wp.vec3()
    vf = wp.vec3()
    for idx in wp.hash_grid_query(grid, x, h):
        if idx != i:
            nrho = prho[idx]
            npr = iso_exp * (nrho - base_rho)
            rel = px[idx] - x
            pf += _diff_pressure(rel, pressure, npr, nrho, h)
            vf += _diff_visc(rel, v, pv[idx], nrho, h)
    pa[i] = pnorm * pf + vnorm * vf + wp.vec3(0.0, 0.0, gravity)


@wp.kernel
def _k_bounds(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    flag: wp.array(dtype=int),
    damp: float,
    wx: float, wy: float, wz: float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    x = px[tid]
    v = pv[tid]
    if x[0] < 0.0: x = wp.vec3(0.0, x[1], x[2]); v = wp.vec3(v[0]*damp, v[1], v[2])
    if x[0] > wx:  x = wp.vec3(wx,  x[1], x[2]); v = wp.vec3(v[0]*damp, v[1], v[2])
    if x[1] < 0.0: x = wp.vec3(x[0], 0.0, x[2]); v = wp.vec3(v[0], v[1]*damp, v[2])
    if x[1] > wy:  x = wp.vec3(x[0], wy,  x[2]); v = wp.vec3(v[0], v[1]*damp, v[2])
    if x[2] < 0.0: x = wp.vec3(x[0], x[1], 0.0); v = wp.vec3(v[0], v[1], v[2]*damp)
    if x[2] > wz:  x = wp.vec3(x[0], x[1], wz);  v = wp.vec3(v[0], v[1], v[2]*damp)
    px[tid] = x
    pv[tid] = v


@wp.kernel
def _k_kick(pv: wp.array(dtype=wp.vec3), pa: wp.array(dtype=wp.vec3),
            flag: wp.array(dtype=int), dt: float):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    pv[tid] = pv[tid] + pa[tid] * dt


@wp.kernel
def _k_clamp(pv: wp.array(dtype=wp.vec3), vmax: float):
    tid = wp.tid()
    v = pv[tid]
    sp = wp.length(v)
    if sp > vmax:
        pv[tid] = v * (vmax / sp)


@wp.kernel
def _k_drift(px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
             flag: wp.array(dtype=int), dt: float):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    px[tid] = px[tid] + pv[tid] * dt


@wp.kernel
def _k_emit(
    px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
    flag: wp.array(dtype=int),
    start: int, total: int,
    origin: wp.vec3, half: wp.vec3, vel: wp.vec3, seed: int,
):
    t = wp.tid()
    i = (start + t) % total
    state = wp.rand_init(seed, i)
    jx = (wp.randf(state) - 0.5) * 2.0 * half[0]
    jy = (wp.randf(state) - 0.5) * 2.0 * half[1]
    jz = (wp.randf(state) - 0.5) * 2.0 * half[2]
    px[i] = origin + wp.vec3(jx, jy, jz)
    pv[i] = vel
    flag[i] = 1


@wp.kernel
def _k_kill(
    px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
    flag: wp.array(dtype=int),
    bmin: wp.vec3, bmax: wp.vec3, margin: float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    p = px[tid]
    if (p[0] < bmin[0] - margin or p[0] > bmax[0] + margin or
        p[1] < bmin[1] - margin or p[1] > bmax[1] + margin or
        p[2] < bmin[2] - margin):
        flag[tid] = 0
        px[tid] = wp.vec3(0.0, 0.0, -1.0e6)
        pv[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _k_collide(
    px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
    flag: wp.array(dtype=int),
    mesh: wp.uint64, radius: float, max_dist: float, friction: float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    p = px[tid]; v = pv[tid]
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        fn = wp.mesh_eval_face_normal(mesh, q.face)
        cv = wp.mesh_eval_velocity(mesh, q.face, q.u, q.v)
        delta = p - cp; dist = wp.length(delta)
        if q.sign < 0.0 or dist < radius:
            p = cp + fn * radius if q.sign < 0.0 else cp + (fn if dist <= 1e-6 else delta/dist) * radius
            vrel = v - cv; vn = wp.dot(vrel, fn)
            if vn < 0.0: vrel = vrel - fn * vn
            v = cv + vrel * friction
    px[tid] = p; pv[tid] = v


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class SPHSolver:
    """Legacy WCSPH solver — use PBFSolver for better performance."""

    def __init__(self, config: SimConfig, device: str = "cuda"):
        wp.init()
        self.cfg    = config
        self.device = device
        h           = config.smoothing_length
        n           = config.max_particles
        d           = config.domain

        self.x    = wp.zeros(n, dtype=wp.vec3, device=device)
        self.v    = wp.zeros(n, dtype=wp.vec3, device=device)
        self.rho  = wp.zeros(n, dtype=float,   device=device)
        self.a    = wp.zeros(n, dtype=wp.vec3, device=device)
        self.flag = wp.zeros(n, dtype=int,     device=device)

        gc         = max(int(max(d.x, d.y, d.z) / h) + 2, 4)
        self.grid  = wp.HashGrid(gc, gc, gc, device=device)
        self._col_mesh = None

        m          = config.particle_mass
        self._dnorm = (315.0 * m) / (64.0 * math.pi * h ** 9)
        self._pnorm = -(45.0 * m) / (math.pi * h ** 6)
        self._vnorm = (45.0 * 0.025 * m) / (math.pi * h ** 6)  # visc fixed at 0.025
        self._bmin  = wp.vec3(0.0, 0.0, 0.0)
        self._bmax  = wp.vec3(d.x, d.y, d.z)

        self.frame        = -1
        self.emit_cursor  = 0

    def reset_dam(self, fill: tuple = (0.4, 1.0, 0.8)):
        cfg = self.cfg; h = cfg.smoothing_length; d = cfg.domain
        sp  = h
        nx  = max(1, int(d.x * fill[0] / sp))
        ny  = max(1, int(d.y * fill[1] / sp))
        nz  = max(1, int(d.z * fill[2] / sp))
        n   = min(nx * ny * nz, cfg.max_particles)
        xs  = (np.arange(nx) + 0.5) * sp
        ys  = (np.arange(ny) + 0.5) * sp
        zs  = (np.arange(nz) + 0.5) * sp + sp
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)[:n].astype(np.float32)
        self.x.zero_(); self.v.zero_(); self.flag.zero_()
        pn = self.x.numpy(); fn = self.flag.numpy()
        pn[:n] = pos; fn[:n] = 1
        self.x.assign(pn); self.flag.assign(fn)
        self.frame = 0

    def step(self, emitters=None):
        cfg  = self.cfg; h = cfg.smoothing_length
        n    = cfg.max_particles
        dt   = cfg.dt
        vmax = cfg.v_max
        dev  = self.device
        stiffness  = 20.0
        base_rho   = cfg.rest_density
        steps      = cfg.substeps

        if emitters:
            for em in emitters:
                start = self.emit_cursor % n
                wp.launch(_k_emit, dim=em.rate,
                          inputs=[self.x, self.v, self.flag, start, n,
                                  wp.vec3(*em.origin), wp.vec3(*em.half_size),
                                  wp.vec3(*em.velocity), self.frame * 997 + start],
                          device=dev)
                self.emit_cursor = (self.emit_cursor + em.rate) % n

        for _ in range(steps):
            self.grid.build(self.x, h)
            wp.launch(_k_density, dim=n,
                      inputs=[self.grid.id, self.x, self.rho, self.flag, self._dnorm, h],
                      device=dev)
            wp.launch(_k_accel, dim=n,
                      inputs=[self.grid.id, self.x, self.v, self.rho, self.a, self.flag,
                               stiffness, base_rho, cfg.gravity, self._pnorm, self._vnorm, h],
                      device=dev)
            if not cfg.open_boundary:
                d = cfg.domain
                wp.launch(_k_bounds, dim=n,
                          inputs=[self.x, self.v, self.flag, -0.5, d.x, d.y, d.z],
                          device=dev)
            wp.launch(_k_kick,  dim=n, inputs=[self.v, self.a, self.flag, dt], device=dev)
            wp.launch(_k_clamp, dim=n, inputs=[self.v, vmax], device=dev)
            wp.launch(_k_drift, dim=n, inputs=[self.x, self.v, self.flag, dt], device=dev)
            if self._col_mesh:
                r = h * 0.5
                wp.launch(_k_collide, dim=n,
                          inputs=[self.x, self.v, self.flag, self._col_mesh.id, r, r*3, 0.5],
                          device=dev)

        if cfg.open_boundary:
            wp.launch(_k_kill, dim=n,
                      inputs=[self.x, self.v, self.flag,
                               self._bmin, self._bmax, cfg.kill_margin],
                      device=dev)

        wp.synchronize_device(dev)
        self.frame += 1

    def positions(self):
        flags = self.flag.numpy()
        return self.x.numpy()[flags == 1]
