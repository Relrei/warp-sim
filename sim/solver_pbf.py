"""
Position-Based Fluids solver (Macklin & Müller 2013).
Standalone — no Blender dependency.

Algorithm per substep:
  1. Predict   p* = p + (v + g·dt)·dt
  2. HashGrid  rebuild on p*
  3. Iterate   compute λ → compute Δp → apply Δp + clamp  (×solver_iterations)
  4. Finalize  v = (p* − p) / dt  then XSPH viscosity
Emit / kill / collider run once per frame, outside the CUDA graph.
"""

import math
import numpy as np
import warp as wp

from .scene import SimConfig, Domain, Collider, Emitter


# ---------------------------------------------------------------------------
# Warp helper functions (device)
# ---------------------------------------------------------------------------

@wp.func
def _poly6(r_sq: float, h_sq: float) -> float:
    """Poly6 kernel W(r²) — unnormalized. norm absorbed into particle mass."""
    d = h_sq - r_sq
    if d > 0.0:
        return d * d * d
    return float(0.0)


@wp.func
def _grad_spiky(r: wp.vec3, r_len: float, h: float) -> wp.vec3:
    """Spiky kernel gradient ∇W(r) — unnormalized."""
    if r_len < h and r_len > 1.0e-6:
        d = h - r_len
        return r * (-d * d / r_len)
    return wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _k_predict(
    p:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    pstar:  wp.array(dtype=wp.vec3),
    flag:   wp.array(dtype=int),
    g_dt:   wp.vec3,    # gravity * dt
    dt:     float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        pstar[tid] = wp.vec3(0.0, 0.0, -1.0e6)
        return
    pstar[tid] = p[tid] + (v[tid] + g_dt) * dt


@wp.kernel
def _k_density_lambda(
    grid:        wp.uint64,
    pstar:       wp.array(dtype=wp.vec3),
    rho:         wp.array(dtype=float),
    lam:         wp.array(dtype=float),
    flag:        wp.array(dtype=int),
    h:           float,
    h_sq:        float,
    mass_p6norm: float,   # particle_mass * poly6_norm  (= mass * 315/(64π h^9))
    rest_rho:    float,
    inv_rho0:    float,
    spiky_norm:  float,   # 45/(π h^6)
    eps:         float,
):
    tid = wp.tid()
    i   = wp.hash_grid_point_id(grid, tid)
    if flag[i] == 0:
        rho[i] = rest_rho
        lam[i] = 0.0
        return

    xi      = pstar[i]
    rho_i   = float(0.0)
    grad_ci = wp.vec3(0.0, 0.0, 0.0)
    grad_sq = float(0.0)           # sum of |∇_pⱼ Cᵢ|² for j≠i

    for j in wp.hash_grid_query(grid, xi, h):
        if flag[j] == 0:
            continue
        r    = xi - pstar[j]
        r_sq = wp.dot(r, r)
        if r_sq >= h_sq:
            continue

        rho_i += _poly6(r_sq, h_sq)

        r_len = wp.sqrt(r_sq)
        g_ij  = _grad_spiky(r, r_len, h) * (spiky_norm * inv_rho0)
        if j != i:
            grad_sq += wp.dot(g_ij, g_ij)
        grad_ci += g_ij

    rho_i  *= mass_p6norm
    rho[i]  = rho_i

    c_i      = rho_i * inv_rho0 - 1.0
    grad_sq += wp.dot(grad_ci, grad_ci)
    lam[i]   = -c_i / (grad_sq + eps)


@wp.kernel
def _k_delta_p(
    grid:       wp.uint64,
    pstar:      wp.array(dtype=wp.vec3),
    lam:        wp.array(dtype=float),
    dp:         wp.array(dtype=wp.vec3),
    flag:       wp.array(dtype=int),
    h:          float,
    h_sq:       float,
    spiky_norm: float,
    inv_rho0:   float,
    scorr_k:    float,
    scorr_wdq:  float,   # W_poly6((0.2h)²) unnorm  — for tensile correction
):
    tid = wp.tid()
    i   = wp.hash_grid_point_id(grid, tid)
    if flag[i] == 0:
        dp[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    xi  = pstar[i]
    li  = lam[i]
    dpi = wp.vec3(0.0, 0.0, 0.0)

    for j in wp.hash_grid_query(grid, xi, h):
        if j == i or flag[j] == 0:
            continue
        r    = xi - pstar[j]
        r_sq = wp.dot(r, r)
        if r_sq >= h_sq:
            continue

        r_len = wp.sqrt(r_sq)
        g     = _grad_spiky(r, r_len, h) * spiky_norm

        # Tensile instability correction s_corr
        s = float(0.0)
        if scorr_wdq > 1.0e-14:
            ratio  = _poly6(r_sq, h_sq) / scorr_wdq
            ratio2 = ratio * ratio
            s      = -scorr_k * (ratio2 * ratio2)

        dpi += (li + lam[j] + s) * g

    dp[i] = dpi * inv_rho0


@wp.kernel
def _k_apply_dp(
    pstar: wp.array(dtype=wp.vec3),
    dp:    wp.array(dtype=wp.vec3),
    flag:  wp.array(dtype=int),
    bmin:  wp.vec3,
    bmax:  wp.vec3,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    p = pstar[tid] + dp[tid]
    # clamp to domain (velocity correction happens naturally in _k_finalize)
    pstar[tid] = wp.vec3(
        wp.clamp(p[0], bmin[0], bmax[0]),
        wp.clamp(p[1], bmin[1], bmax[1]),
        wp.clamp(p[2], bmin[2], bmax[2]),
    )


@wp.kernel
def _k_finalize(
    p:     wp.array(dtype=wp.vec3),
    pstar: wp.array(dtype=wp.vec3),
    v:     wp.array(dtype=wp.vec3),
    vtmp:  wp.array(dtype=wp.vec3),   # scratch for XSPH input
    flag:  wp.array(dtype=int),
    inv_dt: float,
    v_max:  float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    vel  = (pstar[tid] - p[tid]) * inv_dt
    spd  = wp.length(vel)
    if spd > v_max:
        vel = vel * (v_max / spd)
    v[tid]    = vel
    vtmp[tid] = vel
    p[tid]    = pstar[tid]


@wp.kernel
def _k_xsph(
    grid:   wp.uint64,
    pstar:  wp.array(dtype=wp.vec3),
    v_in:   wp.array(dtype=wp.vec3),
    v_out:  wp.array(dtype=wp.vec3),
    flag:   wp.array(dtype=int),
    h:      float,
    h_sq:   float,
    xsph_c: float,
):
    tid = wp.tid()
    i   = wp.hash_grid_point_id(grid, tid)
    if flag[i] == 0:
        v_out[i] = v_in[i]
        return
    xi   = pstar[i]
    vi   = v_in[i]
    corr = wp.vec3(0.0, 0.0, 0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j == i or flag[j] == 0:
            continue
        r_sq = wp.dot(xi - pstar[j], xi - pstar[j])
        if r_sq < h_sq:
            w     = _poly6(r_sq, h_sq)
            corr += (v_in[j] - vi) * w
    v_out[i] = vi + xsph_c * corr


@wp.kernel
def _k_emit(
    p:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    flag:   wp.array(dtype=int),
    start:  int,
    total:  int,
    origin: wp.vec3,
    half:   wp.vec3,
    vel:    wp.vec3,
    seed:   int,
):
    t     = wp.tid()
    i     = (start + t) % total
    state = wp.rand_init(seed, i)
    jx    = (wp.randf(state) - 0.5) * 2.0 * half[0]
    jy    = (wp.randf(state) - 0.5) * 2.0 * half[1]
    jz    = (wp.randf(state) - 0.5) * 2.0 * half[2]
    p[i]    = origin + wp.vec3(jx, jy, jz)
    v[i]    = vel
    flag[i] = 1


@wp.kernel
def _k_kill(
    p:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    flag:   wp.array(dtype=int),
    bmin:   wp.vec3,
    bmax:   wp.vec3,
    margin: float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    pos = p[tid]
    if (pos[0] < bmin[0] - margin or pos[0] > bmax[0] + margin or
        pos[1] < bmin[1] - margin or pos[1] > bmax[1] + margin or
        pos[2] < bmin[2] - margin):
        p[tid]    = wp.vec3(0.0, 0.0, -1.0e6)
        v[tid]    = wp.vec3(0.0, 0.0, 0.0)
        flag[tid] = 0


@wp.kernel
def _k_collide(
    p:        wp.array(dtype=wp.vec3),
    v:        wp.array(dtype=wp.vec3),
    flag:     wp.array(dtype=int),
    mesh:     wp.uint64,
    radius:   float,
    max_dist: float,
    friction: float,
):
    tid = wp.tid()
    if flag[tid] == 0:
        return
    pos = p[tid]
    vel = v[tid]
    q   = wp.mesh_query_point_sign_normal(mesh, pos, max_dist)
    if q.result:
        cp   = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        fn   = wp.mesh_eval_face_normal(mesh, q.face)
        cv   = wp.mesh_eval_velocity(mesh, q.face, q.u, q.v)
        d    = pos - cp
        dist = wp.length(d)
        if q.sign < 0.0 or dist < radius:
            if q.sign < 0.0:
                pos = cp + fn * radius
            else:
                n   = fn if dist <= 1.0e-6 else d / dist
                pos = cp + n * radius
            vrel = vel - cv
            vn   = wp.dot(vrel, fn)
            if vn < 0.0:
                vrel = vrel - fn * vn
            vel = cv + vrel * friction
    p[tid] = pos
    v[tid] = vel


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class PBFSolver:
    """
    GPU PBF fluid solver.

    Usage (standalone bake):
        solver = PBFSolver(config)
        solver.reset_dam(fill=(0.4, 1.0, 0.8))
        for frame in range(240):
            solver.step()
            np.save(f"cache/{frame:04d}.npy", solver.positions())
    """

    def __init__(self, config: SimConfig, device: str = "cuda"):
        wp.init()
        self.cfg    = config
        self.device = device
        n           = config.max_particles
        h           = config.smoothing_length

        # GPU buffers (pre-allocated, never reallocated)
        self.p     = wp.zeros(n, dtype=wp.vec3, device=device)
        self.v     = wp.zeros(n, dtype=wp.vec3, device=device)
        self.pstar = wp.zeros(n, dtype=wp.vec3, device=device)
        self.rho   = wp.zeros(n, dtype=float,   device=device)
        self.lam   = wp.zeros(n, dtype=float,   device=device)
        self.dp    = wp.zeros(n, dtype=wp.vec3, device=device)
        self.vtmp  = wp.zeros(n, dtype=wp.vec3, device=device)
        self.flag  = wp.zeros(n, dtype=int,     device=device)

        # HashGrid — size tuned to smoothing length
        gc = max(int(config.domain.x / h) + 2, 32)
        self.grid = wp.HashGrid(gc, gc, gc, device=device)

        # Collider mesh (optional)
        self._col_mesh  = None
        self._col_verts = None
        self._col_idx   = None
        self._col_vel   = None

        # Derived kernel constants
        h2   = h * h
        h9   = h ** 9
        h6   = h ** 6
        p6n  = 315.0 / (64.0 * math.pi * h9)
        spn  = 45.0  / (math.pi * h6)
        dq2  = (0.2 * h) ** 2
        wdq  = max(h2 - dq2, 0.0) ** 3    # poly6(0.2h) unnorm

        self._h        = h
        self._h_sq     = h2
        self._mp6      = config.particle_mass * p6n
        self._rho0     = config.rest_density
        self._inv_rho0 = 1.0 / config.rest_density
        self._spn      = spn
        self._p6n      = p6n
        self._wdq      = wdq
        self._bmin     = wp.vec3(0.0, 0.0, 0.0)
        self._bmax     = wp.vec3(config.domain.x, config.domain.y, config.domain.z)

        # State
        self.frame        = -1
        self.emit_cursor  = 0
        self._graph       = None    # CUDA graph (captured after first step)
        self._graph_ready = False

        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_dam(self, fill: tuple = (0.4, 1.0, 0.8)):
        """Fill a fraction of the domain (DAM break setup)."""
        cfg   = self.cfg
        d     = cfg.domain
        h     = cfg.smoothing_length
        spc   = h * 0.9

        nx = max(1, int(d.x * fill[0] / spc))
        ny = max(1, int(d.y * fill[1] / spc))
        nz = max(1, int(d.z * fill[2] / spc))
        n  = min(nx * ny * nz, cfg.max_particles)

        xs = np.arange(nx, dtype=np.float32) * spc + spc
        ys = np.arange(ny, dtype=np.float32) * spc + spc
        zs = np.arange(nz, dtype=np.float32) * spc + spc
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)[:n].astype(np.float32)

        self.p.zero_()
        self.v.zero_()
        self.flag.zero_()

        p_np   = self.p.numpy()
        fl_np  = self.flag.numpy()
        p_np[:n]  = pos
        fl_np[:n] = 1
        self.p.assign(p_np)
        self.flag.assign(fl_np)

        self.frame        = 0
        self._graph_ready = False
        print(f"[PBFSolver] DAM reset: {n:,} particles")

    def set_collider(self, col: Collider):
        """Upload a triangle mesh collider to the GPU."""
        if col.vertices.shape[0] == 0:
            self._col_mesh = None
            return
        verts = wp.array(col.vertices.astype(np.float32), dtype=wp.vec3, device=self.device)
        idx   = wp.array(col.triangles.astype(np.int32).ravel(), dtype=int, device=self.device)
        vel   = wp.zeros(col.vertices.shape[0], dtype=wp.vec3, device=self.device)
        self._col_verts = verts
        self._col_idx   = idx
        self._col_vel   = vel
        self._col_mesh  = wp.Mesh(verts, idx, vel)

    def step(self, emitters: list[Emitter] | None = None):
        """Advance one frame (= cfg.substeps substeps)."""
        cfg = self.cfg

        # --- emit / kill (outside graph, once per frame) ---
        if emitters:
            for em in emitters:
                self._emit(em)
        if cfg.open_boundary:
            wp.launch(_k_kill, dim=cfg.max_particles,
                      inputs=[self.p, self.v, self.flag,
                               self._bmin, self._bmax, cfg.kill_margin],
                      device=self.device)

        # --- core substeps (via CUDA graph after first frame) ---
        if self._graph_ready:
            wp.capture_launch(self._graph)
        else:
            self._run_substeps()
            # Capture graph on frame 1 (after kernels are JIT-compiled)
            if self.frame == 1:
                self._capture_graph()

        # --- collider (outside graph: mesh may change) ---
        if self._col_mesh is not None:
            r   = cfg.collide_radius
            wp.launch(_k_collide, dim=cfg.max_particles,
                      inputs=[self.p, self.v, self.flag,
                               self._col_mesh.id, r, r * 3.0, cfg.collide_friction],
                      device=self.device)

        wp.synchronize_device(self.device)
        self.frame += 1

    def positions(self) -> np.ndarray:
        """Return (N_live, 3) float32 array of live particle positions (CPU)."""
        flags = self.flag.numpy()
        pos   = self.p.numpy()
        return pos[flags == 1]

    def all_positions(self) -> np.ndarray:
        """Return full (max_particles, 3) buffer (dead particles at -1e6)."""
        return self.p.numpy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_substeps(self):
        cfg     = self.cfg
        n       = cfg.max_particles
        h       = self._h
        h_sq    = self._h_sq
        dt      = cfg.dt
        inv_dt  = 1.0 / dt
        g_dt    = wp.vec3(0.0, 0.0, cfg.gravity * dt)
        bmin    = self._bmin
        bmax    = self._bmax
        dev     = self.device

        for _ in range(cfg.substeps):
            # 1. Predict
            wp.launch(_k_predict, dim=n,
                      inputs=[self.p, self.v, self.pstar, self.flag, g_dt, dt],
                      device=dev)

            # 2. Grid
            self.grid.build(self.pstar, h)

            # 3. Constraint iterations
            for _ in range(cfg.solver_iterations):
                wp.launch(_k_density_lambda, dim=n,
                          inputs=[self.grid.id, self.pstar, self.rho, self.lam, self.flag,
                                  h, h_sq, self._mp6, self._rho0, self._inv_rho0,
                                  self._spn, cfg.epsilon],
                          device=dev)
                wp.launch(_k_delta_p, dim=n,
                          inputs=[self.grid.id, self.pstar, self.lam, self.dp, self.flag,
                                  h, h_sq, self._spn, self._inv_rho0,
                                  cfg.scorr_k, self._wdq],
                          device=dev)
                wp.launch(_k_apply_dp, dim=n,
                          inputs=[self.pstar, self.dp, self.flag, bmin, bmax],
                          device=dev)

            # 4. Finalize velocity
            wp.launch(_k_finalize, dim=n,
                      inputs=[self.p, self.pstar, self.v, self.vtmp, self.flag,
                               inv_dt, cfg.v_max],
                      device=dev)

            # 5. XSPH viscosity
            wp.launch(_k_xsph, dim=n,
                      inputs=[self.grid.id, self.pstar, self.vtmp, self.v, self.flag,
                               h, h_sq, cfg.xsph_c],
                      device=dev)

    def _capture_graph(self):
        """Capture the substep loop as a CUDA graph for replay."""
        try:
            wp.capture_begin(device=self.device)
            self._run_substeps()
            self._graph       = wp.capture_end(device=self.device)
            self._graph_ready = True
            print("[PBFSolver] CUDA graph captured — subsequent frames use fast replay")
        except Exception as e:
            print(f"[PBFSolver] CUDA graph capture failed ({e}), running without graph")

    def _emit(self, em: Emitter):
        n      = self.cfg.max_particles
        start  = self.emit_cursor % n
        seed   = self.frame * 997 + start + 1
        origin = wp.vec3(*em.origin)
        half   = wp.vec3(*em.half_size)
        vel    = wp.vec3(*em.velocity)
        wp.launch(_k_emit, dim=em.rate,
                  inputs=[self.p, self.v, self.flag,
                           start, n, origin, half, vel, seed],
                  device=self.device)
        self.emit_cursor = (self.emit_cursor + em.rate) % n

    def _warmup(self):
        """JIT-compile all kernels with tiny dummy data."""
        n  = 8
        h  = self._h
        g  = wp.HashGrid(4, 4, 4, device=self.device)
        px = wp.array(np.random.rand(n, 3).astype(np.float32) * h,
                      dtype=wp.vec3, device=self.device)
        v0 = wp.zeros(n, dtype=wp.vec3, device=self.device)
        f1 = wp.array(np.ones(n, np.int32), dtype=int, device=self.device)
        f0 = wp.zeros(n, dtype=int, device=self.device)
        rh = wp.zeros(n, dtype=float, device=self.device)
        lm = wp.zeros(n, dtype=float, device=self.device)
        dp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        ps = wp.array(np.random.rand(n, 3).astype(np.float32) * h,
                      dtype=wp.vec3, device=self.device)
        vt = wp.zeros(n, dtype=wp.vec3, device=self.device)
        g.build(px, h)

        bmin = wp.vec3(0.0, 0.0, 0.0)
        bmax = wp.vec3(1.0, 1.0, 1.0)
        dt   = self.cfg.dt

        wp.launch(_k_predict,        dim=n, inputs=[px, v0, ps, f1, wp.vec3(0,0,-0.1), dt], device=self.device)
        wp.launch(_k_density_lambda, dim=n, inputs=[g.id, ps, rh, lm, f1, h, h*h,
                                                     self._mp6, self._rho0, self._inv_rho0,
                                                     self._spn, self.cfg.epsilon], device=self.device)
        wp.launch(_k_delta_p,        dim=n, inputs=[g.id, ps, lm, dp, f1, h, h*h,
                                                     self._spn, self._inv_rho0,
                                                     self.cfg.scorr_k, self._wdq], device=self.device)
        wp.launch(_k_apply_dp,       dim=n, inputs=[ps, dp, f1, bmin, bmax], device=self.device)
        wp.launch(_k_finalize,       dim=n, inputs=[px, ps, v0, vt, f1, 1.0/dt, self.cfg.v_max], device=self.device)
        wp.launch(_k_xsph,           dim=n, inputs=[g.id, ps, vt, v0, f1, h*h, h*h, 0.01], device=self.device)
        wp.launch(_k_emit,           dim=n, inputs=[px, v0, f0, 0, n,
                                                     wp.vec3(0.5,0.5,0.5), wp.vec3(0.1,0.1,0.1),
                                                     wp.vec3(0,0,-1), 1], device=self.device)
        wp.launch(_k_kill,           dim=n, inputs=[px, v0, f0, bmin, bmax, 1.0], device=self.device)
        # collide: compile with a minimal mesh
        mv = wp.array(np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32), dtype=wp.vec3, device=self.device)
        mi = wp.array(np.array([0,1,2], np.int32), dtype=int, device=self.device)
        me = wp.Mesh(mv, mi, wp.zeros(3, dtype=wp.vec3, device=self.device))
        wp.launch(_k_collide, dim=n, inputs=[px, v0, f1, me.id, 0.05, 0.2, 0.5], device=self.device)

        wp.synchronize_device(self.device)
        print(f"[PBFSolver] kernels compiled  h={h}  max_n={self.cfg.max_particles:,}")
