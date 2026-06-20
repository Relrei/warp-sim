"""Scene description — no Warp/Blender dependency."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Domain:
    """Axis-aligned simulation box, origin at (0,0,0)."""
    size: tuple = (8.0, 8.0, 8.0)   # (x, y, z) extents in meters

    @property
    def x(self): return self.size[0]
    @property
    def y(self): return self.size[1]
    @property
    def z(self): return self.size[2]


@dataclass
class Collider:
    """Triangle mesh collider (world space)."""
    vertices:  np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))


@dataclass
class Emitter:
    """Box-shaped particle source."""
    origin:    tuple = (0.0, 0.0, 6.0)
    half_size: tuple = (0.5, 0.5, 0.5)
    velocity:  tuple = (0.0, 0.0, -2.0)
    rate:      int   = 2000              # particles / frame


@dataclass
class SimConfig:
    """All simulation parameters in one place."""

    # Domain
    domain: Domain = field(default_factory=Domain)

    # Particles
    max_particles:    int   = 2_000_000
    smoothing_length: float = 0.10     # h  (meters)
    rest_density:     float = 1000.0   # ρ₀ (kg/m³)
    particle_mass:    float = 0.001    # kg — tune so ρ₀ ≈ rest_density at rest

    # PBF solver
    substeps:          int   = 6
    solver_iterations: int   = 2       # constraint iterations per substep
    epsilon:           float = 100.0   # λ relaxation (prevents div-by-zero)
    scorr_k:           float = 0.001   # tensile instability coefficient
    xsph_c:            float = 0.01    # XSPH viscosity strength

    # Integration
    fps:       float = 30.0
    dt_scale:  float = 1.0
    gravity:   float = -9.8
    v_max_cfl: float = 0.5             # max move per substep (in units of h)

    # Boundaries
    open_boundary: bool  = False       # False = closed box, True = river/waterfall
    kill_margin:   float = 2.0         # extra margin outside domain for EMIT kill

    # Collider
    collide_radius:   float = None     # defaults to 0.5 * smoothing_length
    collide_friction: float = 0.5

    def __post_init__(self):
        if self.collide_radius is None:
            self.collide_radius = 0.5 * self.smoothing_length

    @property
    def dt(self) -> float:
        return self.dt_scale / (self.fps * self.substeps)

    @property
    def v_max(self) -> float:
        return self.v_max_cfl * self.smoothing_length / self.dt
