"""
Waterfall — open boundary emitter demo.

Usage:
    python examples/waterfall.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from sim.scene      import SimConfig, Domain, Emitter
from sim.solver_pbf import PBFSolver
from sim.bake       import bake


def main():
    cfg = SimConfig(
        domain         = Domain(size=(6.0, 6.0, 10.0)),
        max_particles  = 1_000_000,
        smoothing_length = 0.10,
        substeps       = 6,
        fps            = 30.0,
        gravity        = -9.8,
        open_boundary  = True,
        kill_margin    = 1.0,
    )

    emitters = [
        Emitter(
            origin    = (3.0, 3.0, 9.0),
            half_size = (1.0, 1.0, 0.1),
            velocity  = (0.0, 0.0, -3.0),
            rate      = 3000,
        )
    ]

    solver = PBFSolver(cfg)
    # start with empty domain
    solver.frame = 0

    print("\n--- WATERFALL ---")
    bake(solver, n_frames=300, output_dir="cache/waterfall", emitters=emitters)


if __name__ == "__main__":
    main()
