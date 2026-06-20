"""
DAM break — standalone bake demo.

Usage:
    python examples/dam_break.py
    python examples/dam_break.py --particles 500000 --frames 120
"""

import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from sim.scene   import SimConfig, Domain
from sim.solver_pbf import PBFSolver
from sim.bake    import bake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int,   default=2_000_000)
    ap.add_argument("--frames",    type=int,   default=240)
    ap.add_argument("--h",         type=float, default=0.10,
                    help="Smoothing length (smaller = more particles, slower)")
    ap.add_argument("--substeps",  type=int,   default=6)
    ap.add_argument("--out",       type=str,   default="cache/dam_break")
    args = ap.parse_args()

    cfg = SimConfig(
        domain         = Domain(size=(8.0, 4.0, 8.0)),
        max_particles  = args.particles,
        smoothing_length = args.h,
        substeps       = args.substeps,
        fps            = 30.0,
        gravity        = -9.8,
        open_boundary  = False,
    )

    solver = PBFSolver(cfg)
    # fill left-front corner (40 % x, 100 % y, 80 % z)
    solver.reset_dam(fill=(0.4, 1.0, 0.8))

    print(f"\n--- DAM BREAK ---")
    print(f"  domain    : {cfg.domain.size}")
    print(f"  particles : {args.particles:,}")
    print(f"  substeps  : {args.substeps}  solver_iters: {cfg.solver_iterations}")
    print(f"  frames    : {args.frames}")
    print(f"  output    : {args.out}/\n")

    bake(solver, n_frames=args.frames, output_dir=args.out)


if __name__ == "__main__":
    main()
