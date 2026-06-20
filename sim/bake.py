"""Bake a PBF simulation to a .npy frame cache (no Blender, no display)."""

import os
import time
import numpy as np

from .solver_pbf import PBFSolver
from .scene import Emitter


def bake(
    solver:     PBFSolver,
    n_frames:   int,
    output_dir: str,
    emitters:   list[Emitter] | None = None,
    verbose:    bool = True,
) -> None:
    """
    Run `n_frames` of simulation and save each frame as:
        <output_dir>/frame_NNNN.npy   shape (N_live, 3)  float32

    Only live particles are saved (dead particles excluded).
    """
    os.makedirs(output_dir, exist_ok=True)

    t_total = 0.0
    for frame in range(n_frames):
        t0 = time.perf_counter()
        solver.step(emitters=emitters)
        dt = time.perf_counter() - t0
        t_total += dt

        pos = solver.positions()
        np.save(os.path.join(output_dir, f"frame_{frame:04d}.npy"), pos)

        if verbose:
            fps  = 1.0 / max(dt, 1e-9)
            avg  = t_total / (frame + 1)
            eta  = avg * (n_frames - frame - 1)
            print(f"  frame {frame+1:4d}/{n_frames}  "
                  f"n={len(pos):>9,}  "
                  f"{fps:5.1f} fps  "
                  f"eta {eta/60:.1f} min")

    if verbose:
        print(f"\nBake done — {n_frames} frames in {t_total:.1f}s "
              f"({t_total/n_frames*1000:.1f} ms/frame avg)")
        print(f"Cache: {output_dir}/")


def load_frame(output_dir: str, frame: int) -> np.ndarray:
    """Load a baked frame.  Returns (N, 3) float32."""
    return np.load(os.path.join(output_dir, f"frame_{frame:04d}.npy"))


def frame_count(output_dir: str) -> int:
    """Count how many frames are in a cache directory."""
    return len([f for f in os.listdir(output_dir) if f.endswith(".npy")])
