# warp-sim

GPU-accelerated fluid simulation for Blender and Falcor, powered by [NVIDIA Warp](https://github.com/NVIDIA/warp).

## Overview

A high-performance SPH/PBF fluid simulation pipeline designed for cinematic VFX:

```
Blender (scene setup) → Warp (GPU simulation) → Falcor (path-traced render)
```

- **Solver**: Position-Based Fluids (PBF) — stable at 5–8 substeps vs. 40+ for WCSPH
- **Scale**: 2M–10M particles on RTX 3080 (10 GB VRAM)
- **Output**: Baked `.npz` / `.usdc` cache → Marching Cubes surface → Falcor
- **Rendering**: Spectral photon mapping (SPPM), IOR 1.33, Beer–Lambert, chromatic caustics

## Requirements

- NVIDIA GPU (RTX series recommended)
- [NVIDIA Warp](https://github.com/NVIDIA/warp) (`pip install warp-lang`)
- Blender 4.x / 5.x (for the addon)
- [Falcor](https://github.com/NVIDIAGameWorks/Falcor) (for final rendering)

## Project Structure

```
sim/          # Core solver — no Blender dependency
  solver_pbf.py     # PBF density-constraint solver
  solver_sph.py     # Legacy WCSPH (for comparison)
  cuda_graph.py     # CUDA Graph wrapper (substep capture)
  bake.py           # Bake to .npz / .usdc
  scene.py          # Domain, colliders, emitters

blender/      # Thin Blender addon (I/O only)
  __init__.py
  ops.py            # Bake operator, scene export
  props.py          # UI parameters
  import_cache.py   # Load .npz → point cloud → GN surface

falcor/       # Falcor render bridge
  surface.py        # .npz frames → VDB / mesh
  fluid_pass.py     # Custom render pass
  material.py       # Water material (IOR, SPPM caustics)

examples/
  dam_break.py
  waterfall.py
  river.py
```

## Roadmap

See [docs/PLAN.md](docs/PLAN.md) for the full phased plan.

### Phase 1 — PBF Core + CUDA Graph
- Replace WCSPH with PBF solver (8× fewer substeps)
- Capture substep loop with CUDA Graph (1 Python call per frame)
- Standalone bake to `.npz` (no Blender at sim time)
- **Target**: 2M particles @ 25–40 FPS on RTX 3080

### Phase 2 — Blender Addon
- Scene/collider export (JSON/USD)
- Bake trigger from Blender UI
- Cache import → Geometry Nodes surface

### Phase 3 — Falcor Bridge
- Surface reconstruction (Marching Cubes / NanoVDB)
- USD time-sampled water mesh → Falcor
- SPPM caustics, chromatic dispersion (Cauchy n=A+B/λ²)

### Phase 4 — Scale + FX
- 5M–10M particle support
- Open boundary (river, waterfall)
- Foam / spray / bubbles

## License

MIT
