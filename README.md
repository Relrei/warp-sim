# warp-sim

GPU-accelerated fluid simulation pipeline for cinematic VFX, powered by [NVIDIA Warp](https://github.com/NVIDIA/warp).

## Overview

```
Warp (GPU simulation) → .npy cache → Falcor (path-traced render)
```

- **Solver**: Position-Based Fluids (PBF) — stable at 5–8 substeps vs. 40+ for WCSPH
- **Scale**: 2M–10M particles on RTX 3080 (10 GB VRAM)
- **Output**: Baked `.npy` frame cache → surface reconstruction → Falcor
- **Rendering**: Spectral photon mapping (SPPM), IOR 1.33, Beer–Lambert, chromatic caustics

No Blender dependency at simulation time. Blender (or any DCC) can import the baked cache for scene composition.

## Requirements

- NVIDIA GPU (RTX series recommended)
- [NVIDIA Warp](https://github.com/NVIDIA/warp): `pip install warp-lang`

```bash
pip install warp-lang numpy
```

## Project Structure

```
sim/
  solver_pbf.py   # PBF solver — density constraint + CUDA Graph
  solver_sph.py   # Legacy WCSPH (benchmark comparison)
  bake.py         # Frame loop → .npy cache
  scene.py        # SimConfig, Domain, Collider, Emitter

falcor/           # Falcor render bridge (Phase 3)
  surface.py      # .npy frames → mesh / VDB
  fluid_pass.py   # Custom render pass
  material.py     # Water IOR, SPPM caustics

examples/
  dam_break.py    # 2M particle dam break
  waterfall.py    # Open boundary emitter
```

## Quickstart

```bash
pip install warp-lang numpy
python examples/dam_break.py --particles 2000000 --frames 240
```

Output: `cache/dam_break/frame_0000.npy` … (N_live × 3 float32)

## Roadmap

### Phase 1 — PBF Core + CUDA Graph ✅
- PBF solver (Macklin & Müller 2013) — 8× fewer substeps than WCSPH
- CUDA Graph capture: full substep loop replayed in 1 GPU call/frame
- Standalone bake to `.npy` — no Blender at sim time
- Legacy WCSPH kept in `solver_sph.py` for benchmarking

### Phase 2 — Falcor Bridge
- Surface reconstruction: `.npy` → Marching Cubes → `.usdc`
- USD time-sampled water mesh loaded into Falcor
- PBR water material: IOR 1.33, transmission, Beer–Lambert tint

### Phase 3 — Caustics
- Spectral photon mapping (SPPM)
- Chromatic dispersion: Cauchy n = A + B/λ²
- 3-knob control: Dispersion Gain / Saturation / Mask
- AOV split: base + caustics → composite

### Phase 4 — Scale + FX
- 5M–10M particle support (VRAM pool optimization)
- Foam / spray / bubbles
- NanoVDB surface reconstruction

## License

MIT
