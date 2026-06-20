# warp-sim — Implementation Plan

## Architecture

```
Blender (scene design only)
  ↓ export colliders / domain as JSON
sim/solver_pbf.py  (standalone GPU core)
  ↓ bake to .npz frame cache
falcor/surface.py  (surface reconstruction)
  ↓ .usdc time-sampled mesh
Falcor (path-trace: refraction + SPPM caustics)
```

Blender is never running during simulation. It only provides scene input and displays the result.

---

## Phase 1 — PBF Core + CUDA Graph

**Goal**: 2M particles @ 25–40 FPS on RTX 3080 (currently 4 FPS with WCSPH)

### Why PBF over WCSPH

WCSPH needs `Steps/Frame = 40` because high stiffness makes the system numerically stiff.  
PBF solves incompressibility via a density constraint directly on positions — stable at `Steps = 5–8`.

| | WCSPH (current) | PBF (target) |
|---|---|---|
| Substeps/frame | 40 | 5–8 |
| Stability | Explodes at high stiffness | Unconditionally stable |
| 2M particle FPS | ~4 | ~25–40 (estimated) |

### PBF Algorithm (Macklin & Müller 2013)

Each substep:
1. Predict positions: `p* = p + v*dt`
2. Build HashGrid on `p*`
3. Compute density `ρᵢ = Σⱼ mⱼ W(pᵢ* - pⱼ*, h)` (Poly6 kernel)
4. Compute lambda: `λᵢ = -(ρᵢ/ρ₀ - 1) / (Σₖ |∇Cᵢₖ|²/ρ₀² + ε)`
5. Compute delta-p: `Δpᵢ = (1/ρ₀) Σⱼ (λᵢ + λⱼ) ∇W(pᵢ* - pⱼ*, h)` (Spiky kernel grad)
6. Apply: `p*ᵢ += Δpᵢ`  ← repeat 3–6 for N iterations
7. Update velocity: `vᵢ = (p*ᵢ - pᵢ) / dt`
8. XSPH viscosity (optional): `vᵢ += c Σⱼ (vⱼ - vᵢ) W(...)`
9. Bounds / collider / emit / kill

### CUDA Graph

Without CUDA Graph, each `wp.launch()` call incurs Python→CUDA dispatch overhead.  
With `wp.capture_begin/end`, the full substep loop is recorded once and replayed in 1 call.

```python
# First frame: capture
with wp.ScopedCapture(device=device) as cap:
    for _ in range(substeps):
        _pbf_substep(...)
self.graph = cap.graph

# Every subsequent frame: replay
wp.capture_launch(self.graph)
```

### Files to create

- `sim/solver_pbf.py` — all Warp kernels + `PBFSolver` class
- `sim/cuda_graph.py` — graph capture/replay wrapper
- `sim/bake.py` — frame loop → `cache/frame_NNNN.npz`
- `sim/scene.py` — `Domain`, `Collider`, `Emitter` dataclasses
- `examples/dam_break.py` — standalone demo (no Blender)

---

## Phase 2 — Falcor Bridge

**Goal**: Baked water surface renders with refractive caustics in Falcor

- `falcor/surface.py`
  - Load `cache/frame_NNNN.npz` (particle positions)
  - Marching Cubes → triangle mesh per frame
  - Write as USD time-sampled geometry (`.usdc`)
- `falcor/fluid_pass.py`
  - Load `.usdc` into existing FalcorBridge scene
  - Animate mesh per frame via transform/vertex update
- `falcor/material.py`
  - Water: IOR 1.33, transmission 1.0, Beer–Lambert tint
  - SPPM caustics: Cauchy dispersion n=A+B/λ², 3-knob control
  - AOV split: base + caustics → separate EXR → composite

Reference: `~/warp-blender/docs/HANDOFF.md` (full optical design notes)

---

## Phase 3 — Scale + FX

- **5M–10M particles**: VRAM pool sizing, dead-particle compaction
- **Open boundary**: ring-buffer emitter improvements for river/waterfall
- **Foam/Spray**: particles exceeding velocity threshold → secondary billboards
- **NanoVDB**: faster surface reconstruction than CPU Marching Cubes

---

## Performance Targets (RTX 3080, 10 GB)

| Config | Particles | FPS (interactive) | Time/frame (bake) |
|---|---|---|---|
| Current (WCSPH, addon) | 2M | 4 | 250 ms |
| Phase 1 (PBF, standalone) | 2M | 25–40 | 25–40 ms |
| Phase 1 (PBF, standalone) | 5M | 8–15 | 65–125 ms |
| Phase 4 (PBF + compaction) | 10M | 4–8 | 125–250 ms |
