# GPU Acceleration Plan for pimpleFoam (OpenFOAM v10 — dev mirror)

This document captures the roadmap to evolve the current GPU-enabled k–ε turbulence model into a broader GPU acceleration for pimpleFoam while preserving CPU APIs and behavior. It is intended for the development mirror at `/home/xiaozhou/OpenFOAM/openfoam10-gpu` and complements AGENTS.md.

## Scope & Principles

- Preserve public APIs and CPU behavior; add GPU as additive runtime-selectable backends.
- CPU remains the reference; maintain numerical parity against CPU pitzDaily.
- Work only in this dev mirror; deploy to `/opt/openfoam10` only after gates pass and with approval.
- Keep changes minimal and focused; conventional commits; no artefacts in VCS.

## Environments & Baseline

- Dev mirror: `/home/xiaozhou/OpenFOAM/openfoam10-gpu` (branch `gpu`).
- CPU baseline case: `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/pitzDaily_cpu`.
- Clean template: `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/pitzDaily_clean`.
- Scripts: `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts`.
- Existing GPU: k–ε `nut` update has an optional CUDA path (NVRTC JIT) with CPU fallback.

## Success Criteria & Gates

- Build completes without errors in this mirror.
  - Note: PVReaders is optional in dev; use `WM_NOCMAKE=on` (or skip PVReaders) to keep dev builds green.
- pitzDaily GPU trial runs to completion from a clean case using this mirror environment.
- Numerical parity: relative L2(Ux) at last time step ≤ 1e-2 (threshold configurable by scripts).
- Deployment to `/opt/openfoam10` only after all gates pass and with explicit approval.

## Measurement & Tooling

- Use `tools/check_gpu_toolchain.sh` to enforce CUDA/HIP presence before GPU builds/runs.
- Build wrapper: `tools/build_all.sh` (captures logs under `/home/xiaozhou/OpenFOAM/build_logs/`).
- Run/compare: `run/scripts/run_gpu_trial.sh` + `Calculate_error.py` (auto-picks last time and checks relative L2 with optional per-component stats).

## Phased Roadmap

### Phase 0 — Foundations

- Conditional CUDA build flags (detect `CUDA_HOME` or nvcc; gate `FOAM_USE_CUDA`).
- Keep PVReaders off in dev builds (e.g., export `WM_NOCMAKE=on` in build script) to avoid unrelated CMake errors.
- Augment scripts to record timings and iteration counts for U/p solves.
- Ensure CPU fallback paths are robust and warning-rich.

Deliverables:
- Make/options conditionals; optional environment toggles (no-op on CPU-only hosts).
- Build script minor patch (optional): early toolchain check; skip PVReaders.

Acceptance:
- Mirror builds cleanly for core libs; PVReaders optionally skipped.

### Phase 1 — GPU Linear Solver Offload (single GPU)

Goal: Offload linear solves for `(U|p|k|epsilon)` while keeping assembly on CPU; transfer matrices/vectors to GPU each iteration.

Core tasks:
- ldu→CSR exporter: build once per mesh change; update coefficients per solve.
- Implement a GPU solver (default): Custom CUDA PCG/BiCGStab with Jacobi/ILU0 using cuSPARSE/cuSOLVER.
  - Optional later: AmgX wrapper as an alternative runtime‑selectable backend.
- Runtime selection: add a new `lduMatrix::solver` type (e.g., `cudaPCG` or `amgx`) selectable via `system/fvSolution`; CPU fallback retained.
- Logging/timing around solver calls.

Deliverables:
- New solver under `src/OpenFOAM/matrices/lduMatrix/solvers/cudaPCG/` (gated by `FOAM_USE_CUDA`).
- CSR export utilities (mapping indices and data views) with unit-level validation on small matrices.
- Example `fvSolution` entries (see examples below).

Acceptance:
- pitzDaily passes parity gate (relative L2(Ux) ≤ 1e-2) with GPU solver.
- Speedup targets: solver‑only ≥3× for U/p; end‑to‑end ≥1.3× vs CPU on the same mesh.

### Phase 2 — Resident Data Across PIMPLE

Goal: Minimize PCIe transfers by keeping vectors/matrices on device across inner PIMPLE loops.

Core tasks:
- Device memory manager: mirror arrays for field internals and matrix coefficients; reuse allocations over time steps.
- Modify GPU solver path to accept device-resident data; only copy back when fields are written or needed by CPU-only code.

Deliverables:
- Device residency utilities and lifecycle hooks tied to mesh changes and solver iterations.

Acceptance:
- Parity holds; observed reduction in transfer time and improved end-to-end speedup.

### Phase 3 — Offload Core fv Operators (targeted set)

Goal: GPU kernels for high-cost operators used by pitzDaily schemes.

Initial operators:
- `grad(U)`, `div(phi,U)`, `laplacian(nuEff, U)`, interpolation, and simple limiters used in `Gauss linearUpwind`, `upwind`, and `corrected` snGrad.

Core tasks:
- Implement kernels using mesh addressing; support non-orthogonal corrections.
- Keep runtime selection: per-operator enable with CPU fallback.

Deliverables:
- Operator kernels under `src/finiteVolume/...` with device variants; selection flags.

Acceptance:
- Parity on pitzDaily with GPU operators on; overall speedup ≥2–3× on target GPU.

### Phase 4 — Boundary Conditions & Sources

Goal: GPU implementations for common BCs and source terms.

Targets:
- Fixed/zero-gradient, inletOutlet, wall functions most used in tutorials.
- Extend turbulence GPU beyond `nut` if justified (k/epsilon sources).

Acceptance:
- Parity across broader tutorial set; CPU mode unaffected.

### Phase 5 — Multi‑GPU (MPI)

Goal: Domain-decomposed runs with CUDA‑aware MPI halos.

Core tasks:
- Device-resident halo exchanges (optional GPUDirect RDMA); overlap comm/compute.
- Preserve existing decomposition UX.

Acceptance:
- Parity on decomposed pitzDaily; basic strong scaling validated.

### Phase 6 — Hardening & Docs

Core tasks:
- CI hooks to build (CPU/GPU), run pitzDaily, compare fields, and track timings.
- Documentation of runtime flags (fvSolution, momentumTransport) and troubleshooting.
- Deployment checklist and minimal copy into `/opt/openfoam10` upon approval.

Acceptance:
- Green CI; reproducible runs; concise docs.

## Risks & Mitigations

- Numerical drift: use DP; avoid FMA changes; unit tests on kernels; parity checks per phase.
- Transfer overhead: prioritize residency (Phase 2); avoid redundant copies; pin host buffers.
- Build variability: detect CUDA path; isolate AmgX as optional; keep CPU fallback.
- External deps: make optional; provide clear build switches and diagnostics.

## Estimates (very rough)

- Phase 1: 1–2 weeks to first GPU solver + 1 week tuning.
- Phase 2: 1 week.
- Phase 3: 3–5 weeks (staged).
- Phase 4: 2–4 weeks.
- Phase 5: 2–4 weeks.
- Phase 6: 1 week.

## Decisions

- Backend (Phase 1): Custom CUDA with cuSPARSE/cuSOLVER (PCG/BiCGStab + Jacobi/ILU0) as default path. Keep optional AmgX integration as a future, opt‑in backend.
- Scope: Single‑GPU for the initial milestones (Phase 1–3). Multi‑GPU arrives in Phase 5.
- Performance targets (pitzDaily, DP):
  - Phase 1 (GPU solver only): end‑to‑end speedup ≥1.3×; solver‑only speedup ≥3× for U/p solves.
  - Phase 2 (resident data): end‑to‑end speedup ≥1.6×.
  - Phase 3 (fv operators offloaded): end‑to‑end speedup ≥2.0–3.0×.

## Examples

### momentumTransport (GPU turbulence already supported)

```
simulationType RAS;

RAS
{
    model           kEpsilon;
    turbulence      on;
    printCoeffs     on;

    kEpsilonCoeffs
    {
        device      gpu;      // or: backend gpu, or useGPU true
        gpuDevice   0;        // or: deviceId 0
    }
}
```

### fvSolution (select GPU solver at runtime; proposal)

```
// Ensure lib is loaded (in system/controlDict):
// libs ("libcudaPCG.so");

// Note: cudaPCG (symmetric) is appropriate for pressure. For U/k/epsilon (generally non-symmetric),
// keep existing smoothSolver or use a future GPU BiCGStab when available.

solvers
{
    p
    {
        solver          cudaPCG;        // symmetric GPU solver; or amgx
        tolerance       1e-7;
        relTol          0.01;
        preconditioner  diagonal;       // or: ilu0 (if implemented)
        // amgxConfig  "<path>";       // if AmgX used
    }

    pFinal { $p; relTol 0; }

    // Keep CPU solvers until a GPU BiCGStab is provided
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-5;
        relTol          0.1;
    }

    "(U|k|epsilon)Final" { $U; relTol 0; }
}
```

## Implementation Notes

- Keep `FOAM_USE_CUDA` definitions conditional on detected toolchain; fall back cleanly on CPU.
- Prefer device residency and minimal data marshaling once Phase 2 is in place.
- Do not alter public headers/interfaces outside additive runtime selection and configuration.
- Maintain warnings and explicit fallback messages for missing GPU backends.

## Deployment

- Only copy changed, minimal files into `/opt/openfoam10` after all gates pass and with approval. Document exact diffs and paths in deployment notes.

---

Revision: initial draft. Update as phases progress.
