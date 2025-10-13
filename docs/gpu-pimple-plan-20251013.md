# GPU PIMPLE Loop – Device Residency Plan

## Objectives

- Keep primary volume and surface fields (`volScalarField`, `volVectorField`, `surfaceScalarField`) resident on the GPU whenever the GPU path is enabled.
- Provide an opt-in path that leaves the CPU code untouchhed when the GPU backend is disabled or unavailable.
- Supply explicit host↔device synchronisation hooks so that we can port PIMPLE building blocks (ddt, div, laplacian, turbulence corrector, Courant logic) incrementally.

## Design Highlights

1. **Device Field Wrappers**
   - Introduce lightweight `Foam::gpu::DeviceField<Type>` wrappers that own a CUDA buffer (`cudaMalloc/cudaFree`) and track dirty flags (`hostDirty`, `deviceDirty`).
   - The wrapper is non-intrusive: it references an existing `GeometricField` (or `Field`) and mirrors the internal-field storage when asked.
   - Synchronisation is explicit:
     - `syncHostToDevice(...)` copies the current internal field to the device buffer.
     - `syncDeviceToHost(...)` copies the device buffer back to the host field.
   - When CUDA support is not compiled in (`FOAM_USE_CUDA` off), the class degenerates to a stub that compiles away.

2. **Field Registry**
   - A `Foam::gpu::FieldRegistry` keyed by address of the owning `GeometricField` ensures a single device buffer per field.
   - Registration is explicit (from the solver) to avoid touching the core `GeometricField` constructors/destructors.
   - Registry lifetime is tied to the `objectRegistry` via `MeshObject`, so buffers get released when the mesh/solver finishes.

3. **Stream-aware Launch Context**
   - A per-solver launch context stores the selected device, CUDA stream, and error-handling helpers (shared with `cudaPCG` in future cleanup).
   - All kernels launched through helper functions that take the `launchContext` to manage error propagation and to keep a consistent stream policy.

4. **Initial Kernel Scope**
   - Start with kernels that operate purely on internal fields (no face coupling) to bring up the data path:
     - `assign`: copy / scale / axpy style kernels.
     - `Field-norm` reductions, used by Courant number logic and convergence monitors.
   - Extend to face-coupled kernels (e.g. divergence, laplacian) once the field residency plumbing is validated.

5. **Solver Integration**
   - `pimpleFoamGPU` will:
     - Detect GPU availability (`Foam::gpu::isEnabled()`).
     - Register the primary fields with the registry after `createFields.H`.
     - Swap discrete operator calls (e.g. `fvm::ddt`) with GPU-aware helpers that default to the existing CPU implementation when the GPU backend isn’t ready.
   - Host-side fallbacks remain 1:1 with the upstream solver to preserve CPU behaviour.

## Incremental Roll-out

1. Implement the DeviceField + Registry infrastructure.
2. Provide host↔device sync for `U`, `p`, `phi`, `Uf`, turbulence fields.
3. Port algebraic vector operations (scale, axpy) and reductions (dot, norm) to the GPU.
4. Replace PIMPLE building blocks with GPU kernels one by one, validating against the CPU reference after each major milestone.

This doc will track deviations and updates as the GPU PIMPLE loop matures.

## Current Prototypes

- `fvSolution` toggle `PIMPLE.useGpuFieldOps` enables the first device-side operation:
  - Final non-orthogonal update of `phi` executes as a CUDA kernel (`phi = phiHbyA - flux`).
  - Falls back to CPU automatically and logs a warning if the kernel or data sync fails.
- The velocity corrector now has a CUDA path (`U = HbyA - rAtU*grad(p)`), reusing the same residency plumbing and reporting kernel timing when `PIMPLE.logGpuFieldOps true`.
- Device residency is established for `U`, `p`, and `phi` immediately after `createFields.H`. Transient working fields use scoped device buffers to avoid stale registry entries.

## Verification Notes

- Build targets:
  - `wmake libso src/gpuInfrastructure`
  - `wmake applications/solvers/incompressible/pimpleFoamGPU`
- Smoke test: run `pimpleFoamGPU` on `pitzDaily` twice (with and without `useGpuFieldOps`) and diff the final `phi`.
- Automated harness: `run/scripts/run_gpu_parity.sh` performs the twin runs and reports mismatches (diff on `phi` & `U`).
- Performance sweep: `run/scripts/benchmark_gpu_speed.sh` rebuilds baseline CPU/GPU cases and prints wall-clock and computed speedup.
- Once more kernels land, extend the existing `run/scripts/run_gpu_trial.sh` harness to pass `useGpuFieldOps true` via an overlay `fvSolution` dictionary and compare against the CPU reference (`relative L2(Ux) ≤ 1e-2` gate).
