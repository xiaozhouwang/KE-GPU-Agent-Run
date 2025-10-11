# GPU Acceleration Notes

## Current Baseline
- Solver: `pimpleFoam` with `cudaPCG` (diagonal preconditioner).
- Validation case: `pitzDaily_gpu_phase1`.
- Runtime: ~120 s on GPU after removing per-iteration host/device synchronisation inside the CG loop (CPU reference ~35 s).
- Average pressure iterations: ~180 per time step (preconditioner still diagonal).

## Infrastructure Added (2025-10-10)
- `applications/solvers/incompressible/pimpleFoamGPU` – build target for GPU-specific solver experiments (mirrors the CPU solver for now).
- `src/gpu/fields/gpuField` – initial placeholder library for GPU field handles.
- `cudaPCG` optional dictionary flag `reportGpuStats` to print diagonal ranges, helping us pick safe regularisation factors when we add device-side preconditioners.
- `cudaPCG` now keeps residuals/dots on the device (no vector-size copies each iteration), giving the ~2× runtime reduction above.

## Next Focus Areas
1. Implement fully device-resident preconditioning (e.g. ILU/IC with modern cuSPARSE SpSV routines). Initial attempts via the deprecated CSRsv2 path are blocked by the current CUDA toolkit.
2. Keep field data on the GPU throughout the PIMPLE loop (remove remaining host copies).
3. Move discretisation kernels into the `_gpu` tree outlined in `~/Downloads/acc_plan.md`.

These changes keep the CPU code untouched and establish the parallel `_gpu` scaffolding described in the acceleration plan.
