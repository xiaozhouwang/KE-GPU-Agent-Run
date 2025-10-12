# GPU Acceleration Notes

## Current Baseline
- Solver: `pimpleFoam` with `cudaPCG` (diagonal preconditioner).
- Validation case: `pitzDaily_gpu_phase1`.
- Runtime: ~116 s on GPU after removing the per-iteration host/device synchronisation that copied `Ap`/`p` back to the CPU for residual normalisation (CPU reference ~35 s).
- Average pressure iterations: ~180 per time step (GPU preconditioner work-in-progress; currently falls back to diagonal).

## Infrastructure Added (2025-10-10)
- `applications/solvers/incompressible/pimpleFoamGPU` – build target for GPU-specific solver experiments (mirrors the CPU solver for now).
- `src/gpu/fields/gpuField` – initial placeholder library for GPU field handles.
- `cudaPCG` optional dictionary flag `reportGpuStats` to print diagonal ranges, helping us pick safe regularisation factors when we add device-side preconditioners.
- `cudaPCG` now keeps residuals/dots on the device (no vector-size copies each iteration), giving the ~2× runtime reduction above.

## Next Focus Areas
1. Added a GPU symmetric Gauss–Seidel (SGS) preconditioner path: it builds the (D+L) triangular matrix from the symmetric CSR and uses cuSPARSE SpSV for forward/back solves. It’s selectable via `preconditioner SGS` in `fvSolution`. Default remains `diagonal` until we validate robustness and benefit.
2. Keep field data on the GPU throughout the PIMPLE loop (remove remaining host copies).
3. Move discretisation kernels into the `_gpu` tree outlined in `~/Downloads/acc_plan.md`.

These changes keep the CPU code untouched and establish the parallel `_gpu` scaffolding described in the acceleration plan.
