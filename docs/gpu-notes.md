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

## Coloured preconditioner status (2025-10-12)
- NVRTC compilation fixed; colour kernels now build at runtime and stay active (telemetry and residual logs confirmed zero CPU fallback across the pitzDaily sweep).
- New telemetry:
  - `logResidualTrajectory`, `residualLogEvery`, `residualLogFile`
  - `logColourStats`, `colourStatsFile`
  - Automatically logs disable reasons, min/max/avg colour-set sizes, and writes CSVs under `postProcessing/cudaPCG/`.
- Sweep driver: `bin/gpu_colour_sweep.sh`
  - Produces parameterised runs, captures `compare_latest_U.sh` parity, and writes `summary.csv` + per-case artefacts under `run/logs/<prefix>_<timestamp>/`.
  - Latest full sweep (27 combos, pitzDaily) lives at `run/logs/pitzDaily_gpu_colour_scan2_20251012-150314/`.
- Recommended defaults (now baked into `cudaPCG`):
  - `colourOmega = 0.65`, `colourBackwardOmega = 0.85`, `colourDiagFloor = 1e-12`.
  - Combination chosen from sweep for lowest relative L2(Ux) (~1.36e-5) with stable residual decay to ~1e-7.
- When benchmarking, enable telemetry in `system/fvSolution` to capture artefacts, e.g.:
  ```foam
  reportGpuStats        true;
  logColourStats        true;
  logResidualTrajectory true;
  residualLogEvery      1;
  residualLogFile       "postProcessing/cudaPCG/residual.csv";
  colourStatsFile       "postProcessing/cudaPCG/colour_stats.csv";
  ```

## Pipelined CG & CUDA Graph (2025-10-12)
- Runtime switches added to the solver dictionary:
  - `usePipelinedCG` toggles the Chronopoulos–Gear style update (`p = z + β(p - αAp)`), reducing redundant vector operations per iteration.
  - `useCudaGraph` plus optional `cudaGraphWarmup` records the steady-state `cusparseSpMV` into a CUDA Graph after the warm-up stage and replays it on subsequent iterations via a dedicated non-blocking stream.
- `bin/gpu_colour_sweep.sh` now records PCG iteration counts, final residuals, and Execution/Clock time for each case. Artefacts for the pipelined smoke test sit under `run/logs/pitzDaily_gpu_colour_pipelined_smoke_20251012-183303/`.
- Observation: on pitzDaily the pipelined+graph path increases parity drift slightly (rel L2(Ux)≈3.3e-5 vs 1.36e-5) and is slower (~245 s vs 235 s). Keep it behind the runtime flag until further tuning, but the infrastructure is in place for Step 2 experiments.

These changes keep the CPU code untouched and establish the parallel `_gpu` scaffolding described in the acceleration plan.
