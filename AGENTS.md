**Scope**
- Applies to `/home/xiaozhou/OpenFOAM/openfoam10-gpu` (user-owned development mirror of OpenFOAM v10).
- Do not modify `/opt/openfoam10` during development; treat it as a stable CPU baseline.

**CPU-As-Reference**
- Preserve public APIs and behaviour; add GPU code paths via feature flags or run‑time selection.
- Validate GPU results against the existing CPU reference cases (pitzDaily CPU run) before deployment.

**Git Workflow**
- Branches: `main` mirrors the CPU baseline; `gpu` is the active development branch.
- Keep changes minimal and focused; avoid unrelated refactors. Use conventional commits.
- Do not commit build artefacts (`platforms/*`, `*/lnInclude/`, `Make/*/*`, logs).

**Build & Environment**
- Source this tree when building or running GPU trials: `. etc/bashrc` (sets `WM_PROJECT_DIR`, paths).
- Full build: `./Allwmake`
- Targeted library: `wmake libso src/<libPath>`
- If link errors occur, try `wclean libso src/<libPath>` then rebuild. If ABI/symbol issues persist, `wclean all && ./Allwmake`.

**Deployment Gate (copying to /opt)**
- Only deploy changes into `/opt/openfoam10` when ALL of the following pass:
  - Build completes without errors in this mirror.
  - A pitzDaily GPU trial runs to completion from a clean case.
  - Comparison vs CPU reference meets parity target (see run AGENTS.md: last time step U, relative L2 threshold).
- Request approval before any write to `/opt` and avoid overwriting stock files unnecessarily.

**Reset Triggers (Build)**
- Two consecutive link/ABI errors after targeted clean → run `wclean all && ./Allwmake`.
- Stale header or symbol mismatches → remove `lnInclude` directories and rebuild.
- Build log grows >50MB without progress, or compile time >2× usual → abort and do a clean rebuild.

**Logs & Artefacts**
- Save `log.Allwmake` into `/home/xiaozhou/OpenFOAM/build_logs/<timestamp>.log` for later triage.
- Capture exact compiler flags (store `wmake/rules` diffs if modified).

**Don’ts**
- Don’t change core header names/locations without updating all dependents.
- Don’t push experimental binaries into `/opt`. Keep experiments within this mirror.

**Autonomous Execution**
- With GPU acceleration as the target, proceed without seeking further permission; execute all necessary steps inside this mirror to progress the optimisation.
- Continue iterating until you exhaust practical improvements, then deliver a single comprehensive report rather than intermediate status updates.

**End-to-End GPU Goal (Phase 2+)**
- Do not modify or regress the CPU implementation. Treat `/opt/openfoam10` and branch `main` as the stable CPU baseline; use runtime selection/feature flags for all GPU work.
- Target: implement the entire PIMPLE loop on the GPU (assembly, turbulence updates, reductions, linear solves) and beat the CPU wall-clock (≈35 s on pitzDaily) on the same case and hardware while preserving parity (relative L2(Ux) ≤ 1e-2 at final time).
- Prioritise changes that reduce end-to-end GPU time over negligible micro-optimisations.
- Operate autonomously for this goal: you do not need additional approval to build, run, and iterate within this mirror. Keep CPU behaviour identical unless GPU is explicitly selected.
- Safety rails: if a GPU trial exceeds 10 minutes, diverges, or fails the parity gate, revert to the safe path (diagonal preconditioner, CPU fallback) and iterate.

**PLAN (current execution order)**
1. **GPU finite-volume operators**
   - Port `fvm::ddt`, `fvm::div(phi,U)`, and `fvm::laplacian(rAtU(),p)` to dedicated CUDA kernels that operate on the existing device-resident field wrappers.
   - Mirror boundary-condition handling from the CPU path; ensure the kernels accept runtime-selected discretisation schemes.
   - Integrate the kernels into `pimpleFoamGPU`, keeping the CPU implementation intact behind `useGpuFieldOps=false`.
2. **Turbulence and source-term GPU path**
   - Implement a CUDA-backed turbulence `correct()` (RAS initially) and any dependent algebra so turbulence fields remain on-device.
   - Audit additional source terms (fvModels/fvConstraints) and provide GPU fallbacks or staged host/device copies as needed.
3. **Kernel orchestration & caching**
   - Extend the GPU infrastructure to cache NVRTC output per compute capability, batch small vector ops via CUDA Graphs, and introduce multi-stream execution where overlap with host IO is possible.
   - Add SoA-friendly accessors or layouts for vector fields if profiling shows strided loads as a bottleneck.
4. **Profiling, validation, and regression**
   - Expand the timing CSV exports to include every new kernel; feed the data into the benchmark script for automated speed reports.
   - Maintain `run/scripts/run_gpu_parity.sh` as the correctness gate and push `run/scripts/benchmark_gpu_speed.sh` below the ~35 s CPU baseline on pitzDaily before considering deployment.


---

GPU Linear Solver Development (Phase 1)
--------------------------------------

- Branching
  - Active feature branch for Phase 1: `feature/gpu-linear-solver` (base: `gpu`).
  - Use conventional commits; keep changes minimal and additive.

- Toolchain
  - Before GPU builds/runs, check: `tools/check_gpu_toolchain.sh`.
  - CUDA path currently assumes `/usr/local/cuda` in `src/MomentumTransportModels/**/Make/options`.
    - If CUDA is elsewhere, set up symlink or adjust Make/options locally.
    - CPU fallback must remain operational if CUDA is not present.

- Build
  - Source environment from this mirror: `. etc/bashrc`.
  - To keep dev builds green, PVReaders is optional: export `WM_NOCMAKE=on` when running `./Allwmake` or avoid building `applications/utilities/postProcessing/graphics`.
  - Targeted builds:
    - Core matrices/solvers will live under `src/OpenFOAM/matrices/lduMatrix/solvers/` (e.g., `cudaPCG`).
    - Use `wmake libso` on the corresponding library when iterating.

- Runtime selection (planned)
  - A new `lduMatrix::solver` (e.g., `cudaPCG`) will be selectable via `system/fvSolution`.
  - If the GPU solver/backend is unavailable or disabled, runtime must fall back to the CPU solver without errors.
  - When using the new solver, ensure the library is loaded via `system/controlDict`:
    - `libs ("libcudaPCG.so");`

- Validation
  - Use `run/scripts/run_gpu_trial.sh pitzDaily_gpu<N> <mirror> 1e-2`.
  - Acceptance gate: pitzDaily runs to completion; relative L2(Ux) ≤ 1e-2 at the last time.
  - Record timings for solver phases to track speedups.

- Deployment
  - Follow existing deployment gate. Do not copy into `/opt/openfoam10` until all gates pass. Copy only minimal changed files; request approval first.
