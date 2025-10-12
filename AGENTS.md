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
1. Add a coloured GPU preconditioner:
   - Build colour sets from the pressure matrix addressing and launch per-colour forward/back sweeps with damping.
   - Auto-fallback to the diagonal preconditioner if divergence or instability is detected.
2. Reduce CG iteration overhead:
   - Implement a pipelined CG variant that fuses dot products/reductions.
   - Capture the steady-state kernel sequence in a CUDA Graph to trim launch latency (retain the legacy path as a runtime switch).
3. Keep the full PIMPLE loop on the GPU:
   - Extend `pimpleFoamGPU` so ddt/div/laplacian kernels, turbulence `correct()`, and Courant calculations operate on device-resident fields.
   - Copy results back only at write times; CPU path remains untouched.
4. Validate on the standard pitzDaily case:
   - Confirm parity (relative L2(Ux) ≤ 1e-2) and measure wall-clock vs the 35 s CPU baseline.
   - Iterate until the GPU path beats the CPU timing or no further practical improvements are found.


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
