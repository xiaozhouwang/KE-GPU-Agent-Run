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
