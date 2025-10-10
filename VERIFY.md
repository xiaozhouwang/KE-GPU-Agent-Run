# Phase 1 Verification (GPU Linear Solver Offload)

Purpose: Provide a repeatable checklist to validate the Phase 1 work (GPU linear solver offload) in the dev mirror at `home/xiaozhou/OpenFOAM/openfoam10-gpu`, ensuring builds, runs, and comparisons are correct and safe.

## Read First
- AGENTS.md (repo root)
- PLAN.md (repo root)
- Run AGENTS.md: `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/AGENTS.md`

## Preconditions
- You are on the development machine with CUDA available (for GPU paths).
- Dev mirror env will be sourced: `. etc/bashrc`.
- CPU reference case is present and complete at `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/pitzDaily_cpu`.

## Verify Structure
1) Dev mirror exists and is a git repo with required branches
   - `cd /home/xiaozhou/OpenFOAM/openfoam10-gpu`
   - `git branch --all` (expect: `main`, `gpu`, and feature branch `feature/gpu-linear-solver`)

2) Scripts exist and are executable
   - `/home/xiaozhou/OpenFOAM/openfoam10-gpu/tools/build_all.sh`
   - `/home/xiaozhou/OpenFOAM/openfoam10-gpu/tools/check_gpu_toolchain.sh`
   - `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts/last_time.sh`
   - `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts/compare_latest_U.sh`
   - `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts/run_gpu_trial.sh`

3) Planned solver path (Phase 1)
   - `src/OpenFOAM/matrices/lduMatrix/solvers/cudaPCG/` (gated by `FOAM_USE_CUDA`).
   - Built artifact: `$FOAM_USER_LIBBIN/libcudaPCG.so` after targeted build.
   - If not present yet, Phase 1 implementation has not been merged; verify CPU fallback steps below.

4) Make/options CUDA flags
   - Confirm `FOAM_USE_CUDA` and CUDA includes/libs appear where needed (temporary assumption: `/usr/local/cuda`).

## Toolchain Check
Run the helper to validate CUDA/HIP presence:
```
/home/xiaozhou/OpenFOAM/openfoam10-gpu/tools/check_gpu_toolchain.sh
```
Expect: detects `nvcc` or `hipcc`. If neither is found, GPU paths must gracefully fall back to CPU.

## Build Checks
1) Source the dev environment:
```
cd /home/xiaozhou/OpenFOAM/openfoam10-gpu
. etc/bashrc
```
2) Keep dev builds green (PVReaders optional):
```
export WM_NOCMAKE=on
```
3) Full build (logs stored automatically):
```
./tools/build_all.sh
```
Confirm a new log in `/home/xiaozhou/OpenFOAM/build_logs/build-*.log`.

4) Targeted build (when cudaPCG exists):
```
wmake libso src/OpenFOAM/matrices/lduMatrix/solvers/cudaPCG
```
Confirm the library at `$FOAM_USER_LIBBIN/libcudaPCG.so`.
If link/ABI issues occur, follow AGENTS.md reset triggers (targeted `wclean`, then `wclean all && ./Allwmake`).

## Run & Compare Checks
1) Sanity (CPU reference last time):
```
/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts/last_time.sh \
  /home/xiaozhou/OpenFOAM/xiaozhou-10/run/pitzDaily_cpu
```
Expect: prints the largest numeric time (e.g., `0.3`).

2) Sanity (self-compare on CPU case):
```
/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts/compare_latest_U.sh \
  /home/xiaozhou/OpenFOAM/xiaozhou-10/run/pitzDaily_cpu \
  /home/xiaozhou/OpenFOAM/xiaozhou-10/run/pitzDaily_cpu 1e-2
```
Expect: succeeds with near-zero relative L2(Ux).

3) GPU trial automation path (dev mirror env):
```
/home/xiaozhou/OpenFOAM/xiaozhou-10/run/scripts/run_gpu_trial.sh \
  pitzDaily_gpu_phase1 \
  /home/xiaozhou/OpenFOAM/openfoam10-gpu \
  1e-2
```
Expect:
- Creates/cleans the case, runs to completion, writes logs to `/home/xiaozhou/OpenFOAM/xiaozhou-10/run/logs/pitzDaily_gpu_phase1/`.
- `compare.txt` exists and reports relative L2; must be ≤ 1e-2 to pass the parity gate.

4) Runtime selection for GPU solver (once implemented)
- Load the library in `system/controlDict`:
  - `libs ("libcudaPCG.so");`
- In the case `system/fvSolution`, set solver names for `(U|k|epsilon|p)` to the GPU variant (planned `cudaPCG`).
- Rerun step (3). If GPU solver fails to initialize or is unavailable, the run must fall back to CPU solvers without crashing.

## Policies & Safety
- CPU-as-reference: comparisons always use the last numeric time directory’s `U`.
- Never write to `/opt/openfoam10` during development; deploy only after all gates pass and with approval.
- PVReaders can be skipped in dev builds (`WM_NOCMAKE=on`).
- Keep changes additive; do not commit build artefacts (`platforms/*`, `*/lnInclude/`, `Make/*/*`, logs).

## Report Template
- Structure
  - Mirror present: [PASS/FAIL] — path: `/home/xiaozhou/OpenFOAM/openfoam10-gpu`
  - Branches (`main`, `gpu`, `feature/gpu-linear-solver`): [PASS/FAIL]
  - Scripts exist/executable (list all five): [PASS/FAIL]
  - Solver path exists (cudaPCG): [PASS/NA]
- Toolchain
  - CUDA/HIP detection output: [text]
- Build
  - Build completed (dev) with `WM_NOCMAKE=on`: [PASS/FAIL]
  - Targeted build `cudaPCG` (if present): [PASS/NA]
  - Build log saved: [path]
- Run & Compare
  - last_time.sh on CPU case: [output]
  - CPU self-compare: [PASS/FAIL]
  - GPU trial: [PASS/FAIL], compare log: [path], rel L2(Ux): [value]
  - GPU runtime selection (if implemented): fallback behavior [OK/FAIL]
- Policy
  - Scripts avoid writing to `/opt`: [PASS/FAIL]
  - Env sourcing uses dev mirror: [PASS/FAIL]

## Notes
- If CUDA is missing, ensure CPU fallback paths run and the audit still completes with clear warnings.
- When Phase 1 components are not yet merged, treat solver build/runtime selection checks as N/A; the rest of the pipeline must still be functional.
