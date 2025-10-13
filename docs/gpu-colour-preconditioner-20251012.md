# Coloured Preconditioner Sweep – 2025-10-12

All runs executed on the development mirror (`feature/gpu-linear-solver`) with `libcudaPCG` built from commit-in-progress. Each sweep case copies `pitzDaily_clean`, enables the colour preconditioner for `p`, logs GPU telemetry, and compares the final write against the CPU reference (`run/scripts/compare_latest_U.sh ... 1e-2`).

- Driver: `bin/gpu_colour_sweep.sh`
- Artefacts: `run/logs/pitzDaily_gpu_colour_scan2_20251012-150314/`
  - `summary.csv` – aggregated parity + colour metrics (see table below)
  - `*/colour_stats.csv` – per-case colour-set statistics
  - `*/residual.csv` – per-case residual trajectory (iteration vs. ‖r‖₁/‖b‖)
  - `*/compare.txt` – parity report for `U`, `p`, `k`, `epsilon`

## Summary statistics (per omega / backward omega trio)

Values averaged across diagonal floors `{1e-12, 5e-12, 1e-11}`. `relL2(Ux)` reported in 1e-5 units for readability; residuals scaled similarly.

| ω_fwd | ω_back | mean relL2(Ux)×1e⁻⁵ | mean PCG iterations | mean final residual ×1e⁻⁷ | notes |
|------:|-------:|---------------------:|--------------------:|---------------------------:|:------|
| 0.65  | 0.65   | 2.50 | 91  | 43.8 | Slower decay; higher steady-state residual. |
| 0.65  | 0.75   | 2.07 | 176 | 29.4 | Improves parity vs. ω_back = 0.65. |
| **0.65** | **0.85** | **1.36** | **782** | **0.96** | Lowest parity error, residual ≈ 1e⁻⁷. |
| 0.75  | 0.65   | 3.36 | 38  | 299  | Converges quickly but stalls far above tolerance. |
| 0.75  | 0.75   | 4.26 | 747 | 2.79 | Stable but parity poorest. |
| 0.75  | 0.85   | 2.05 | 732 | 2.45 | Acceptable parity; residual O(1e⁻⁷). |
| 0.85  | 0.65   | 1.93 | 732 | 3.29 | Final residual O(1e⁻⁶); parity moderately low. |
| 0.85  | 0.75   | 1.43 | 745 | 0.99 | Second-best parity; balanced residual. |
| 0.85  | 0.85   | 2.62 | 100 | 39.7 | Residual plateaus above 1e⁻⁶. |

Observations:
- Diagonal floor had negligible effect on parity or convergence in this range.
- Aggressive backward relaxation (`ω_back ≥ 0.85`) noticeably reduces parity drift; matching ω_forward produced worse residuals.
- Combinations that left the residual ≥O(1e⁻⁶) also accumulated higher parity error over the time horizon.

## Recommended defaults

- **Forward relaxation (`colourOmega`)**: 0.65  
- **Backward relaxation (`colourBackwardOmega`)**: 0.85  
- **Diagonal floor (`colourDiagFloor`)**: 1e-12

These values are now the compilation-time defaults inside `cudaPCG` (overridable from `fvSolution`). They minimise the observed parity error while maintaining residual decay to ≈1e⁻⁷. Future sweeps on larger or non-symmetric cases should reuse the driver script and append results here for traceability.

## Additional validation – pitzDaily_gpu_phase1

- Artefacts: `run/logs/pitzDaily_gpu_phase1_colour_scan2_20251012-183737/`
  - `summary.csv` (same schema as above) + per-case residual/colour logs.
- The transient/longer-run case exhibits the same ordering as the steady pitzDaily sweep—`(ω_fwd, ω_back) = (0.65, 0.85)` remains the lowest-drift choice.

| ω_fwd | ω_back | mean relL2(Ux)×1e⁻⁵ | mean PCG iterations | mean final residual ×1e⁻⁶ | mean ExecTime [s] |
|------:|-------:|---------------------:|--------------------:|---------------------------:|------------------:|
| 0.65  | 0.65   | 2.50 | 91.0  | 4.38  | 229.089 |
| 0.65  | 0.75   | 2.07 | 176.0 | 2.94  | 232.375 |
| **0.65** | **0.85** | **1.36** | **782.0** | **0.10** | **235.307** |
| 0.75  | 0.65   | 3.35 | 38.0  | 29.95 | 229.385 |
| 0.75  | 0.75   | 4.26 | 747.0 | 0.28  | 222.972 |
| 0.75  | 0.85   | 2.05 | 732.0 | 0.25  | 226.754 |
| 0.85  | 0.65   | 1.93 | 732.0 | 0.33  | 244.257 |
| 0.85  | 0.75   | 1.43 | 745.0 | 0.10  | 237.860 |
| 0.85  | 0.85   | 2.62 | 100.0 | 3.97  | 219.951 |

Final residuals are scaled to 1e⁻⁶ for readability; execution times come from the tail `ExecutionTime` line in each `log.pimpleFoam`. The relative ranking mirrors the steady case, lending confidence to the selected defaults.

## Targeted low-iteration sweep (2025-10-12 evening)

- Artefacts: `run/logs/pitzDaily_gpu_colour_grid_small_*` (16 runs, single floor `1e-12`).
- Goal: find `(colourOmega, colourBackwardOmega)` pairs that minimise PCG iterations while keeping rel L2(Ux) ≤ ~4e-5.
- Observations:
  * Lowering ω_fwd below 0.55 is risky (parity drifts, execution time unaffected).
  * ω_fwd=0.65, ω_back=0.80 yields ~77 iterations and the best runtime so far (~231 s) with rel L2≈3.2e-5.
  * Extreme back-relaxation (ω_back=0.85) still gives the best parity but pushes iterations toward the 800 range.

| ω_fwd | ω_back | iterations | ExecTime [s] | relL2(Ux) |
|------:|-------:|-----------:|-------------:|-----------:|
| 0.50 | 0.85 | 66 | 247.60 | 2.22e-05 |
| 0.55 | 0.70 | 67 | 236.16 | 3.12e-05 |
| 0.65 | 0.80 | 77 | 230.72 | 3.20e-05 |
| 0.50 | 0.75 | 84 | 249.60 | 4.40e-05 |
| 0.55 | 0.75 | 87 | 236.79 | 3.31e-05 |
| 0.60 | 0.85 | 116 | 238.32 | 3.80e-05 |
| 0.65 | 0.75 | 176 | 235.50 | 2.07e-05 |
| 0.65 | 0.70 | 186 | 235.42 | 2.26e-05 |
| 0.60 | 0.80 | 252 | 242.19 | 3.24e-05 |
| 0.50 | 0.70 | 721 | 245.93 | 2.75e-05 |
| 0.60 | 0.70 | 767 | 236.29 | 1.44e-05 |
| 0.55 | 0.80 | 773 | 234.98 | 2.78e-05 |
| 0.65 | 0.85 | 782 | 235.14 | 1.36e-05 |
| 0.60 | 0.75 | 790 | 237.53 | 8.14e-06 |
| 0.50 | 0.80 | 817 | 245.75 | 1.39e-05 |
| 0.55 | 0.85 | 829 | 245.46 | 1.05e-05 |

Takeaway: ω_fwd≈0.65 with ω_back≈0.80 is a promising compromise (iterations ≈77, rel L2≈3.2e-5, ExecTime ≈231 s). Further tuning should focus on nudging ω_fwd between 0.6–0.65 and ω_back 0.78–0.82, possibly with adaptive diagonal floors, to recover the 116 s baseline while keeping parity intact.
