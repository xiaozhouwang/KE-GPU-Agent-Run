#!/usr/bin/env bash
# Sweep coloured cudaPCG preconditioner parameters on pitzDaily-like cases.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: gpu_colour_sweep.sh [options]

Options:
  --run-root DIR        Run root containing template case and logs (default: /home/xiaozhou/OpenFOAM/xiaozhou-10/run)
  --template NAME       Template case under run-root (default: pitzDaily_clean)
  --cpu-case NAME       CPU reference case (default: pitzDaily_cpu)
  --prefix NAME         Prefix for generated cases/logs (default: pitzDaily_gpu_colour)
  --omegas LIST         Forward colour relaxation list (comma-separated, default: 0.65,0.75,0.85,0.95)
  --backward LIST       Backward relaxation list (comma-separated, default: reuse --omegas)
  --floors LIST         Diagonal floor list (comma-separated, default: 1e-12,5e-12,1e-11)
  --residual-every N    Residual logging stride (default: 1)
  --threshold VALUE     Relative L2(Ux) threshold (default: 1e-2)
  --pipelined           Enable pipelined CG path (sets usePipelinedCG true)
  --cuda-graph          Enable CUDA Graph replay (sets useCudaGraph true)
  --graph-warmup N      Iterations before capturing CUDA Graph (default: 5)
  --help                Show this message
EOF
}

sanitize_token() {
    local token="$1"
    token="${token//./p}"
    token="${token//-/m}"
    token="${token//+/p}"
    printf '%s' "$token"
}

remove_path() {
    local target="$1"
    python3 - "$target" <<'PY'
import os
import shutil
import sys

path = sys.argv[1]
if os.path.isdir(path):
    shutil.rmtree(path)
PY
}

replace_file_entry() {
    local dict="$1"
    local key="$2"
    local value="$3"
    python3 - "$dict" "$key" "$value" <<'PY'
import re
import sys

path, key, value = sys.argv[1:4]
with open(path, 'r', encoding='utf-8') as fh:
    text = fh.read()
pattern = rf'({re.escape(key)})\s+[^;\n]+;'
replacement = rf'\1 "{value}";'
updated, count = re.subn(pattern, replacement, text, count=1)
if count == 0:
    raise SystemExit(f"Failed to set {key} in {path}")
with open(path, 'w', encoding='utf-8') as fh:
    fh.write(updated)
PY
}

foam_dict_set() {
    local dict="$1"
    local entry="$2"
    local value="$3"
    foamDictionary "$dict" -entry "$entry" -set "$value" >/dev/null
}

run_root=/home/xiaozhou/OpenFOAM/xiaozhou-10/run
template_case=pitzDaily_clean
cpu_case=pitzDaily_cpu
prefix=pitzDaily_gpu_colour
omega_list="0.65,0.75,0.85,0.95"
backward_list=""
floor_list="1e-12,5e-12,1e-11"
res_stride=1
threshold=1e-2
enable_pipelined=false
enable_cuda_graph=false
graph_warmup=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) run_root="$2"; shift 2;;
        --template) template_case="$2"; shift 2;;
        --cpu-case) cpu_case="$2"; shift 2;;
        --prefix) prefix="$2"; shift 2;;
        --omegas) omega_list="$2"; shift 2;;
        --backward) backward_list="$2"; shift 2;;
        --floors) floor_list="$2"; shift 2;;
        --residual-every) res_stride="$2"; shift 2;;
        --threshold) threshold="$2"; shift 2;;
        --pipelined) enable_pipelined=true; shift ;;
        --cuda-graph) enable_cuda_graph=true; shift ;;
        --graph-warmup) graph_warmup="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 2;;
    esac
done

IFS=',' read -r -a omegas <<<"$omega_list"
if [[ -n "$backward_list" ]]; then
    IFS=',' read -r -a backward_omegas <<<"$backward_list"
else
    backward_omegas=("${omegas[@]}")
fi
IFS=',' read -r -a diag_floors <<<"$floor_list"

timestamp=$(date +%Y%m%d-%H%M%S)
log_root="$run_root/logs/${prefix}_${timestamp}"
mkdir -p "$log_root"
summary="$log_root/summary.csv"
echo "case,omega,backwardOmega,diagFloor,compareStatus,relL2_Ux,nColours,minColourSize,maxColourSize,avgColourSize,disableCount,pcgIterations,pcgFinalResidual,executionTime,clockTime,colourStats,residualLog" >"$summary"

set +u

template_path="$run_root/$template_case"
cpu_path="$run_root/$cpu_case"
compare_script="$run_root/scripts/compare_latest_U.sh"

if [[ ! -d "$template_path" ]]; then
    echo "Template case not found: $template_path" >&2
    exit 1
fi
if [[ ! -d "$cpu_path" ]]; then
    echo "CPU reference case not found: $cpu_path" >&2
    exit 1
fi
if [[ ! -x "$compare_script" ]]; then
    echo "compare_latest_U.sh not found at $compare_script" >&2
    exit 1
fi

echo "Writing logs to $log_root"

for omega in "${omegas[@]}"; do
    for back in "${backward_omegas[@]}"; do
        for floor_val in "${diag_floors[@]}"; do
            token_o=$(sanitize_token "$omega")
            token_b=$(sanitize_token "$back")
            token_f=$(sanitize_token "$floor_val")
            case_name="${prefix}_o${token_o}_b${token_b}_d${token_f}"
            case_path="$run_root/$case_name"
            case_log="$log_root/$case_name"
            mkdir -p "$case_log"

            echo "--> Running $case_name (omega=$omega, back=$back, floor=$floor_val)"

            remove_path "$case_path"
            cp -a "$template_path" "$case_path"

            (
                cd "$case_path"
                rm -f log.pimpleFoam log.* || true
                if command -v foamClearCase >/dev/null 2>&1; then
                    foamClearCase || true
                else
                    rm -rf 0.[0-9]* [1-9]* processor* postProcessing || true
                fi
            )

            dict="$case_path/system/fvSolution"
            foam_dict_set "$dict" "solvers/p/preconditioner" "colour"
            foam_dict_set "$dict" "solvers/pFinal/preconditioner" "colour"
            foam_dict_set "$dict" "solvers/p/colourOmega" "$omega"
            foam_dict_set "$dict" "solvers/p/colourBackwardOmega" "$back"
            foam_dict_set "$dict" "solvers/p/colourDiagFloor" "$floor_val"
            foam_dict_set "$dict" "solvers/p/reportGpuStats" "true"
            foam_dict_set "$dict" "solvers/p/logColourStats" "true"
            foam_dict_set "$dict" "solvers/p/logResidualTrajectory" "true"
            foam_dict_set "$dict" "solvers/p/residualLogEvery" "$res_stride"
            foam_dict_set "$dict" "solvers/p/residualLogFile" "postProcessing/cudaPCG/residual.csv"
            replace_file_entry "$dict" "residualLogFile" "postProcessing/cudaPCG/residual.csv"
            foam_dict_set "$dict" "solvers/p/colourStatsFile" "postProcessing/cudaPCG/colour_stats.csv"
            replace_file_entry "$dict" "colourStatsFile" "postProcessing/cudaPCG/colour_stats.csv"
            foam_dict_set "$dict" "solvers/p/gpuDevice" "0"
            if [[ "$enable_pipelined" == true ]]; then
                foam_dict_set "$dict" "solvers/p/usePipelinedCG" "true"
            fi
            if [[ "$enable_cuda_graph" == true ]]; then
                foam_dict_set "$dict" "solvers/p/useCudaGraph" "true"
                foam_dict_set "$dict" "solvers/p/cudaGraphWarmup" "$graph_warmup"
            fi

            foam_dict_set "$dict" "solvers/pFinal/colourOmega" "$omega"
            foam_dict_set "$dict" "solvers/pFinal/colourBackwardOmega" "$back"
            foam_dict_set "$dict" "solvers/pFinal/colourDiagFloor" "$floor_val"

            (
                cd "$case_path"
                chmod +x ./Allrun || true
                ./Allrun
            ) | tee "$case_log/Allrun.log"

            compare_status="PASS"
            if "$compare_script" "$cpu_path" "$case_path" "$threshold" | tee "$case_log/compare.txt"; then
                :
            else
                compare_status="FAIL"
            fi

            colour_stats_src="$case_path/postProcessing/cudaPCG/colour_stats.csv"
            residual_src="$case_path/postProcessing/cudaPCG/residual.csv"
            colour_stats_dest="$case_log/colour_stats.csv"
            residual_dest="$case_log/residual.csv"
            if [[ -f "$colour_stats_src" ]]; then
                cp "$colour_stats_src" "$colour_stats_dest"
            fi
            if [[ -f "$residual_src" ]]; then
                cp "$residual_src" "$residual_dest"
            fi

            pcg_iters=""
            pcg_final=""
            if [[ -f "$residual_dest" ]]; then
                read -r pcg_iters pcg_final <<<"$(python3 - "$residual_dest" <<'PY'
import csv
import sys

path = sys.argv[1]
last_iter = ""
last_res = ""
with open(path, 'r', encoding='utf-8') as fh:
    reader = csv.reader(row for row in fh if row.strip() and not row.startswith('#'))
    header = next(reader, None)
    for row in reader:
        if len(row) >= 2:
            last_iter = row[0].strip()
            last_res = row[1].strip()
print(f"{last_iter} {last_res}")
PY
)"
            fi

            exec_time=""
            clock_time=""
            if [[ -f "$case_path/log.pimpleFoam" ]]; then
                read -r exec_time clock_time <<<"$(python3 - "$case_path/log.pimpleFoam" <<'PY'
import sys

path = sys.argv[1]
exec_time = ""
clock_time = ""
with open(path, 'r', encoding='utf-8') as fh:
    for line in fh:
        if "ExecutionTime" in line and "ClockTime" in line:
            parts = line.strip().replace('=', '').split()
            # Expected format: ExecutionTime value ClockTime value
            try:
                idx_exec = parts.index('ExecutionTime')
                exec_time = parts[idx_exec + 1]
                idx_clock = parts.index('ClockTime')
                clock_time = parts[idx_clock + 1]
            except Exception:
                continue
if exec_time:
    print(f"{exec_time} {clock_time}")
else:
    print()
PY
)"
            fi

            colour_line=""
            if [[ -f "$colour_stats_dest" ]]; then
                colour_line=$(tail -n 1 "$colour_stats_dest")
            fi

            n_colours="" min_colour="" max_colour="" avg_colour="" disable_count=""
            if [[ -n "$colour_line" ]]; then
                IFS=',' read -r n_colours min_colour max_colour avg_colour _ _ _ _ _ _ _ _ disable_count disable_stages <<<"$colour_line"
            fi

            rel_l2_ux=""
            if [[ -f "$case_log/compare.txt" ]]; then
                rel_l2_ux=$(python3 - "$case_log/compare.txt" <<'PY'
import re
import sys

log_path = sys.argv[1]
ux_value = ""
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if "Ux:" in line:
            match = re.search(r"相对L2=([0-9eE\+\-\.]+)", line)
            if match:
                ux_value = match.group(1)
                break
if ux_value:
    print(ux_value)
PY
)
            fi

            echo "$case_name,$omega,$back,$floor_val,$compare_status,$rel_l2_ux,$n_colours,$min_colour,$max_colour,$avg_colour,$disable_count,$pcg_iters,$pcg_final,$exec_time,$clock_time,$colour_stats_dest,$residual_dest" >>"$summary"
        done
    done
done

echo "Sweep complete. Summary: $summary"
