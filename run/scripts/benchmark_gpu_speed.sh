#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

CASE_NAME=${1:-pitzDaily_gpu_colour}
BASE_CASE="${REPO_ROOT}/tutorials/incompressible/pimpleFoam/RAS/pitzDaily"

if [[ ! -d "${BASE_CASE}" ]]; then
    echo "Base tutorial not found at ${BASE_CASE}" >&2
    exit 1
fi

if [[ -z "${WM_PROJECT_DIR:-}" ]]; then
    . "${REPO_ROOT}/etc/bashrc"
fi

CPU_SOLVER="pimpleFoam"
GPU_SOLVER="pimpleFoamGPU"

CPU_CASE="${REPO_ROOT}/run/${CASE_NAME}_cpu_baseline"
GPU_CASE="${REPO_ROOT}/run/${CASE_NAME}_gpu"

rm -rf "${CPU_CASE}" "${GPU_CASE}"
cp -r "${BASE_CASE}" "${CPU_CASE}"
cp -r "${BASE_CASE}" "${GPU_CASE}"

# CPU baseline uses stock solver.

# GPU case: enable GPU switches
foamDictionary -entry application -set ${GPU_SOLVER} "${GPU_CASE}/system/controlDict" >/dev/null
foamDictionary -entry "PIMPLE/useGpuFieldOps" -set true "${GPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE/logGpuFieldOps" -set false "${GPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE/logGpuFieldOps" -set false "${CPU_CASE}/system/fvSolution" >/dev/null 2>&1 || true
foamDictionary -entry "PIMPLE/useGpuCudaGraphs" -set true "${GPU_CASE}/system/fvSolution" >/dev/null 2>&1 || true

# Default to enabling CUDA Graphs unless caller overrides.
export FOAM_GPU_ENABLE_CUDA_GRAPHS="${FOAM_GPU_ENABLE_CUDA_GRAPHS:-1}"

run_case() {
    local solver=$1
    local caseDir=$2
    (
        cd "${caseDir}"
        blockMesh -dict "$FOAM_TUTORIALS/resources/blockMesh/pitzDaily" > log.blockMesh
        ${solver} > log.${solver}
    )
}

run_case "${CPU_SOLVER}" "${CPU_CASE}"
run_case "${GPU_SOLVER}" "${GPU_CASE}"

extract_time() {
    local logFile=$1
    awk '/ExecutionTime/ {time=$3} END {print time}' "${logFile}"
}

CPU_TIME=$(extract_time "${CPU_CASE}/log.${CPU_SOLVER}")
GPU_TIME=$(extract_time "${GPU_CASE}/log.${GPU_SOLVER}")

echo "CPU ExecutionTime: ${CPU_TIME} s"
echo "GPU ExecutionTime: ${GPU_TIME} s"

if [[ -n "${GPU_TIME}" && -n "${CPU_TIME}" ]]; then
    python3 - "$CPU_TIME" "$GPU_TIME" <<'PYCODE'
import sys
cpu = float(sys.argv[1])
gpu = float(sys.argv[2])
speedup = cpu / gpu if gpu else float('inf')
print(f"Speedup (CPU/GPU): {speedup:.3f}x")
PYCODE
fi

REPORT_DIR="${REPO_ROOT}/run/benchmark_results/${CASE_NAME}"
rm -rf "${REPORT_DIR}"
mkdir -p "${REPORT_DIR}"

cp "${CPU_CASE}/log.${CPU_SOLVER}" "${REPORT_DIR}/log.cpu"
cp "${GPU_CASE}/log.${GPU_SOLVER}" "${REPORT_DIR}/log.gpu"

if [[ -d "${GPU_CASE}/postProcessing/gpuFieldOps" ]]; then
    cp -r "${GPU_CASE}/postProcessing/gpuFieldOps" "${REPORT_DIR}/gpuFieldOps"
    SUMMARY_FILE="${REPORT_DIR}/gpuFieldOps/summary.csv"
    if [[ -f "${SUMMARY_FILE}" ]]; then
        total_ms=$(awk -F',' 'NR>1 {sum+=$4} END {printf "%.6f", sum+0}' "${SUMMARY_FILE}")
        fallback_samples=$(awk -F',' 'NR>1 && $2 ~ /_cpu$/ {sum+=$3} END {printf "%d", sum+0}' "${SUMMARY_FILE}")
        {
            echo "cpuExecutionTime=${CPU_TIME}"
            echo "gpuExecutionTime=${GPU_TIME}"
            echo "gpuKernelTotalMilliseconds=${total_ms}"
            echo "gpuFallbackSamples=${fallback_samples}"
        } > "${REPORT_DIR}/summary.txt"
    fi
fi
