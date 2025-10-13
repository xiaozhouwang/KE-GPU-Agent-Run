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
foamDictionary -entry "PIMPLE.useGpuFieldOps" -set true "${GPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE.logGpuFieldOps" -set true "${GPU_CASE}/system/fvSolution" >/dev/null

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
