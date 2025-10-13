#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

usage() {
    echo "Usage: $(basename "$0") [caseName]" >&2
    echo "Defaults to pitzDaily." >&2
}

CASE_NAME=${1:-pitzDaily}
BASE_CASE="${REPO_ROOT}/tutorials/incompressible/pimpleFoam/RAS/pitzDaily"

if [[ ! -d "${BASE_CASE}" ]]; then
    echo "Base tutorial not found at ${BASE_CASE}" >&2
    exit 1
fi

if [[ -z "${WM_PROJECT_DIR:-}" ]]; then
    . "${REPO_ROOT}/etc/bashrc"
fi

CPU_CASE="${REPO_ROOT}/run/${CASE_NAME}_cpu"
GPU_CASE="${REPO_ROOT}/run/${CASE_NAME}_gpu"

rm -rf "${CPU_CASE}" "${GPU_CASE}"
cp -r "${BASE_CASE}" "${CPU_CASE}"
cp -r "${BASE_CASE}" "${GPU_CASE}"

foamDictionary -entry application -set pimpleFoamGPU "${CPU_CASE}/system/controlDict" >/dev/null
foamDictionary -entry application -set pimpleFoamGPU "${GPU_CASE}/system/controlDict" >/dev/null
foamDictionary -entry "PIMPLE.useGpuFieldOps" -set false "${CPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE.useGpuFieldOps" -set true "${GPU_CASE}/system/fvSolution" >/dev/null

set_turbulence_gpu_flag()
{
    local caseDir=$1
    local value=$2
    local dictPath="${caseDir}/constant/RASProperties"
    if [[ -f "${dictPath}" ]]; then
        for entry in \
            "RASModelCoeffs.useGPU" \
            "kEpsilonCoeffs.useGPU" \
            "coeffs.useGPU"
        do
            foamDictionary -entry "${entry}" -set "${value}" "${dictPath}" >/dev/null 2>&1 || true
        done
    fi

    dictPath="${caseDir}/constant/momentumTransport"
    if [[ -f "${dictPath}" ]]; then
        for entry in \
            "RAS.coeffs.useGPU" \
            "RAS.kEpsilonCoeffs.useGPU"
        do
            foamDictionary -entry "${entry}" -set "${value}" "${dictPath}" >/dev/null 2>&1 || true
        done
    fi
}

set_turbulence_gpu_flag "${CPU_CASE}" false
set_turbulence_gpu_flag "${GPU_CASE}" true

run_case() {
    local caseDir=$1
    (
        cd "${caseDir}"
        blockMesh -dict "$FOAM_TUTORIALS/resources/blockMesh/pitzDaily" > log.blockMesh
        pimpleFoamGPU > log.pimpleFoamGPU
    )
}

run_case "${CPU_CASE}"
run_case "${GPU_CASE}"

LATEST_TIME=$(foamListTimes -case "${CPU_CASE}" -latestTime)

CPU_PHI="${CPU_CASE}/${LATEST_TIME}/phi"
GPU_PHI="${GPU_CASE}/${LATEST_TIME}/phi"
CPU_U="${CPU_CASE}/${LATEST_TIME}/U"
GPU_U="${GPU_CASE}/${LATEST_TIME}/U"

if ! diff -q "${CPU_PHI}" "${GPU_PHI}" >/dev/null; then
    echo "Mismatch detected in phi between CPU and GPU runs." >&2
    exit 2
fi

if ! diff -q "${CPU_U}" "${GPU_U}" >/dev/null; then
    echo "Mismatch detected in U between CPU and GPU runs." >&2
    exit 3
fi

echo "GPU parity check passed at time ${LATEST_TIME}."
