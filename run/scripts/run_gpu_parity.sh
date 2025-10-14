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
foamDictionary -entry "PIMPLE/useGpuFieldOps" -set false "${CPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE/useGpuFieldOps" -set true "${GPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE/logGpuFieldOps" -set false "${CPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE/logGpuFieldOps" -set true "${GPU_CASE}/system/fvSolution" >/dev/null
FORCE_CPU_PRESSURE=${FORCE_CPU_PRESSURE:-false}
foamDictionary -entry "PIMPLE/forceCpuPressureCorrector" -set false "${CPU_CASE}/system/fvSolution" >/dev/null
foamDictionary -entry "PIMPLE/forceCpuPressureCorrector" -set "${FORCE_CPU_PRESSURE}" "${GPU_CASE}/system/fvSolution" >/dev/null

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

# Watchdog timeout for GPU parity run (seconds)
GPU_TRIAL_TIMEOUT_SEC=${GPU_TRIAL_TIMEOUT_SEC:-600}

run_case() {
    local caseDir=$1
    (
        cd "${caseDir}"
        blockMesh -dict "$FOAM_TUTORIALS/resources/blockMesh/pitzDaily" > log.blockMesh
        timeout -k 10 ${GPU_TRIAL_TIMEOUT_SEC}s pimpleFoamGPU > log.pimpleFoamGPU || {
            echo "Parity GPU run exceeded ${GPU_TRIAL_TIMEOUT_SEC}s or failed; see log.pimpleFoamGPU" >&2
            exit 124
        }
    )
}

run_case "${CPU_CASE}"
run_case "${GPU_CASE}"

RESULT_BASE="${REPO_ROOT}/run/parity_results/${CASE_NAME}"
rm -rf "${RESULT_BASE}"
mkdir -p "${RESULT_BASE}"

cp "${CPU_CASE}/log.pimpleFoamGPU" "${RESULT_BASE}/log.cpu"
cp "${GPU_CASE}/log.pimpleFoamGPU" "${RESULT_BASE}/log.gpu"

if [[ -d "${GPU_CASE}/postProcessing/gpuFieldOps" ]]; then
    cp -r "${GPU_CASE}/postProcessing/gpuFieldOps" "${RESULT_BASE}/gpuFieldOps"
fi

LATEST_TIME=$(foamListTimes -case "${CPU_CASE}" -latestTime)

CPU_PHI="${CPU_CASE}/${LATEST_TIME}/phi"
GPU_PHI="${GPU_CASE}/${LATEST_TIME}/phi"
CPU_U="${CPU_CASE}/${LATEST_TIME}/U"
GPU_U="${GPU_CASE}/${LATEST_TIME}/U"

TOLERANCE=${3:-0}

python3 - "$CPU_PHI" "$GPU_PHI" "$TOLERANCE" <<'PYCODE' || exit 2
import math
import sys

def read_field(path):
    with open(path) as fh:
        lines = fh.readlines()
    idx = 0
    n = len(lines)
    data = []
    while idx < n:
        line = lines[idx].strip()
        if line.startswith("internalField"):
            if "nonuniform" in line:
                idx += 1
                count = int(lines[idx].strip())
                idx += 1
                while idx < n and len(data) < count:
                    token = lines[idx].strip()
                    idx += 1
                    if not token or token in ("(", ")", ");"):
                        continue
                    token = token.replace(";", "")
                    if token.startswith("("):
                        vals = [float(x) for x in token.strip("()").split()]
                    else:
                        vals = [float(token)]
                    data.append(vals)
                return data
            elif "uniform" in line:
                value = line.split("uniform", 1)[1].split(";")[0].strip()
                if value.startswith("("):
                    vals = [float(x) for x in value.strip("()").split()]
                else:
                    vals = [float(value)]
                data.append(vals)
                return data
        idx += 1
    raise RuntimeError(f"No internalField found in {path}")

def norm2(entry):
    return sum(value*value for value in entry)

ref = read_field(sys.argv[1])
test = read_field(sys.argv[2])

if len(ref) != len(test):
    print(f"Size mismatch: {len(ref)} vs {len(test)}", file=sys.stderr)
    sys.exit(1)

sq_diff = 0.0
sq_ref = 0.0
for a, b in zip(ref, test):
    if len(a) != len(b):
        print("Entry size mismatch", file=sys.stderr)
        sys.exit(1)
    diff = [ai - bi for ai, bi in zip(a, b)]
    sq_diff += norm2(diff)
    sq_ref += norm2(a)

err = math.sqrt(sq_diff)
den = math.sqrt(sq_ref)
if den < 1e-30:
    den = 1.0
rel = err/den
print(f"phi relative L2 error: {rel:.6e}")
if rel > float(sys.argv[3]):
    sys.exit(1)
PYCODE

python3 - "$CPU_U" "$GPU_U" "$TOLERANCE" <<'PYCODE' || exit 3
import math
import sys

def read_field(path):
    with open(path) as fh:
        lines = fh.readlines()
    idx = 0
    n = len(lines)
    data = []
    while idx < n:
        line = lines[idx].strip()
        if line.startswith("internalField"):
            if "nonuniform" in line:
                idx += 1
                count = int(lines[idx].strip())
                idx += 1
                while idx < n and len(data) < count:
                    token = lines[idx].strip()
                    idx += 1
                    if not token or token in ("(", ")", ");"):
                        continue
                    token = token.replace(";", "")
                    if token.startswith("("):
                        vals = [float(x) for x in token.strip("()").split()]
                    else:
                        vals = [float(token)]
                    data.append(vals)
                return data
            elif "uniform" in line:
                value = line.split("uniform", 1)[1].split(";")[0].strip()
                if value.startswith("("):
                    vals = [float(x) for x in value.strip("()").split()]
                else:
                    vals = [float(value)]
                data.append(vals)
                return data
        idx += 1
    raise RuntimeError(f"No internalField found in {path}")

def norm2(entry):
    return sum(value*value for value in entry)

ref = read_field(sys.argv[1])
test = read_field(sys.argv[2])

if len(ref) != len(test):
    print(f"Size mismatch: {len(ref)} vs {len(test)}", file=sys.stderr)
    sys.exit(1)

sq_diff = 0.0
sq_ref = 0.0
for a, b in zip(ref, test):
    if len(a) != len(b):
        print("Entry size mismatch", file=sys.stderr)
        sys.exit(1)
    diff = [ai - bi for ai, bi in zip(a, b)]
    sq_diff += norm2(diff)
    sq_ref += norm2(a)

err = math.sqrt(sq_diff)
den = math.sqrt(sq_ref)
if den < 1e-30:
    den = 1.0
rel = err/den
print(f"U relative L2 error: {rel:.6e}")
if rel > float(sys.argv[3]):
    sys.exit(1)
PYCODE

RESULT_BASE="${REPO_ROOT}/run/parity_results/${CASE_NAME}"
rm -rf "${RESULT_BASE}"
mkdir -p "${RESULT_BASE}"

cp "${CPU_CASE}/log.pimpleFoamGPU" "${RESULT_BASE}/log.cpu"
cp "${GPU_CASE}/log.pimpleFoamGPU" "${RESULT_BASE}/log.gpu"

if [[ -d "${RESULT_BASE}/gpuFieldOps" ]]; then
    SUMMARY_FILE="${RESULT_BASE}/gpuFieldOps/summary.csv"
    if [[ -f "${SUMMARY_FILE}" ]]; then
        total_ms=$(awk -F',' 'NR>1 {sum+=$4} END {printf "%.6f", sum+0}' "${SUMMARY_FILE}")
        fallback_samples=$(awk -F',' 'NR>1 && $2 ~ /_cpu$/ {sum+=$3} END {printf "%d", sum+0}' "${SUMMARY_FILE}")
        {
            echo "gpuKernelTotalMilliseconds=${total_ms}"
            echo "gpuFallbackSamples=${fallback_samples}"
        } > "${RESULT_BASE}/summary.txt"
        echo "GPU kernel total (ms): ${total_ms}"
        if [[ "${fallback_samples}" -gt 0 ]]; then
            echo "GPU CPU fallbacks detected: ${fallback_samples}" >&2
        fi
    fi
fi

echo "GPU parity check passed at time ${LATEST_TIME}."
