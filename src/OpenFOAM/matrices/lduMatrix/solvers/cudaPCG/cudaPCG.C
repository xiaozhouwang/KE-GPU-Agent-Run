#include "cudaPCG.H"
#include "PCG.H"
#include "CsrExport.H"
#include "Switch.H"
#include "List.H"
#include "PstreamReduceOps.H"
#include "DynamicList.H"
#include "OFstream.H"
#include "OSspecific.H"
#include "Pstream.H"
#include "Time.H"

#include <algorithm>
#include <cmath>
#include <ios>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <unordered_map>

#ifdef FOAM_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#define CUSPARSE_USE_DEPRECATED_API
#include <cusparse_v2.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif

namespace Foam
{
    defineTypeNameAndDebug(cudaPCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<cudaPCG>
        addCudaPCGSymMatrixConstructorToTable_;
}

namespace
{
inline std::string toLower(const std::string& s)
{
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
    return out;
}

#ifdef FOAM_USE_CUDA

struct ColourOverrideEntry
{
    bool baseInitialised = false;
    bool active = false;
    int stage = 0;
    int successStreak = 0;
    Foam::scalar baseOmega = 0;
    Foam::scalar baseBackward = 0;
    Foam::scalar baseDiag = 0;
    Foam::scalar omega = 0;
    Foam::scalar backwardOmega = 0;
    Foam::scalar diagFloor = 0;
};

struct ColourOverrideRegistry
{
    std::mutex mutex;
    std::unordered_map<std::string, ColourOverrideEntry> entries;
};

ColourOverrideRegistry& colourOverrides()
{
    static ColourOverrideRegistry registry;
    return registry;
}

inline bool autoTuneEnabled(const Foam::dictionary& dict)
{
    return dict.lookupOrDefault<Foam::Switch>("colourAutoTune", Foam::Switch(true));
}

constexpr int maxColourStage = 3;
constexpr int colourDemoteThreshold = 5;

static void computeStageParams
(
    const ColourOverrideEntry& entry,
    const int stage,
    Foam::scalar& omega,
    Foam::scalar& backward,
    Foam::scalar& diag
)
{
    omega = entry.baseOmega;
    backward = entry.baseBackward;
    diag = entry.baseDiag;

    if (stage >= 1)
    {
        omega = std::min(omega, Foam::scalar(0.80));
        backward = std::min(backward, Foam::scalar(0.85));
        diag = std::max(diag, Foam::scalar(1e-9));
    }
    if (stage >= 2)
    {
        omega = std::min(omega, Foam::scalar(0.75));
        backward = std::min(backward, Foam::scalar(0.82));
        diag = std::max(diag, Foam::scalar(5e-9));
    }
    if (stage >= 3)
    {
        omega = std::min(omega, Foam::scalar(0.70));
        backward = std::min(backward, Foam::scalar(0.80));
        diag = std::max(diag, Foam::scalar(1e-8));
    }
    backward = std::max(backward, omega);
}

static void applyStage(ColourOverrideEntry& entry, int stage)
{
    stage = std::max(0, std::min(stage, maxColourStage));
    entry.stage = stage;
    entry.active = stage > 0;
    computeStageParams(entry, stage, entry.omega, entry.backwardOmega, entry.diagFloor);
    if (stage == 0)
    {
        entry.omega = entry.baseOmega;
        entry.backwardOmega = entry.baseBackward;
        entry.diagFloor = entry.baseDiag;
        entry.active = false;
    }
    entry.successStreak = 0;
}

static ColourOverrideEntry& accessOverride
(
    const Foam::word& field,
    const Foam::scalar baseOmega,
    const Foam::scalar baseBackward,
    const Foam::scalar baseDiag
)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    ColourOverrideEntry& entry = registry.entries[std::string(field)];
    if (!entry.baseInitialised)
    {
        entry.baseInitialised = true;
        entry.stage = 0;
        entry.successStreak = 0;
    }
    entry.baseOmega = baseOmega;
    entry.baseBackward = baseBackward;
    entry.baseDiag = baseDiag;
    if (entry.stage == 0)
    {
        entry.omega = baseOmega;
        entry.backwardOmega = baseBackward;
        entry.diagFloor = baseDiag;
    }
    return entry;
}

static void colourAutoTuneSetup
(
    const Foam::word& field,
    const Foam::scalar baseOmega,
    const Foam::scalar baseBackward,
    const Foam::scalar baseDiag,
    Foam::scalar& omega,
    Foam::scalar& backward,
    Foam::scalar& diag,
    int& stageOut
)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    ColourOverrideEntry& entry = registry.entries[std::string(field)];
    if (!entry.baseInitialised)
    {
        entry.baseInitialised = true;
        entry.stage = 0;
        entry.successStreak = 0;
    }
    entry.baseOmega = baseOmega;
    entry.baseBackward = baseBackward;
    entry.baseDiag = baseDiag;
    if (entry.stage == 0)
    {
        entry.omega = baseOmega;
        entry.backwardOmega = baseBackward;
        entry.diagFloor = baseDiag;
        entry.active = false;
    }
    else
    {
        computeStageParams(entry, entry.stage, entry.omega, entry.backwardOmega, entry.diagFloor);
    }

    omega = entry.omega;
    backward = entry.backwardOmega;
    diag = entry.diagFloor;
    stageOut = entry.stage;
}

static void colourAutoTunePromote
(
    const Foam::word& field,
    const Foam::scalar baseOmega,
    const Foam::scalar baseBackward,
    const Foam::scalar baseDiag,
    Foam::scalar& omega,
    Foam::scalar& backward,
    Foam::scalar& diag,
    int& stageOut
)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    ColourOverrideEntry& entry = registry.entries[std::string(field)];
    if (!entry.baseInitialised)
    {
        entry.baseInitialised = true;
        entry.stage = 0;
        entry.successStreak = 0;
        entry.baseOmega = baseOmega;
        entry.baseBackward = baseBackward;
        entry.baseDiag = baseDiag;
    }
    else
    {
        entry.baseOmega = baseOmega;
        entry.baseBackward = baseBackward;
        entry.baseDiag = baseDiag;
    }
    const int newStage = std::min(entry.stage + 1, maxColourStage);
    applyStage(entry, newStage);
    omega = entry.omega;
   backward = entry.backwardOmega;
   diag = entry.diagFloor;
   stageOut = entry.stage;
}

static void stageThresholds
(
    const int stage,
    const Foam::label baseWindow,
    const Foam::label baseBestWindow,
    const Foam::scalar baseRatio,
    const Foam::scalar baseBestRatio,
    Foam::label& outWindow,
    Foam::label& outBestWindow,
    Foam::scalar& outRatio,
    Foam::scalar& outBestRatio
)
{
    outWindow = baseWindow;
    outBestWindow = baseBestWindow;
    outRatio = baseRatio;
    outBestRatio = baseBestRatio;

    if (stage >= 1)
    {
        outRatio = std::max(outRatio, Foam::scalar(0.9995));
        outBestRatio = std::max(outBestRatio, Foam::scalar(1.02));
        outWindow = std::max(outWindow, Foam::label(400));
        outBestWindow = std::max(outBestWindow, Foam::label(400));
    }
    if (stage >= 2)
    {
        outRatio = std::max(outRatio, Foam::scalar(0.9997));
        outBestRatio = std::max(outBestRatio, Foam::scalar(1.03));
        outWindow = std::max(outWindow, Foam::label(600));
        outBestWindow = std::max(outBestWindow, Foam::label(600));
    }
    if (stage >= 3)
    {
        outRatio = std::max(outRatio, Foam::scalar(0.9998));
        outBestRatio = std::max(outBestRatio, Foam::scalar(1.04));
        outWindow = std::max(outWindow, Foam::label(800));
        outBestWindow = std::max(outBestWindow, Foam::label(800));
    }
}

static void colourAutoTuneRecordSuccess(const Foam::word& field)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto iter = registry.entries.find(std::string(field));
    if (iter == registry.entries.end())
    {
        return;
    }
    ColourOverrideEntry& entry = iter->second;
    if (entry.stage == 0)
    {
        entry.successStreak = std::min(entry.successStreak + 1, colourDemoteThreshold);
        return;
    }
    entry.successStreak++;
    if (entry.successStreak >= colourDemoteThreshold)
    {
        const int newStage = std::max(entry.stage - 1, 0);
        applyStage(entry, newStage);
    }
}

static void colourAutoTuneRecordFailure(const Foam::word& field)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto iter = registry.entries.find(std::string(field));
    if (iter == registry.entries.end())
    {
        return;
    }
    iter->second.successStreak = 0;
}

static void colourAutoTuneReset(const Foam::word& field)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto iter = registry.entries.find(std::string(field));
    if (iter == registry.entries.end())
    {
        return;
    }
    applyStage(iter->second, 0);
}

static int colourAutoTuneStage(const Foam::word& field)
{
    auto& registry = colourOverrides();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto iter = registry.entries.find(std::string(field));
    if (iter == registry.entries.end())
    {
        return 0;
    }
    return iter->second.stage;
}

static void normaliseCpuPreconditioner(Foam::dictionary& dict)
{
    const Foam::word precon = dict.lookupOrDefault<Foam::word>("preconditioner", Foam::word("DIC"));
    const Foam::word lower = toLower(precon);

    if (lower == Foam::word("sgs") || lower == Foam::word("symgaussseidel")
     || lower == Foam::word("ic") || lower == Foam::word("ic0") || lower == Foam::word("dic")
     || lower == Foam::word("colour") || lower == Foam::word("colored") || lower == Foam::word("coloured"))
    {
        dict.set("preconditioner", Foam::word("DIC"));
    }
    else if (lower == Foam::word("fdic"))
    {
        dict.set("preconditioner", Foam::word("FDIC"));
    }
    else if (lower == Foam::word("ilu0") || lower == Foam::word("ilu"))
    {
        dict.set("preconditioner", Foam::word("DIC"));
    }
    else if (lower == Foam::word("gamg"))
    {
        dict.set("preconditioner", Foam::word("GAMG"));
    }
    else if (lower == Foam::word("diagonal") || lower == Foam::word("none"))
    {
        dict.set("preconditioner", lower);
    }

    dict.remove("colourOmega");
    dict.remove("colourBackwardOmega");
    dict.remove("colourDiagFloor");
    dict.remove("stallRatioTol");
    dict.remove("stallBestRatioTol");
    dict.remove("stallWindow");
    dict.remove("stallBestWindow");
    dict.remove("polyJacobiAuto");
    dict.remove("polyJacobiSweeps");
    dict.remove("jacobiOmega");
}

inline const char* cuStatusName(CUresult code)
{
    const char* name = nullptr;
    if (code == CUDA_SUCCESS)
    {
        return "CUDA_SUCCESS";
    }
    if (cuGetErrorName(code, &name) != CUDA_SUCCESS || !name)
    {
        return "CUDA_ERROR_UNKNOWN";
    }
    return name;
}

inline const char* nvrtcStatusName(nvrtcResult code)
{
    return nvrtcGetErrorString(code);
}

static const char* colouredKernelSource = R"(
#include <cuda_runtime.h>
#include <math_functions.h>

extern "C" __global__
void colouredForwardSweep(
    int count,
    const int* __restrict__ rowIndices,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colInd,
    const double* __restrict__ values,
    const int* __restrict__ colours,
    const double* __restrict__ rhs,
    double* __restrict__ x,
    int colourId,
    double omega,
    double diagFloor,
    int* __restrict__ failFlag
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count)
    {
        return;
    }

    const int row = rowIndices[tid];
    double sum = rhs[row];
    double diag = 0.0;

    for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k)
    {
        const int col = colInd[k];
        const double val = values[k];
        if (col == row)
        {
            diag = val;
        }
        else if (colours[col] < colourId)
        {
            sum -= val * x[col];
        }
    }

    if (fabs(diag) <= diagFloor)
    {
        atomicMax(failFlag, 1);
        return;
    }

    const double newVal = sum / diag;
    const double oldVal = x[row];
    const double updated = oldVal + omega * (newVal - oldVal);
    if (!isfinite(updated))
    {
        atomicMax(failFlag, 2);
    }
    x[row] = updated;
}

extern "C" __global__
void colouredBackwardSweep(
    int count,
    const int* __restrict__ rowIndices,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colInd,
    const double* __restrict__ values,
    const int* __restrict__ colours,
    const double* __restrict__ rhs,
    double* __restrict__ x,
    int colourId,
    double omega,
    double diagFloor,
    int* __restrict__ failFlag
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count)
    {
        return;
    }

    const int row = rowIndices[tid];
    double sum = rhs[row];
    double diag = 0.0;

    for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k)
    {
        const int col = colInd[k];
        const double val = values[k];
        if (col == row)
        {
            diag = val;
        }
        else if (colours[col] > colourId)
        {
            sum -= val * x[col];
        }
    }

    if (fabs(diag) <= diagFloor)
    {
        atomicMax(failFlag, 1);
        return;
    }

    const double newVal = sum / diag;
    const double oldVal = x[row];
    const double updated = oldVal + omega * (newVal - oldVal);
    if (!isfinite(updated))
    {
        atomicMax(failFlag, 2);
    }
    x[row] = updated;
}
)";

struct ColourKernelCacheEntry
{
    int major;
    int minor;
    CUmodule module;
    CUfunction forward;
    CUfunction backward;
    bool ok;
};

static std::mutex colourKernelMutex;
static std::vector<ColourKernelCacheEntry> colourKernelCache;
static std::once_flag driverInitFlag;
static CUresult driverInitStatus = CUDA_SUCCESS;

inline bool ensureDriverInitialised()
{
    std::call_once
    (
        driverInitFlag,
        []()
        {
            driverInitStatus = cuInit(0);
        }
    );

    return driverInitStatus == CUDA_SUCCESS;
}

bool compileColourKernels
(
    const cudaDeviceProp& prop,
    ColourKernelCacheEntry& entry,
    bool verbose
)
{
    entry.major = prop.major;
    entry.minor = prop.minor;
    entry.module = nullptr;
    entry.forward = nullptr;
    entry.backward = nullptr;
    entry.ok = false;

    if (!ensureDriverInitialised())
    {
        if (verbose)
        {
            WarningInFunction << "cuInit failed: " << cuStatusName(driverInitStatus) << Foam::endl;
        }
        return false;
    }

    nvrtcProgram prog = nullptr;
    nvrtcResult nvRes =
        nvrtcCreateProgram
        (
            &prog,
            colouredKernelSource,
            "colouredPreconditioner.cu",
            0,
            nullptr,
            nullptr
        );

    if (nvRes != NVRTC_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "nvrtcCreateProgram failed: " << nvrtcStatusName(nvRes) << Foam::endl;
        }
        return false;
    }

    std::ostringstream archOpt;
    archOpt << "--gpu-architecture=compute_" << prop.major << prop.minor;
    const std::string archStr = archOpt.str();

    std::vector<const char*> compileOptions;
    compileOptions.push_back(archStr.c_str());
    compileOptions.push_back("--fmad=false");
    compileOptions.push_back("--std=c++14");
    compileOptions.push_back("-I/usr/include");
    compileOptions.push_back("-I/usr/local/cuda/include");
    compileOptions.push_back("-I/usr/include/x86_64-linux-gnu");

    nvRes = nvrtcCompileProgram
    (
        prog,
        static_cast<int>(compileOptions.size()),
        compileOptions.data()
    );

    if (nvRes != NVRTC_SUCCESS)
    {
        if (verbose)
        {
            size_t logSize = 0;
            nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            nvrtcGetProgramLog(prog, &log[0]);
            WarningInFunction
                << "nvrtcCompileProgram failed (" << nvrtcStatusName(nvRes) << "):"
                << Foam::nl << log.c_str() << Foam::endl;
        }
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    nvRes = nvrtcGetPTXSize(prog, &ptxSize);
    if (nvRes != NVRTC_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "nvrtcGetPTXSize failed: " << nvrtcStatusName(nvRes) << Foam::endl;
        }
        nvrtcDestroyProgram(&prog);
        return false;
    }

    std::string ptx(ptxSize, '\0');
    nvRes = nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);
    if (nvRes != NVRTC_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "nvrtcGetPTX failed: " << nvrtcStatusName(nvRes) << Foam::endl;
        }
        return false;
    }

    CUresult cuRes =
        cuModuleLoadDataEx
        (
            &entry.module,
            static_cast<const void*>(ptx.data()),
            0,
            nullptr,
            nullptr
        );

    if (cuRes != CUDA_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "cuModuleLoadDataEx failed: " << cuStatusName(cuRes) << Foam::endl;
        }
        entry.module = nullptr;
        return false;
    }

    cuRes = cuModuleGetFunction(&entry.forward, entry.module, "colouredForwardSweep");
    if (cuRes != CUDA_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "cuModuleGetFunction forward failed: " << cuStatusName(cuRes) << Foam::endl;
        }
        cuModuleUnload(entry.module);
        entry.module = nullptr;
        return false;
    }

    cuRes = cuModuleGetFunction(&entry.backward, entry.module, "colouredBackwardSweep");
    if (cuRes != CUDA_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "cuModuleGetFunction backward failed: " << cuStatusName(cuRes) << Foam::endl;
        }
        cuModuleUnload(entry.module);
        entry.module = nullptr;
        entry.forward = nullptr;
        return false;
    }

    entry.ok = true;
    return true;
}

bool getColourKernels
(
    const int deviceId,
    CUfunction& forward,
    CUfunction& backward,
    bool verbose
)
{
    std::lock_guard<std::mutex> guard(colourKernelMutex);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    {
        if (verbose)
        {
            WarningInFunction << "cudaGetDeviceProperties failed for device " << deviceId << Foam::endl;
        }
        return false;
    }

    for (const ColourKernelCacheEntry& entry : colourKernelCache)
    {
        if (entry.major == prop.major && entry.minor == prop.minor)
        {
            if (entry.ok)
            {
                forward = entry.forward;
                backward = entry.backward;
                return true;
            }
            return false;
        }
    }

    ColourKernelCacheEntry entry;
    if (compileColourKernels(prop, entry, verbose))
    {
        forward = entry.forward;
        backward = entry.backward;
        colourKernelCache.push_back(entry);
        return true;
    }

    colourKernelCache.push_back(entry);
    return false;
}

#endif // FOAM_USE_CUDA
}

Foam::cudaPCG::cudaPCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}

Foam::solverPerformance Foam::cudaPCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    bool requestedGpu = false;

    Switch gpuSwitch(false);
    if (gpuSwitch.readIfPresent("useGPU", controlDict_))
    {
        requestedGpu = gpuSwitch;
    }

    if (!requestedGpu)
    {
        const word deviceWord = controlDict_.lookupOrDefault<word>("device", "gpu");
        const std::string deviceLower = toLower(deviceWord);
        requestedGpu = deviceLower == "gpu" || deviceLower == "cuda";

        if (!requestedGpu && controlDict_.found("backend"))
        {
            const word backend = controlDict_.lookup<word>("backend");
            const std::string backendLower = toLower(backend);
            requestedGpu = backendLower == "gpu" || backendLower == "cuda";
        }
    }

    const label requestedDevice = controlDict_.lookupOrDefault<label>("gpuDevice", 0);
    const label deviceId = controlDict_.lookupOrDefault<label>("deviceId", requestedDevice);
    const Switch hybridCpuCorrection =
        controlDict_.lookupOrDefault<Switch>("hybridCpuCorrection", Switch(false));
    const label hybridStageThreshold =
        std::max<label>(0, controlDict_.lookupOrDefault<label>("hybridStageThreshold", 1));
    const scalar hybridBlend =
        std::min<scalar>(1.0, std::max<scalar>(0.0, controlDict_.lookupOrDefault<scalar>("hybridBlend", scalar(1))));

#ifdef FOAM_USE_CUDA
    if (requestedGpu)
    {
        solverPerformance gpuPerf
        (
            lduMatrix::preconditioner::getName(controlDict_) + typeName,
            fieldName_
        );

        word message;
        if (gpuSolve(psi, source, cmpt, gpuPerf, deviceId, message, controlDict_))
        {
            if (hybridCpuCorrection)
            {
                const int stage = colourAutoTuneStage(fieldName_);
                if (stage >= hybridStageThreshold)
                {
                    dictionary hybridCpuDict(controlDict_);
                    hybridCpuDict.set("solver", word("PCG"));
                    normaliseCpuPreconditioner(hybridCpuDict);
                    autoPtr<lduMatrix::solver> hybridSolver = lduMatrix::solver::New
                    (
                        fieldName_,
                        matrix_,
                        interfaceBouCoeffs_,
                        interfaceIntCoeffs_,
                        interfaces_,
                        hybridCpuDict
                    );
                    scalarField gpuPsi(psi);
                    solverPerformance cpuPerf = hybridSolver->solve(psi, source, cmpt);
                    if (hybridBlend > 0 && hybridBlend < 1)
                    {
                        const scalar alpha = hybridBlend;
                        const scalar beta = 1 - alpha;
                        forAll(psi, i)
                        {
                            psi[i] = alpha*psi[i] + beta*gpuPsi[i];
                        }
                    }
                    gpuPerf.initialResidual() = cpuPerf.initialResidual();
                    gpuPerf.finalResidual() = cpuPerf.finalResidual();
                    gpuPerf.nIterations() += cpuPerf.nIterations();
                    InfoInFunction
                        << "cudaPCG hybrid CPU correction stage=" << stage
                        << " iterations=" << cpuPerf.nIterations() << nl;
                }
            }
            return gpuPerf;
        }

        if (message == word("stallDetected"))
        {
            const word preconWord = controlDict_.lookupOrDefault<word>("preconditioner", word("colour"));
            const std::string preconLower = toLower(preconWord);
            const bool colourPrecon =
                preconLower.find("colour") != std::string::npos
             || preconLower.find("color") != std::string::npos
             || preconLower.find("composite") != std::string::npos;

            if (colourPrecon)
            {
                const scalar baseColourOmega = controlDict_.lookupOrDefault<scalar>("colourOmega", scalar(0.65));
                const scalar baseColourBackward = controlDict_.lookupOrDefault<scalar>("colourBackwardOmega", scalar(0.85));
                const scalar baseColourDiag = controlDict_.lookupOrDefault<scalar>("colourDiagFloor", scalar(1e-12));
                const label baseStallWindow = controlDict_.lookupOrDefault<label>("stallWindow", 50);
                const label baseStallBestWindow = controlDict_.lookupOrDefault<label>("stallBestWindow", baseStallWindow);
                const scalar baseStallRatioTol = controlDict_.lookupOrDefault<scalar>("stallRatioTol", scalar(0.99));
                const scalar baseStallBestRatioTol = controlDict_.lookupOrDefault<scalar>("stallBestRatioTol", scalar(1.001));
                dictionary retryDict(controlDict_);
                scalar tunedOmega = baseColourOmega;
                scalar tunedBackward = baseColourBackward;
                scalar tunedDiag = baseColourDiag;

                if (autoTuneEnabled(controlDict_))
                {
                    int retryStage = 0;
                    colourAutoTunePromote
                    (
                        fieldName_,
                        baseColourOmega,
                        baseColourBackward,
                        baseColourDiag,
                        tunedOmega,
                        tunedBackward,
                        tunedDiag,
                        retryStage
                    );
                    Foam::label tunedStallWindow = baseStallWindow;
                    Foam::label tunedStallBestWindow = baseStallBestWindow;
                    Foam::scalar tunedStallRatio = baseStallRatioTol;
                    Foam::scalar tunedStallBestRatio = baseStallBestRatioTol;
                    stageThresholds
                    (
                        retryStage,
                        baseStallWindow,
                        baseStallBestWindow,
                        baseStallRatioTol,
                        baseStallBestRatioTol,
                        tunedStallWindow,
                        tunedStallBestWindow,
                        tunedStallRatio,
                        tunedStallBestRatio
                    );
                    retryDict.set("stallWindow", tunedStallWindow);
                    retryDict.set("stallBestWindow", tunedStallBestWindow);
                    retryDict.set("stallRatioTol", tunedStallRatio);
                    retryDict.set("stallBestRatioTol", tunedStallBestRatio);
                }
                else
                {
                    tunedOmega = scalar(std::max<scalar>(0.5, std::min<scalar>(baseColourOmega, scalar(0.8))));
                    tunedBackward = scalar(std::max<scalar>(0.5, std::min<scalar>(baseColourBackward, scalar(0.85))));
                    tunedDiag = scalar(std::max<scalar>(baseColourDiag, scalar(1e-9)));
                    Foam::label tunedStallWindow = baseStallWindow;
                    Foam::label tunedStallBestWindow = baseStallBestWindow;
                    Foam::scalar tunedStallRatio = baseStallRatioTol;
                    Foam::scalar tunedStallBestRatio = baseStallBestRatioTol;
                    stageThresholds
                    (
                        1,
                        baseStallWindow,
                        baseStallBestWindow,
                        baseStallRatioTol,
                        baseStallBestRatioTol,
                        tunedStallWindow,
                        tunedStallBestWindow,
                        tunedStallRatio,
                        tunedStallBestRatio
                    );
                    retryDict.set("stallRatioTol", tunedStallRatio);
                    retryDict.set("stallBestRatioTol", tunedStallBestRatio);
                    retryDict.set("stallWindow", tunedStallWindow);
                    retryDict.set("stallBestWindow", tunedStallBestWindow);
                }

                retryDict.set("colourOmega", tunedOmega);
                retryDict.set("colourBackwardOmega", tunedBackward);
                retryDict.set("colourDiagFloor", tunedDiag);

                solverPerformance retryPerf
                (
                    lduMatrix::preconditioner::getName(retryDict) + typeName + "::retry",
                    fieldName_
                );

                word retryMessage;
                if (gpuSolve(psi, source, cmpt, retryPerf, deviceId, retryMessage, retryDict))
                {
                    if (autoTuneEnabled(controlDict_) && colourPrecon)
                    {
                        colourAutoTuneRecordSuccess(fieldName_);
                    }
                    if (hybridCpuCorrection)
                    {
                        const int stage = colourAutoTuneStage(fieldName_);
                        if (stage >= hybridStageThreshold)
                        {
                        dictionary hybridCpuDict(controlDict_);
                        hybridCpuDict.set("solver", word("PCG"));
                        normaliseCpuPreconditioner(hybridCpuDict);
                        autoPtr<lduMatrix::solver> hybridSolver = lduMatrix::solver::New
                        (
                            fieldName_,
                            matrix_,
                            interfaceBouCoeffs_,
                            interfaceIntCoeffs_,
                            interfaces_,
                            hybridCpuDict
                            );
                            scalarField gpuPsi(psi);
                            solverPerformance cpuPerf = hybridSolver->solve(psi, source, cmpt);
                            if (hybridBlend > 0 && hybridBlend < 1)
                            {
                                const scalar alpha = hybridBlend;
                                const scalar beta = 1 - alpha;
                                forAll(psi, i)
                                {
                                    psi[i] = alpha*psi[i] + beta*gpuPsi[i];
                                }
                            }
                            retryPerf.initialResidual() = cpuPerf.initialResidual();
                            retryPerf.finalResidual() = cpuPerf.finalResidual();
                            retryPerf.nIterations() += cpuPerf.nIterations();
                            InfoInFunction
                                << "cudaPCG hybrid CPU correction stage=" << stage
                                << " iterations=" << cpuPerf.nIterations() << nl;
                        }
                    }
                    InfoInFunction
                        << "cudaPCG retry succeeded after stall; colourOmega="
                        << retryDict.lookupOrDefault<scalar>("colourOmega", scalar(0))
                        << " colourDiagFloor="
                        << retryDict.lookupOrDefault<scalar>("colourDiagFloor", scalar(0))
                        << nl;
                    return retryPerf;
                }
                else
                {
                    message = retryMessage;
                    if (autoTuneEnabled(controlDict_) && colourPrecon)
                    {
                        colourAutoTuneRecordFailure(fieldName_);
                    }
                    WarningInFunction
                        << "cudaPCG retry attempt failed: " << message << nl;
                }
            }
        }

        WarningInFunction
            << "cudaPCG GPU path disabled: " << message << nl
            << "Falling back to CPU PCG." << endl;
        if (autoTuneEnabled(controlDict_))
        {
            const word preconWord = controlDict_.lookupOrDefault<word>("preconditioner", word("colour"));
            const std::string preconLowerFinal = toLower(preconWord);
            if
            (
                preconLowerFinal.find("colour") != std::string::npos
             || preconLowerFinal.find("color") != std::string::npos
             || preconLowerFinal.find("composite") != std::string::npos
            )
            {
                colourAutoTuneRecordFailure(fieldName_);
            }
        }
    }
#else
    if (requestedGpu)
    {
        WarningInFunction
            << "cudaPCG requested GPU backend but this build lacks CUDA support" << nl
            << "Continuing with CPU solver." << endl;
    }
#endif

    dictionary cpuDict(controlDict_);
    cpuDict.set("solver", word("PCG"));
    normaliseCpuPreconditioner(cpuDict);

    autoPtr<lduMatrix::solver> cpuSolver = lduMatrix::solver::New
    (
        fieldName_,
        matrix_,
        interfaceBouCoeffs_,
        interfaceIntCoeffs_,
        interfaces_,
        cpuDict
    );

    return cpuSolver->solve(psi, source, cmpt);
}

#ifdef FOAM_USE_CUDA

namespace
{
inline const char* cudaStatusName(cudaError_t code)
{
    return cudaGetErrorName(code);
}

inline const char* cusparseStatusName(cusparseStatus_t code)
{
    return cusparseGetErrorString(code);
}

inline const char* cublasStatusName(cublasStatus_t code)
{
    switch (code)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#ifdef CUBLAS_STATUS_NOT_SUPPORTED
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#ifdef CUBLAS_STATUS_LICENSE_ERROR
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}
} // namespace

bool Foam::cudaPCG::gpuSolve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt,
        solverPerformance& solverPerf,
        const label deviceId,
        word& message,
        const dictionary& dict
    ) const
{
    message.clear();

    const label nCells = psi.size();
    if (!nCells)
    {
        solverPerf.finalResidual() = 0;
        solverPerf.initialResidual() = 0;
        return true;
    }

    const bool reportStats =
        dict.lookupOrDefault<Switch>("reportGpuStats", Switch(false));

    const Switch logResidualTrajectorySwitch =
        dict.lookupOrDefault<Switch>("logResidualTrajectory", Switch(false));
    const bool logResidualTrajectory = bool(logResidualTrajectorySwitch);

    const Switch pipelinedSwitch =
        dict.lookupOrDefault<Switch>("usePipelinedCG", Switch(false));
    const bool usePipelinedCG = bool(pipelinedSwitch);

    const Switch cudaGraphSwitch =
        dict.lookupOrDefault<Switch>("useCudaGraph", Switch(false));
    const bool useCudaGraph = bool(cudaGraphSwitch);
    const label cudaGraphWarmup =
        std::max<label>(0, dict.lookupOrDefault<label>("cudaGraphWarmup", 5));

    const Switch iterStatsSwitch =
        dict.lookupOrDefault<Switch>("logIterationStats", Switch(false));
    const bool logIterationStats = bool(iterStatsSwitch);

    const label residualLogEvery =
        std::max<label>(1, dict.lookupOrDefault<label>("residualLogEvery", 1));

    fileName residualLogFile =
        dict.lookupOrDefault<fileName>("residualLogFile", fileName::null);

    const Switch logColourStatsSwitch =
        dict.lookupOrDefault<Switch>("logColourStats", Switch(reportStats));
    const bool logColourStats = bool(logColourStatsSwitch);

    fileName colourStatsFile =
        dict.lookupOrDefault<fileName>("colourStatsFile", fileName::null);

    if (logResidualTrajectory && residualLogFile == fileName::null)
    {
        residualLogFile =
            fileName("postProcessing/cudaPCG/" + fieldName_ + "_residual.csv");
    }

    if ((logColourStats || reportStats) && colourStatsFile == fileName::null)
    {
        colourStatsFile =
            fileName("postProcessing/cudaPCG/" + fieldName_ + "_colour.csv");
    }

    std::vector<label> residualIterHistory;
    std::vector<scalar> residualValueHistory;
    if (logResidualTrajectory)
    {
        residualIterHistory.reserve(maxIter_ + 2);
        residualValueHistory.reserve(maxIter_ + 2);
    }

    bool telemetryWritten = false;
    bool colourInitiallyRequested = false;
    bool colourBuiltSuccessfully = false;
    bool colourAppliedSuccessfully = false;
    label colourSetCount = 0;
    label colourMinSize = 0;
    label colourMaxSize = 0;
    scalar colourAvgSize = 0;
    label colourDisableCount = 0;
    std::vector<std::string> colourDisableEvents;
    bool colourFallbackTriggered = false;

    cudaStream_t computeStream = nullptr;
    cudaGraph_t spmvGraph = nullptr;
    cudaGraphExec_t spmvGraphExec = nullptr;
    bool spmvGraphReady = false;

    double setupTimeSeconds = 0.0;
   double iterationAccumSeconds = 0.0;
   double iterationMaxSeconds = 0.0;
   label iterationCountStats = 0;
    label stallInitCounter = 0;
    label stallWindowCounter = 0;
    std::deque<scalar> residualWindow;
    scalar bestResidualSoFar = std::numeric_limits<scalar>::max();

    cudaError_t cudaCode = cudaSetDevice(deviceId);
    if (cudaCode != cudaSuccess)
    {
        message = word("cudaSetDevice: ") + cudaStatusName(cudaCode);
        return false;
    }

    cudaPCGDetail::CsrMatrix csrMatrix = cudaPCGDetail::buildSymmetricCsr(matrix_);
    if (csrMatrix.nRows != nCells)
    {
        message = "CSR row mismatch";
        return false;
    }

    static_assert(sizeof(scalar) == sizeof(double), "cudaPCG currently requires double precision build");

using DeviceScalar = double;

    const int rows = static_cast<int>(csrMatrix.nRows);
    const int nnz = static_cast<int>(csrMatrix.nnz);

    List<int> rowPtr(rows + 1);
    List<int> colInd(nnz);
    List<DeviceScalar> values(nnz);

    for (int i=0; i<=rows; ++i) rowPtr[i] = static_cast<int>(csrMatrix.rowPtr[i]);
    for (int i=0; i<nnz; ++i)
    {
        colInd[i] = static_cast<int>(csrMatrix.colInd[i]);
        values[i] = static_cast<DeviceScalar>(csrMatrix.values[i]);
    }

    const scalarField& diag = matrix_.diag();
    const scalar diagFloorAbsCfg =
        dict.lookupOrDefault<scalar>("preconditionerDiagFloor", scalar(1e-20));
    const scalar diagFloorRelCfg =
        dict.lookupOrDefault<scalar>("preconditionerDiagRelFloor", scalar(0));

    scalar minDiag = std::numeric_limits<scalar>::max();
    scalar maxDiag = -std::numeric_limits<scalar>::max();
    scalar maxAbsDiag = 0;
    for (const scalar d : diag)
    {
        minDiag = std::min(minDiag, d);
        maxDiag = std::max(maxDiag, d);
        maxAbsDiag = std::max(maxAbsDiag, mag(d));
    }

    const scalar diagFloorCfg = std::max(diagFloorAbsCfg, diagFloorRelCfg*maxAbsDiag);

    const labelUList& upperAddr = matrix_.lduAddr().upperAddr();
    const labelUList& lowerAddr = matrix_.lduAddr().lowerAddr();
    const scalarField& upper = matrix_.upper();

    scalarField dicDiag(diag);
    forAll(dicDiag, i)
    {
        scalar& d = dicDiag[i];
        d = std::max(mag(d), diagFloorCfg);
    }

    const label nFaces = upper.size();
    for (label face = 0; face < nFaces; ++face)
    {
        const label upCell = upperAddr[face];
        const label lowCell = lowerAddr[face];
        const scalar lowDiag = dicDiag[lowCell];
        const scalar denom = std::max(mag(lowDiag), diagFloorCfg);
        dicDiag[upCell] -= (upper[face]*upper[face])/denom;
    }

    scalar minDiagReg = std::numeric_limits<scalar>::max();
    scalar maxDiagReg = -std::numeric_limits<scalar>::max();
    List<DeviceScalar> precondDiag(dicDiag.size());
    List<DeviceScalar> diagInv(dicDiag.size());

    forAll(dicDiag, i)
    {
        scalar reg = std::max(mag(dicDiag[i]), diagFloorCfg);
        minDiagReg = std::min(minDiagReg, reg);
        maxDiagReg = std::max(maxDiagReg, reg);
        precondDiag[i] = static_cast<DeviceScalar>(reg);
        diagInv[i] = mag(reg) > VSMALL
            ? static_cast<DeviceScalar>(1.0/reg)
            : static_cast<DeviceScalar>(0);
    }

    if (reportStats)
    {
        Info<< "cudaPCG: diagonal range [" << minDiag << ", " << maxDiag << "]"
            << " (regularised [" << minDiagReg << ", " << maxDiagReg << "])" << nl;
    }

    DeviceScalar *d_vals=nullptr, *d_x=nullptr, *d_b=nullptr;
    DeviceScalar *d_r=nullptr, *d_p=nullptr, *d_Ap=nullptr;
    DeviceScalar *d_z=nullptr, *d_diagInv=nullptr;
    int *d_rowPtr=nullptr, *d_colInd=nullptr;
    int *d_colors=nullptr, *d_colorIndices=nullptr;
    int *d_failFlag=nullptr;
    // ILU/IC resources
    DeviceScalar *d_iluVals = nullptr; // values for ILU/IC factors (separate from d_vals)
    cusparseMatDescr_t iluDescr = nullptr;
    csrilu02Info_t iluInfo = nullptr; // ILU0 info
    csric02Info_t icInfo = nullptr;   // IC0 info
    cusparseSpMatDescr_t matL = nullptr, matU = nullptr;      // ILU L and U
    cusparseSpMatDescr_t matLchol = nullptr;                  // IC0 L
    cusparseSpSVDescr_t spsvLowerIlu = nullptr, spsvUpperIlu = nullptr;
    cusparseSpSVDescr_t spsvLowerChol = nullptr, spsvUpperTChol = nullptr;
    void* iluBuf = nullptr; size_t iluBufSize = 0;

    // Preconditioner configuration
    const word preconditionerWord = dict.lookupOrDefault<word>("preconditioner", word("diagonal"));
    const std::string precondLower = toLower(preconditionerWord);

    bool useColour = false;
    bool useSGS = false;
    bool useILU = false;
    bool useIC = false;
    bool useComposite = false;

    if
    (
        precondLower == "colour"
     || precondLower == "coloured"
     || precondLower == "colored"
     || precondLower == "multicolour"
     || precondLower == "multicolor"
     || precondLower == "colorgs"
     || precondLower == "colourgs"
     || precondLower.find("colour+ic") != std::string::npos
     || precondLower.find("color+ic") != std::string::npos
    )
    {
        useColour = true;
    }
    else if (precondLower == "sgs" || precondLower == "symgaussseidel")
    {
        useSGS = true;
    }
    else if (precondLower == "ilu0" || precondLower == "ilu")
    {
        useILU = true;
    }
    else if (precondLower == "ic0" || precondLower == "ic" || precondLower == "dic" || precondLower.find("+ic") != std::string::npos)
    {
        // Map DIC to IC0 on GPU for closer parity with CPU
        useIC = true;
    }
    if (precondLower == "composite" || precondLower.find("colour+ic") != std::string::npos || precondLower.find("color+ic") != std::string::npos)
    {
        useComposite = true;
    }

    colourInitiallyRequested = useColour;

    label polySweepsCfg = dict.lookupOrDefault<label>("polyJacobiSweeps", 0);
    scalar jacOmegaCfg = dict.lookupOrDefault<scalar>("jacobiOmega", scalar(0.7));
    const Switch polyAuto = dict.lookupOrDefault<Switch>("polyJacobiAuto", Switch(true));
    bool usePolyJacobi = (!useColour && !useSGS && !useILU && !useIC) && ((polySweepsCfg > 0) || polyAuto);
    label polySweepsEff = polySweepsCfg;
    double jacOmegaEff = static_cast<double>(jacOmegaCfg);

    const Switch earlyAbortOnStall =
        dict.lookupOrDefault<Switch>("earlyAbortOnStall", Switch(false));
    label stallWindow =
        std::max<label>(1, dict.lookupOrDefault<label>("stallWindow", 50));
    scalar stallRatioTol =
        dict.lookupOrDefault<scalar>("stallRatioTol", scalar(0.99));
    label stallBestWindow =
        std::max<label>
        (
            1,
            dict.lookupOrDefault<label>("stallBestWindow", stallWindow)
        );
    scalar stallBestRatioTol =
        dict.lookupOrDefault<scalar>("stallBestRatioTol", scalar(1.001));
    const label baseStallWindow = stallWindow;
    const label baseStallBestWindow = stallBestWindow;
    const scalar baseStallRatioTol = stallRatioTol;
    const scalar baseStallBestRatioTol = stallBestRatioTol;

    const Switch equilibrateMatrix = dict.lookupOrDefault<Switch>("equilibrateMatrix", Switch(false));
    scalar colourOmegaCfg = dict.lookupOrDefault<scalar>("colourOmega", scalar(0.65));
    scalar colourBackwardOmegaCfg = dict.lookupOrDefault<scalar>("colourBackwardOmega", scalar(0.85));
    scalar colourDiagFloorCfg = dict.lookupOrDefault<scalar>("colourDiagFloor", scalar(1e-12));
    const scalar baseColourOmega = colourOmegaCfg;
    const scalar baseColourBackward = colourBackwardOmegaCfg;
    const scalar baseColourDiag = colourDiagFloorCfg;
    if (useColour)
    {
        int colourStage = 0;
        if (autoTuneEnabled(dict))
        {
            colourAutoTuneSetup
            (
                fieldName_,
                baseColourOmega,
                baseColourBackward,
                baseColourDiag,
                colourOmegaCfg,
                colourBackwardOmegaCfg,
                colourDiagFloorCfg,
                colourStage
            );
            stageThresholds
            (
                colourStage,
                baseStallWindow,
                baseStallBestWindow,
                baseStallRatioTol,
                baseStallBestRatioTol,
                stallWindow,
                stallBestWindow,
                stallRatioTol,
                stallBestRatioTol
            );
        }
        else
        {
            colourAutoTuneReset(fieldName_);
        }
    }
    else if (autoTuneEnabled(dict))
    {
        colourAutoTuneReset(fieldName_);
    }
    const label colourBlockSizeCfg =
        std::max<label>
        (
            32,
            std::min<label>(1024, dict.lookupOrDefault<label>("colourBlockSize", 128))
        );
    const bool colourVerbose =
        dict.lookupOrDefault<Switch>("colourVerbose", Switch(false));

    const double colourOmegaEff = std::max(0.0, std::min(1.0, static_cast<double>(colourOmegaCfg)));
    const double colourBackwardOmegaEff = std::max(0.0, std::min(1.0, static_cast<double>(colourBackwardOmegaCfg)));
    const double colourDiagFloor = std::max(1e-30, static_cast<double>(colourDiagFloorCfg));
    const int colourBlockSize = static_cast<int>(colourBlockSizeCfg);

    cudaPCGDetail::Colouring colouring;
    List<int> colourPtrHost;
    List<int> colourIndicesHost;
    List<int> colourOfRowHost;
    label nColourSets = 0;
    bool colourReady = false;
    bool colourOperational = false;
    CUfunction colourForwardKernel = nullptr;
    CUfunction colourBackwardKernel = nullptr;

    int *d_preRowPtr=nullptr, *d_preColInd=nullptr;
    DeviceScalar *d_preVals=nullptr, *d_preTmp=nullptr, *d_preTmp2=nullptr;
    cusparseSpMatDescr_t matDL = nullptr; // (D+L)
    cusparseSpSVDescr_t spsvLower = nullptr, spsvUpperT = nullptr;
    cusparseDnVecDescr_t preIn = nullptr, preOut = nullptr;
    void* spsvBuf = nullptr;
    size_t spsvBufSize = 0;
    cusparseHandle_t cusparseHandle = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    void* spmvBuffer = nullptr;

    if (useColour)
    {
        colouring = cudaPCGDetail::buildGreedyColouring(matrix_);
        nColourSets = colouring.nColours;

        const bool validColouring =
        (
            nColourSets > 0
         && colouring.colourPtr.size() == static_cast<label>(nColourSets + 1)
         && colouring.colourIndices.size() == csrMatrix.nRows
         && colouring.cellToColour.size() == csrMatrix.nRows
        );

        if (!validColouring)
        {
            if (colourVerbose || reportStats)
            {
                Info
                    << "cudaPCG: disabling coloured preconditioner (colouring invalid)"
                    << nl;
            }
            colourFallbackTriggered = true;
            ++colourDisableCount;
            colourDisableEvents.push_back("invalidColouring");
            useColour = false;
        }
        else
        {
            colourPtrHost.setSize(nColourSets + 1);
            colourIndicesHost.setSize(rows);
            colourOfRowHost.setSize(rows);

            for (label i = 0; i <= nColourSets; ++i)
            {
                colourPtrHost[i] = static_cast<int>(colouring.colourPtr[i]);
            }
            for (int i = 0; i < rows; ++i)
            {
                colourIndicesHost[i] = static_cast<int>(colouring.colourIndices[i]);
                colourOfRowHost[i] = static_cast<int>(colouring.cellToColour[i]);
            }

            colourReady = true;
            colourOperational = true;
            colourBuiltSuccessfully = true;
            colourSetCount = nColourSets;

            if (nColourSets > 0)
            {
                label minColourSizeLocal = std::numeric_limits<label>::max();
                label maxColourSizeLocal = 0;
                scalar sumColourSize = 0;
                for (label colourI = 0; colourI < nColourSets; ++colourI)
                {
                    const label size =
                        colourPtrHost[colourI + 1] - colourPtrHost[colourI];
                    minColourSizeLocal = min(minColourSizeLocal, size);
                    maxColourSizeLocal = max(maxColourSizeLocal, size);
                    sumColourSize += size;
                }
                colourMinSize = minColourSizeLocal;
                colourMaxSize = maxColourSizeLocal;
                colourAvgSize = sumColourSize/static_cast<scalar>(nColourSets);
            }
            else
            {
                colourMinSize = 0;
                colourMaxSize = 0;
                colourAvgSize = 0;
            }

            if (colourVerbose || reportStats)
            {
                Info<< "cudaPCG: coloured preconditioner using " << nColourSets << " colour sets" << nl;
            }
        }
    }

    auto disableColour = [&](const char* stage)
    {
        if (colourVerbose || reportStats)
        {
            Info<< "cudaPCG: coloured preconditioner disabled (" << stage << ')' << nl;
        }
        colourFallbackTriggered = true;
        ++colourDisableCount;
        colourDisableEvents.push_back(stage ? std::string(stage) : std::string("unspecified"));
        if (d_colors)
        {
            cudaFree(d_colors);
            d_colors = nullptr;
        }
        if (d_colorIndices)
        {
            cudaFree(d_colorIndices);
            d_colorIndices = nullptr;
        }
        if (d_failFlag)
        {
            cudaFree(d_failFlag);
            d_failFlag = nullptr;
        }
        colourReady = false;
        colourOperational = false;
        useColour = false;
    };

    auto writeTelemetry =
        [&](const bool success, const word& failureMessage)
    {
        if (telemetryWritten)
        {
            return;
        }
        telemetryWritten = true;

        const bool master = Pstream::master();
        const Time& runTime = matrix_.mesh().thisDb().time();

        auto resolvePath = [&](const fileName& fName) -> fileName
        {
            if (fName == fileName::null)
            {
                return fName;
            }
            if (fName.isAbsolute())
            {
                return fName;
            }
            return runTime.globalPath()/fName;
        };

        if (logResidualTrajectory && !residualValueHistory.empty() && master)
        {
            const fileName resolved = resolvePath(residualLogFile);
            const fileName residualParent = resolved.path();
            if (!residualParent.empty())
            {
                mkDir(residualParent);
            }
            OFstream os(resolved);
            os.setf(std::ios::scientific);
            os.precision(IOstream::defaultPrecision());
            os<< "# cudaPCG residual trajectory for " << fieldName_ << nl;
            os<< "# success," << (success ? "true" : "false") << nl;
            if (failureMessage.size())
            {
                os<< "# failure," << failureMessage << nl;
            }
            os<< "iteration,residual" << nl;
            for (std::size_t i = 0; i < residualValueHistory.size(); ++i)
            {
                os<< residualIterHistory[i] << ',' << residualValueHistory[i] << nl;
            }
            Info<< "cudaPCG(" << fieldName_ << "): residual log -> " << resolved << nl;
        }

        const bool colourActiveFinal = useColour && colourOperational;
        const bool emitColourSummary =
            colourInitiallyRequested && (reportStats || colourVerbose || logColourStats);
        const bool masterHasColourData =
            colourInitiallyRequested && master;

        std::ostringstream disableSummary;
        if (!colourDisableEvents.empty())
        {
            for (std::size_t i = 0; i < colourDisableEvents.size(); ++i)
            {
                if (i)
                {
                    disableSummary << '|';
                }
                disableSummary << colourDisableEvents[i];
            }
        }

        if (emitColourSummary && master)
        {
            Info<< "cudaPCG(" << fieldName_ << "): colour sets=" << colourSetCount
                << " size[min,max,avg]=[" << colourMinSize << ',' << colourMaxSize
                << ',' << colourAvgSize << ']'
                << " omegaF=" << colourOmegaEff
                << " omegaB=" << colourBackwardOmegaEff
                << " diagFloor=" << colourDiagFloor
                << " built=" << (colourBuiltSuccessfully ? "true" : "false")
                << " applied=" << (colourAppliedSuccessfully ? "true" : "false")
                << " activeFinal=" << (colourActiveFinal ? "true" : "false")
                << " disables=" << colourDisableCount;
            if (colourFallbackTriggered)
            {
                Info<< " (fallback triggered)";
            }
            Info<< nl;
            if (!colourDisableEvents.empty())
            {
                Info<< "cudaPCG(" << fieldName_ << "): colour disable history "
                    << disableSummary.str() << nl;
            }
        }

        if ((reportStats || logColourStats) && masterHasColourData)
        {
            const fileName resolved = resolvePath(colourStatsFile);
            const fileName colourParent = resolved.path();
            if (!colourParent.empty())
            {
                mkDir(colourParent);
            }
            OFstream os(resolved);
            os.setf(std::ios::scientific);
            os.precision(IOstream::defaultPrecision());
            os<< "# cudaPCG colour statistics for " << fieldName_ << nl;
            os<< "# success," << (success ? "true" : "false") << nl;
            if (failureMessage.size())
            {
                os<< "# failure," << failureMessage << nl;
            }
            os<< "nColours,minSize,maxSize,avgSize,diagFloor,omegaForward,omegaBackward,diagMin,diagMax,built,applied,activeFinal,disableCount,disableStages" << nl;
            os<< colourSetCount << ','
               << colourMinSize << ','
               << colourMaxSize << ','
               << colourAvgSize << ','
               << colourDiagFloor << ','
               << colourOmegaEff << ','
               << colourBackwardOmegaEff << ','
               << minDiag << ','
               << maxDiag << ','
               << (colourBuiltSuccessfully ? 1 : 0) << ','
               << (colourAppliedSuccessfully ? 1 : 0) << ','
               << (colourActiveFinal ? 1 : 0) << ','
               << colourDisableCount << ',';
            if (!colourDisableEvents.empty())
            {
                os<< disableSummary.str();
            }
            os<< nl;
            Info<< "cudaPCG(" << fieldName_ << "): colour stats -> " << resolved << nl;
        }
    };

    auto cleanup = [&]()
    {
        if (vecX)
        {
            cusparseDestroyDnVec(vecX);
            vecX = nullptr;
        }
        if (vecY)
        {
            cusparseDestroyDnVec(vecY);
            vecY = nullptr;
        }
        if (matA)
        {
            cusparseDestroySpMat(matA);
            matA = nullptr;
        }
        if (spmvBuffer)
        {
            cudaFree(spmvBuffer);
            spmvBuffer = nullptr;
        }
        if (cublasHandle)
        {
            cublasDestroy(cublasHandle);
            cublasHandle = nullptr;
        }
        if (cusparseHandle)
        {
            cusparseDestroy(cusparseHandle);
            cusparseHandle = nullptr;
        }
        if (d_vals) cudaFree(d_vals);
        if (d_x) cudaFree(d_x);
        if (d_b) cudaFree(d_b);
        if (d_r) cudaFree(d_r);
        if (d_p) cudaFree(d_p);
        if (d_Ap) cudaFree(d_Ap);
        if (d_z) cudaFree(d_z);
        if (d_diagInv) cudaFree(d_diagInv);
        if (d_rowPtr) cudaFree(d_rowPtr);
        if (d_colInd) cudaFree(d_colInd);
        if (d_colors) cudaFree(d_colors);
        if (d_colorIndices) cudaFree(d_colorIndices);
        if (d_failFlag) cudaFree(d_failFlag);
        if (d_preRowPtr) cudaFree(d_preRowPtr);
        if (d_preColInd) cudaFree(d_preColInd);
        if (d_preVals) cudaFree(d_preVals);
        if (d_preTmp) cudaFree(d_preTmp);
        if (preIn) cusparseDestroyDnVec(preIn);
        if (preOut) cusparseDestroyDnVec(preOut);
        if (matDL) cusparseDestroySpMat(matDL);
        if (spsvLower) cusparseSpSV_destroyDescr(spsvLower);
        if (spsvUpperT) cusparseSpSV_destroyDescr(spsvUpperT);
        if (spsvBuf) cudaFree(spsvBuf);
        if (d_preTmp2) cudaFree(d_preTmp2);
        // ILU/IC resources
        if (matL) cusparseDestroySpMat(matL);
        if (matU) cusparseDestroySpMat(matU);
        if (matLchol) cusparseDestroySpMat(matLchol);
        if (spsvLowerIlu) cusparseSpSV_destroyDescr(spsvLowerIlu);
        if (spsvUpperIlu) cusparseSpSV_destroyDescr(spsvUpperIlu);
        if (spsvLowerChol) cusparseSpSV_destroyDescr(spsvLowerChol);
        if (spsvUpperTChol) cusparseSpSV_destroyDescr(spsvUpperTChol);
        if (iluBuf) cudaFree(iluBuf);
        if (iluInfo) cusparseDestroyCsrilu02Info(iluInfo);
        if (icInfo) cusparseDestroyCsric02Info(icInfo);
        if (iluDescr) cusparseDestroyMatDescr(iluDescr);
        if (d_iluVals) cudaFree(d_iluVals);
        if (spmvGraphExec)
        {
            cudaGraphExecDestroy(spmvGraphExec);
            spmvGraphExec = nullptr;
        }
        if (spmvGraph)
        {
            cudaGraphDestroy(spmvGraph);
            spmvGraph = nullptr;
        }
        if (computeStream)
        {
            cudaStreamDestroy(computeStream);
            computeStream = nullptr;
        }
    };

    auto fail = [&](const word& msg)
    {
        cleanup();
        message = msg;
        writeTelemetry(false, msg);
        return false;
    };

    auto syncStream = [&]() -> bool
    {
        if (!computeStream)
        {
            return true;
        }
        if (cudaStreamSynchronize(computeStream) != cudaSuccess)
        {
            message = word("cudaStreamSyncFail");
            cleanup();
            return false;
        }
        return true;
    };

    // Allocate
    if (cudaMalloc(reinterpret_cast<void**>(&d_vals), sizeof(DeviceScalar)*nnz) != cudaSuccess)
        return fail("cudaMalloc(vals)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_rowPtr), sizeof(int)*(rows+1)) != cudaSuccess)
        return fail("cudaMalloc(rowPtr)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_colInd), sizeof(int)*nnz) != cudaSuccess)
        return fail("cudaMalloc(colInd)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_x), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(x)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(b)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_r), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(r)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_p), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(p)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_Ap), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(Ap)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_z), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(z)");
    if (cudaMalloc(reinterpret_cast<void**>(&d_diagInv), sizeof(DeviceScalar)*rows) != cudaSuccess)
        return fail("cudaMalloc(diagInv)");

    // Optional symmetric diagonal equilibration A' = S*A*S, b' = S*b, x' = S^{-1} x
    List<DeviceScalar> scaleHost;
    List<DeviceScalar> invScaleHost;
    if (equilibrateMatrix)
    {
        scaleHost.setSize(rows);
        invScaleHost.setSize(rows);
        for (int i = 0; i < rows; ++i)
        {
            const double reg = static_cast<double>(precondDiag[i]);
            const double s = (reg > 0) ? 1.0/std::sqrt(reg) : 1.0;
            scaleHost[i] = s;
            invScaleHost[i] = (s != 0.0) ? (1.0/s) : 1.0;
        }
        // Scale CSR values on host: a'_{ij} = s_i * a_{ij} * s_j
        for (int i = 0; i < rows; ++i)
        {
            const double si = static_cast<double>(scaleHost[i]);
            for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k)
            {
                const int j = colInd[k];
                const double sj = static_cast<double>(scaleHost[j]);
                values[k] = static_cast<DeviceScalar>(static_cast<double>(values[k]) * si * sj);
            }
        }
        // Adjust diagInv for scaled diagonal (approximate: reg' = s_i^2 * reg)
        for (int i = 0; i < rows; ++i)
        {
            const double si = static_cast<double>(scaleHost[i]);
            const double reg = static_cast<double>(precondDiag[i]) * si * si;
            diagInv[i] = (reg > VSMALL) ? static_cast<DeviceScalar>(1.0/reg) : static_cast<DeviceScalar>(0);
        }
    }

    // Copy values
    if (cudaMemcpy(d_vals, values.begin(), sizeof(DeviceScalar)*nnz, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(vals)");
    if (cudaMemcpy(d_rowPtr, rowPtr.begin(), sizeof(int)*(rows+1), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(rowPtr)");
    if (cudaMemcpy(d_colInd, colInd.begin(), sizeof(int)*nnz, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(colInd)");
    if (equilibrateMatrix)
    {
        List<DeviceScalar> xScaled(rows), bScaled(rows);
        for (int i = 0; i < rows; ++i)
        {
            xScaled[i] = static_cast<DeviceScalar>(static_cast<double>(psi[i]) * static_cast<double>(invScaleHost[i]));
            bScaled[i] = static_cast<DeviceScalar>(static_cast<double>(source[i]) * static_cast<double>(scaleHost[i]));
        }
        if (cudaMemcpy(d_x, xScaled.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("cudaMemcpy(xScaled)");
        if (cudaMemcpy(d_b, bScaled.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("cudaMemcpy(bScaled)");
    }
    else
    {
        if (cudaMemcpy(d_x, psi.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("cudaMemcpy(x)");
        if (cudaMemcpy(d_b, source.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("cudaMemcpy(b)");
    }
    if (cudaMemcpy(d_diagInv, diagInv.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(diagInv)");

    if (colourReady)
    {
        if (cudaMalloc(reinterpret_cast<void**>(&d_colors), sizeof(int)*rows) != cudaSuccess)
        {
            disableColour("cudaMalloc colours");
        }
        else if (cudaMalloc(reinterpret_cast<void**>(&d_colorIndices), sizeof(int)*rows) != cudaSuccess)
        {
            d_colorIndices = nullptr;
            disableColour("cudaMalloc colourIndices");
        }
        else if (cudaMalloc(reinterpret_cast<void**>(&d_failFlag), sizeof(int)) != cudaSuccess)
        {
            d_failFlag = nullptr;
            disableColour("cudaMalloc colourFlag");
        }
        else if
        (
            cudaMemcpy(d_colors, colourOfRowHost.begin(), sizeof(int)*rows, cudaMemcpyHostToDevice) != cudaSuccess
         || cudaMemcpy(d_colorIndices, colourIndicesHost.begin(), sizeof(int)*rows, cudaMemcpyHostToDevice) != cudaSuccess
        )
        {
            disableColour("cudaMemcpy colourData");
        }
        else if (!getColourKernels(deviceId, colourForwardKernel, colourBackwardKernel, colourVerbose || reportStats))
        {
            disableColour("kernelCompile");
        }
    }

    if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS)
        return fail("cusparseCreate");
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
        return fail("cublasCreate");

    if (usePipelinedCG || useCudaGraph)
    {
        if (cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking) != cudaSuccess)
        {
            return fail("cudaStreamCreate");
        }
        if (cublasSetStream(cublasHandle, computeStream) != CUBLAS_STATUS_SUCCESS)
        {
            return fail("cublasSetStream");
        }
        if (cusparseSetStream(cusparseHandle, computeStream) != CUSPARSE_STATUS_SUCCESS)
        {
            return fail("cusparseSetStream");
        }
    }

    if
    (
        cusparseCreateCsr
        (
            &matA,
            rows,
            rows,
            nnz,
            d_rowPtr,
            d_colInd,
            d_vals,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ) != CUSPARSE_STATUS_SUCCESS
    )
    {
        return fail("cusparseCreateCsr");
    }

    if (cusparseCreateDnVec(&vecX, rows, d_x, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        return fail("cusparseCreateDnVec(x)");
    if (cusparseCreateDnVec(&vecY, rows, d_Ap, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        return fail("cusparseCreateDnVec(y)");

    // Build optional SGS preconditioner structures: CSR of (D+L)
    bool sgsReady = false;
    bool sgsBuiltSuccessfully = false;
    std::string sgsDisableReason;
    int preRows = 0, preNnz = 0;
    if (useSGS)
    {
        // Count nnz in (D+L)
        preRows = rows;
        List<int> preRowPtrHost(rows + 1, 0);
        for (int i = 0; i < rows; ++i)
        {
            int count = 0;
            for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k)
            {
                if (colInd[k] <= i) ++count;
            }
            preRowPtrHost[i+1] = preRowPtrHost[i] + count;
        }
        preNnz = preRowPtrHost[rows];
        List<int> preColIndHost(preNnz);
        List<DeviceScalar> preValsHost(preNnz);
        // Fill
        for (int i = 0; i < rows; ++i)
        {
            int cursor = preRowPtrHost[i];
            for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k)
            {
                const int j = colInd[k];
                if (j <= i)
                {
                    preColIndHost[cursor] = j;
                    DeviceScalar v = values[k];
                    if (j == i)
                    {
                        // Use regularised diagonal; if equilibrated, account for scaling (s_i^2)
                        if (equilibrateMatrix)
                        {
                            const double si = (scaleHost.size() ? static_cast<double>(scaleHost[i]) : 1.0);
                            v = static_cast<DeviceScalar>(static_cast<double>(precondDiag[i]) * si * si);
                        }
                        else
                        {
                            v = precondDiag[i];
                        }
                    }
                    preValsHost[cursor] = v;
                    ++cursor;
                }
            }
        }

        // Device allocate/copy
        if
        (
            cudaMalloc(reinterpret_cast<void**>(&d_preRowPtr), sizeof(int)*(preRows+1)) == cudaSuccess
         && cudaMalloc(reinterpret_cast<void**>(&d_preColInd), sizeof(int)*preNnz) == cudaSuccess
         && cudaMalloc(reinterpret_cast<void**>(&d_preVals), sizeof(DeviceScalar)*preNnz) == cudaSuccess
         && cudaMalloc(reinterpret_cast<void**>(&d_preTmp), sizeof(DeviceScalar)*rows) == cudaSuccess
         && cudaMalloc(reinterpret_cast<void**>(&d_preTmp2), sizeof(DeviceScalar)*rows) == cudaSuccess
         && cusparseCreateDnVec(&preIn, rows, d_preTmp, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
         && cusparseCreateDnVec(&preOut, rows, d_preTmp, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
        )
        {
            if
            (
                cudaMemcpy(d_preRowPtr, preRowPtrHost.begin(), sizeof(int)*(preRows+1), cudaMemcpyHostToDevice) == cudaSuccess
             && cudaMemcpy(d_preColInd, preColIndHost.begin(), sizeof(int)*preNnz, cudaMemcpyHostToDevice) == cudaSuccess
             && cudaMemcpy(d_preVals, preValsHost.begin(), sizeof(DeviceScalar)*preNnz, cudaMemcpyHostToDevice) == cudaSuccess
             && cusparseCreateCsr
                (
                    &matDL,
                    static_cast<int64_t>(preRows),
                    static_cast<int64_t>(preRows),
                    static_cast<int64_t>(preNnz),
                    d_preRowPtr,
                    d_preColInd,
                    d_preVals,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F
                ) == CUSPARSE_STATUS_SUCCESS
             && cusparseSpSV_createDescr(&spsvLower) == CUSPARSE_STATUS_SUCCESS
             && cusparseSpSV_createDescr(&spsvUpperT) == CUSPARSE_STATUS_SUCCESS
            )
            {
                cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
                cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
                cusparseSpMatSetAttribute(matDL, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
                cusparseSpMatSetAttribute(matDL, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));

                // Buffer size for SpSV (forward and transpose)
                const double oneD = 1.0;
                size_t fwd=0, bwd=0;
                if
                (
                    cusparseSpSV_bufferSize
                    (
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &oneD,
                        matDL,
                        preIn,
                        preOut,
                        CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsvLower,
                        &fwd
                    ) == CUSPARSE_STATUS_SUCCESS
                 && cusparseSpSV_bufferSize
                    (
                        cusparseHandle,
                        CUSPARSE_OPERATION_TRANSPOSE,
                        &oneD,
                        matDL,
                        preIn,
                        preOut,
                        CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsvUpperT,
                        &bwd
                    ) == CUSPARSE_STATUS_SUCCESS
                )
                {
                    spsvBufSize = std::max(fwd, bwd);
                    if (spsvBufSize == 0 || cudaMalloc(&spsvBuf, spsvBufSize) == cudaSuccess)
                    {
                        // Analyses
                        if
                        (
                            cusparseSpSV_analysis
                            (
                                cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &oneD,
                                matDL,
                                preIn,
                                preOut,
                                CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT,
                                spsvLower,
                                spsvBuf
                            ) == CUSPARSE_STATUS_SUCCESS
                         && cusparseSpSV_analysis
                            (
                                cusparseHandle,
                                CUSPARSE_OPERATION_TRANSPOSE,
                                &oneD,
                                matDL,
                                preIn,
                                preOut,
                                CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT,
                                spsvUpperT,
                                spsvBuf
                            ) == CUSPARSE_STATUS_SUCCESS
                        )
                        {
                            sgsReady = true;
                            sgsBuiltSuccessfully = true;
                        }
                    }
                }
            }
        }
        else
        {
            sgsDisableReason = std::string("allocOrDescrCreate");
        }
    }

    if ((usePolyJacobi || useColour || useSGS || useILU || useIC) && !d_preTmp)
    {
        if (cudaMalloc(reinterpret_cast<void**>(&d_preTmp), sizeof(DeviceScalar)*rows) != cudaSuccess)
        {
            return fail("cudaMalloc(preTmp)");
        }
    }
    if (usePolyJacobi && !d_preTmp2)
    {
        if (cudaMalloc(reinterpret_cast<void**>(&d_preTmp2), sizeof(DeviceScalar)*rows) != cudaSuccess)
        {
            return fail("cudaMalloc(preTmp2)");
        }
    }

    // Build optional ILU0/IC0 preconditioner using cuSPARSE (legacy csrilu02/csric02)
    bool iluReady = false;
    bool icReady = false;
    std::string iluDisableReason;
    if ((useILU || useIC) && !preIn)
    {
        if (cusparseCreateDnVec(&preIn, rows, d_preTmp, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        {
            // preIn required for SpSV buffer sizing/analysis
            useILU = false; useIC = false;
        }
        else if (cusparseCreateDnVec(&preOut, rows, d_preTmp, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        {
            useILU = false; useIC = false;
        }
    }
    if (useILU || useIC)
    {
        // Allocate separate values array for in-place factorization
        // Start from host values and apply diagonal regularisation specifically for ILU/IC
        List<DeviceScalar> iluValsHost(nnz);
        for (int i = 0; i < nnz; ++i) iluValsHost[i] = values[i];

        const scalar iluDiagAbsFloorCfg = dict.lookupOrDefault<scalar>("iluDiagAbsFloor", diagFloorCfg);
        const scalar iluDiagRelFloorCfg = dict.lookupOrDefault<scalar>("iluDiagRelFloor", scalar(0));
        const double iluDiagFloorEff = std::max(static_cast<double>(iluDiagAbsFloorCfg), static_cast<double>(iluDiagRelFloorCfg)*static_cast<double>(maxAbsDiag));

        if (useIC || useILU)
        {
            for (int i = 0; i < rows; ++i)
            {
                int diagK = -1;
                for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k)
                {
                    if (colInd[k] == i) { diagK = k; break; }
                }
                if (diagK >= 0)
                {
                    double dv = static_cast<double>(iluValsHost[diagK]);
                    if (useIC)
                    {
                        // IC0 expects positive diagonals
                        if (dv <= 0.0 || std::abs(dv) < iluDiagFloorEff) dv = iluDiagFloorEff;
                    }
                    else
                    {
                        // ILU0: avoid near-zero diagonals, preserve sign
                        if (std::abs(dv) < iluDiagFloorEff) dv = (dv >= 0.0 ? iluDiagFloorEff : -iluDiagFloorEff);
                    }
                    iluValsHost[diagK] = static_cast<DeviceScalar>(dv);
                }
            }
        }

        if (cudaMalloc(reinterpret_cast<void**>(&d_iluVals), sizeof(DeviceScalar)*nnz) != cudaSuccess)
        {
            iluDisableReason = std::string("allocVals");
        }
        else if (cudaMemcpy(d_iluVals, iluValsHost.begin(), sizeof(DeviceScalar)*nnz, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            iluDisableReason = std::string("copyVals");
        }
        else if (cusparseCreateMatDescr(&iluDescr) != CUSPARSE_STATUS_SUCCESS)
        {
            iluDisableReason = std::string("createDescr");
        }
        else
        {
            cusparseSetMatType(iluDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(iluDescr, CUSPARSE_INDEX_BASE_ZERO);

            if (useILU)
            {
                if (cusparseCreateCsrilu02Info(&iluInfo) == CUSPARSE_STATUS_SUCCESS)
                {
                    // Optional numeric boost to avoid zero pivots
                    const Switch iluNumericBoost = dict.lookupOrDefault<Switch>("iluNumericBoost", Switch(false));
                    const scalar iluBoostTolCfg = dict.lookupOrDefault<scalar>("iluNumericBoostTol", diagFloorCfg);
                    const scalar iluBoostValCfg = dict.lookupOrDefault<scalar>("iluNumericBoostVal", diagFloorCfg);
                    double boostTol = static_cast<double>(iluBoostTolCfg);
                    double boostVal = static_cast<double>(iluBoostValCfg);
                    int bs = 0;
                    if
                    (
                        cusparseDcsrilu02_bufferSize
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            d_rowPtr, d_colInd,
                            iluInfo, &bs
                        ) == CUSPARSE_STATUS_SUCCESS
                     && (bs == 0 || cudaMalloc(&iluBuf, bs) == cudaSuccess)
                     && cusparseDcsrilu02_analysis
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            d_rowPtr, d_colInd,
                            iluInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                            iluBuf
                        ) == CUSPARSE_STATUS_SUCCESS
                     && (
                        (!iluNumericBoost)
                        || (cusparseDcsrilu02_numericBoost
                            (
                                cusparseHandle,
                                iluInfo,
                                1,
                                &boostTol,
                                &boostVal
                            ) == CUSPARSE_STATUS_SUCCESS)
                       )
                     && cusparseDcsrilu02
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            d_rowPtr, d_colInd,
                            iluInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                            iluBuf
                        ) == CUSPARSE_STATUS_SUCCESS
                    )
                    {
                        // Zero pivot check
                        int pivot = -1;
                        cusparseStatus_t zp = cusparseXcsrilu02_zeroPivot(cusparseHandle, iluInfo, &pivot);
                        if (zp == CUSPARSE_STATUS_ZERO_PIVOT)
                        {
                            iluDisableReason = std::string("zeroPivot");
                        }
                        else
                        {
                        if
                        (
                            cusparseCreateCsr(&matL, rows, rows, nnz, d_rowPtr, d_colInd, d_iluVals,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                         && cusparseCreateCsr(&matU, rows, rows, nnz, d_rowPtr, d_colInd, d_iluVals,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                         && cusparseSpSV_createDescr(&spsvLowerIlu) == CUSPARSE_STATUS_SUCCESS
                         && cusparseSpSV_createDescr(&spsvUpperIlu) == CUSPARSE_STATUS_SUCCESS
                        )
                        {
                            cusparseFillMode_t fillL = CUSPARSE_FILL_MODE_LOWER;
                            cusparseDiagType_t diagL = CUSPARSE_DIAG_TYPE_UNIT;
                            cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fillL, sizeof(fillL));
                            cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diagL, sizeof(diagL));
                            cusparseFillMode_t fillU = CUSPARSE_FILL_MODE_UPPER;
                            cusparseDiagType_t diagU = CUSPARSE_DIAG_TYPE_NON_UNIT;
                            cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &fillU, sizeof(fillU));
                            cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE, &diagU, sizeof(diagU));

                            const double oneD = 1.0; size_t bL=0, bU=0;
                            if
                            (
                                cusparseSpSV_bufferSize
                                (
                                    cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &oneD,
                                    matL,
                                    preIn,
                                    preOut,
                                    CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    spsvLowerIlu,
                                    &bL
                                ) == CUSPARSE_STATUS_SUCCESS
                             && cusparseSpSV_bufferSize
                                (
                                    cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &oneD,
                                    matU,
                                    preIn,
                                    preOut,
                                    CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    spsvUpperIlu,
                                    &bU
                                ) == CUSPARSE_STATUS_SUCCESS
                            )
                            {
                                iluBufSize = std::max(iluBufSize, std::max(bL, bU));
                                if ((iluBufSize == 0) || (cudaMalloc(&iluBuf, iluBufSize) == cudaSuccess))
                                {
                                    if
                                    (
                                        cusparseSpSV_analysis
                                        (
                                            cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &oneD,
                                            matL,
                                            preIn,
                                            preOut,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvLowerIlu,
                                            iluBuf
                                        ) == CUSPARSE_STATUS_SUCCESS
                                     && cusparseSpSV_analysis
                                        (
                                            cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &oneD,
                                            matU,
                                            preIn,
                                            preOut,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvUpperIlu,
                                            iluBuf
                                        ) == CUSPARSE_STATUS_SUCCESS
                                    )
                                    {
                                        iluReady = true;
                                    }
                                }
                            }
                        }
                        }
                    }
                    else
                    {
                        iluDisableReason = std::string("csrilu02");
                    }
                }
                else
                {
                    iluDisableReason = std::string("createInfo");
                }
            }
            else if (useIC)
            {
                if (cusparseCreateCsric02Info(&icInfo) == CUSPARSE_STATUS_SUCCESS)
                {
                    int bs = 0;
                    if
                    (
                        cusparseDcsric02_bufferSize
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            d_rowPtr, d_colInd,
                            icInfo, &bs
                        ) == CUSPARSE_STATUS_SUCCESS
                     && (bs == 0 || cudaMalloc(&iluBuf, bs) == cudaSuccess)
                     && cusparseDcsric02_analysis
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            d_rowPtr, d_colInd,
                            icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                            iluBuf
                        ) == CUSPARSE_STATUS_SUCCESS
                     && cusparseDcsric02
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            d_rowPtr, d_colInd,
                            icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                            iluBuf
                        ) == CUSPARSE_STATUS_SUCCESS
                    )
                    {
                        // Zero pivot check (IC0)
                        int pivot = -1;
                        cusparseStatus_t zp = cusparseXcsric02_zeroPivot(cusparseHandle, icInfo, &pivot);
                        if (zp == CUSPARSE_STATUS_ZERO_PIVOT)
                        {
                            iluDisableReason = std::string("zeroPivot");
                        }
                        else
                        {
                        if
                        (
                            cusparseCreateCsr(&matLchol, rows, rows, nnz, d_rowPtr, d_colInd, d_iluVals,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                         && cusparseSpSV_createDescr(&spsvLowerChol) == CUSPARSE_STATUS_SUCCESS
                         && cusparseSpSV_createDescr(&spsvUpperTChol) == CUSPARSE_STATUS_SUCCESS
                        )
                        {
                            cusparseFillMode_t fillL = CUSPARSE_FILL_MODE_LOWER;
                            cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
                            cusparseSpMatSetAttribute(matLchol, CUSPARSE_SPMAT_FILL_MODE, &fillL, sizeof(fillL));
                            cusparseSpMatSetAttribute(matLchol, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

                            const double oneD = 1.0; size_t bL=0, bUT=0;
                            if
                            (
                                cusparseSpSV_bufferSize
                                (
                                    cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &oneD,
                                    matLchol,
                                    preIn,
                                    preOut,
                                    CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    spsvLowerChol,
                                    &bL
                                ) == CUSPARSE_STATUS_SUCCESS
                             && cusparseSpSV_bufferSize
                                (
                                    cusparseHandle,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    &oneD,
                                    matLchol,
                                    preIn,
                                    preOut,
                                    CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    spsvUpperTChol,
                                    &bUT
                                ) == CUSPARSE_STATUS_SUCCESS
                            )
                            {
                                iluBufSize = std::max(iluBufSize, std::max(bL, bUT));
                                if ((iluBufSize == 0) || (cudaMalloc(&iluBuf, iluBufSize) == cudaSuccess))
                                {
                                    if
                                    (
                                        cusparseSpSV_analysis
                                        (
                                            cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &oneD,
                                            matLchol,
                                            preIn,
                                            preOut,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvLowerChol,
                                            iluBuf
                                        ) == CUSPARSE_STATUS_SUCCESS
                                     && cusparseSpSV_analysis
                                        (
                                            cusparseHandle,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &oneD,
                                            matLchol,
                                            preIn,
                                            preOut,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvUpperTChol,
                                            iluBuf
                                        ) == CUSPARSE_STATUS_SUCCESS
                                    )
                                    {
                                        icReady = true;
                                    }
                                }
                            }
                        }
                        }
                    }
                    else
                    {
                        iluDisableReason = std::string("csric02");
                    }
                }
                else
                {
                    iluDisableReason = std::string("createInfo");
                }
            }
        }
    }

    auto applyDiagonalDevice =
        [&](const double* rhsVec, double* outVec) -> bool
    {
        if (cublasDcopy(cublasHandle, rows, rhsVec, 1, outVec, 1) != CUBLAS_STATUS_SUCCESS)
        {
            message = word("DiagCopyFail");
            return false;
        }
        if
        (
            cublasDdgmm
            (
                cublasHandle,
                CUBLAS_SIDE_LEFT,
                rows,
                1,
                outVec,
                rows,
                reinterpret_cast<const double*>(d_diagInv),
                1,
                outVec,
                rows
            ) != CUBLAS_STATUS_SUCCESS
        )
        {
            message = word("DiagScaleFail");
            return false;
        }
        return true;
    };

    auto applyColourDevice =
        [&](const double* rhsVec, double* outVec, const char* stage) -> bool
    {
        (void)stage;
        if (!colourOperational || !colourReady)
        {
            return false;
        }
        if (!d_preTmp || !d_colors || !d_colorIndices || !d_failFlag || !colourForwardKernel || !colourBackwardKernel)
        {
            disableColour("missingResources");
            return false;
        }

        if (cudaMemset(d_preTmp, 0, sizeof(DeviceScalar)*rows) != cudaSuccess)
        {
            disableColour("zeroForward");
            return false;
        }
        if (cudaMemset(outVec, 0, sizeof(DeviceScalar)*rows) != cudaSuccess)
        {
            disableColour("zeroResult");
            return false;
        }

        int zeroFlag = 0;
        if (cudaMemcpy(d_failFlag, &zeroFlag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            disableColour("resetFlag");
            return false;
        }

        const double diagFloorLocal = colourDiagFloor;
        const double omegaForward = colourOmegaEff;
        const double omegaBackward = colourBackwardOmegaEff;

        for (label colour = 0; colour < nColourSets; ++colour)
        {
            const label start = colourPtrHost[colour];
            const int count = static_cast<int>(colourPtrHost[colour+1] - start);
            if (!count)
            {
                continue;
            }

            const int* rowIdxDev = d_colorIndices + start;
            const int* rowPtrDev = d_rowPtr;
            const int* colIndDev = d_colInd;
            const double* valDev = reinterpret_cast<const double*>(d_vals);
            const int* coloursDev = d_colors;
            const double* rhsDev = rhsVec;
            double* solDev = reinterpret_cast<double*>(d_preTmp);
            int colourId = static_cast<int>(colour);
            double omegaLocal = omegaForward;
            double diagFloorPass = diagFloorLocal;
            int* failDev = d_failFlag;

            int countLocal = count;
            void* args[] =
            {
                reinterpret_cast<void*>(&countLocal),
                reinterpret_cast<void*>(&rowIdxDev),
                reinterpret_cast<void*>(&rowPtrDev),
                reinterpret_cast<void*>(&colIndDev),
                reinterpret_cast<void*>(&valDev),
                reinterpret_cast<void*>(&coloursDev),
                reinterpret_cast<void*>(&rhsDev),
                reinterpret_cast<void*>(&solDev),
                reinterpret_cast<void*>(&colourId),
                reinterpret_cast<void*>(&omegaLocal),
                reinterpret_cast<void*>(&diagFloorPass),
                reinterpret_cast<void*>(&failDev)
            };

            const int blocks = (count + colourBlockSize - 1)/colourBlockSize;
            if (blocks <= 0)
            {
                continue;
            }

            CUresult launch =
                cuLaunchKernel
                (
                    colourForwardKernel,
                    blocks, 1, 1,
                    colourBlockSize, 1, 1,
                    0,
                    nullptr,
                    args,
                    nullptr
                );

            if (launch != CUDA_SUCCESS)
            {
                disableColour("launchForward");
                return false;
            }

            cudaError_t kernelErr = cudaGetLastError();
            if (kernelErr != cudaSuccess)
            {
                disableColour("forwardKernel");
                return false;
            }
        }

        int failHost = 0;
        if (cudaMemcpy(&failHost, d_failFlag, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            disableColour("forwardFlag");
            return false;
        }
        if (failHost != 0)
        {
            if (failHost == 1)
            {
                disableColour("forwardDiagFloor");
            }
            else if (failHost == 2)
            {
                disableColour("forwardNonFinite");
            }
            else
            {
                disableColour("forwardInstability");
            }
            return false;
        }

        if (cudaMemcpy(d_failFlag, &zeroFlag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            disableColour("resetFlagBack");
            return false;
        }

        if
        (
            cudaMemcpy
            (
                outVec,
                d_preTmp,
                sizeof(DeviceScalar)*rows,
                cudaMemcpyDeviceToDevice
            ) != cudaSuccess
        )
        {
            disableColour("prepBackward");
            return false;
        }

        for (label colour = nColourSets; colour-- > 0; )
        {
            const label start = colourPtrHost[colour];
            const int count = static_cast<int>(colourPtrHost[colour+1] - start);
            if (!count)
            {
                continue;
            }

            const int* rowIdxDev = d_colorIndices + start;
            const int* rowPtrDev = d_rowPtr;
            const int* colIndDev = d_colInd;
            const double* valDev = reinterpret_cast<const double*>(d_vals);
            const int* coloursDev = d_colors;
            const double* rhsDev = reinterpret_cast<const double*>(d_preTmp);
            double* solDev = outVec;
            int colourId = static_cast<int>(colour);
            double omegaLocal = omegaBackward;
            double diagFloorPass = diagFloorLocal;
            int* failDev = d_failFlag;

            int countLocal = count;
            void* args[] =
            {
                reinterpret_cast<void*>(&countLocal),
                reinterpret_cast<void*>(&rowIdxDev),
                reinterpret_cast<void*>(&rowPtrDev),
                reinterpret_cast<void*>(&colIndDev),
                reinterpret_cast<void*>(&valDev),
                reinterpret_cast<void*>(&coloursDev),
                reinterpret_cast<void*>(&rhsDev),
                reinterpret_cast<void*>(&solDev),
                reinterpret_cast<void*>(&colourId),
                reinterpret_cast<void*>(&omegaLocal),
                reinterpret_cast<void*>(&diagFloorPass),
                reinterpret_cast<void*>(&failDev)
            };

            const int blocks = (count + colourBlockSize - 1)/colourBlockSize;
            if (blocks <= 0)
            {
                continue;
            }

            CUresult launch =
                cuLaunchKernel
                (
                    colourBackwardKernel,
                    blocks, 1, 1,
                    colourBlockSize, 1, 1,
                    0,
                    nullptr,
                    args,
                    nullptr
                );

            if (launch != CUDA_SUCCESS)
            {
                disableColour("launchBackward");
                return false;
            }

            cudaError_t kernelErr = cudaGetLastError();
            if (kernelErr != cudaSuccess)
            {
                disableColour("backwardKernel");
                return false;
            }
        }

        if (cudaMemcpy(&failHost, d_failFlag, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            disableColour("backwardFlag");
            return false;
        }
        if (failHost != 0)
        {
            if (failHost == 1)
            {
                disableColour("backwardDiagFloor");
            }
            else if (failHost == 2)
            {
                disableColour("backwardNonFinite");
            }
            else
            {
                disableColour("backwardInstability");
            }
            return false;
        }

        colourAppliedSuccessfully = true;
        return true;
    };

    auto applyPreconditionerVec =
        [&](const double* rhs, double* out, const char* stage) -> bool
    {
        (void)stage;
        if (usePolyJacobi)
        {
            // out = omega*D^{-1} rhs
            if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("PolyCopyFail");
                return false;
            }
            if
            (
                cublasDdgmm
                (
                    cublasHandle,
                    CUBLAS_SIDE_LEFT,
                    rows,
                    1,
                    out,
                    rows,
                    reinterpret_cast<const double*>(d_diagInv),
                    1,
                    out,
                    rows
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                message = word("PolyDiagScaleFail");
                return false;
            }
            const double omega0 = jacOmegaEff;
            if (cublasDscal(cublasHandle, rows, &omega0, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("PolyScaleOmegaFail");
                return false;
            }

            const double oneD = 1.0;
            const double zeroD = 0.0;
            for (label s = 1; s < polySweepsEff; ++s)
            {
                // t = A*out
                cusparseDnVecDescr_t vpx = nullptr, vpy = nullptr;
                if (cusparseCreateDnVec(&vpx, rows, out, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
                { message = word("PolyCreateVecXFail"); return false; }
                if (cusparseCreateDnVec(&vpy, rows, d_preTmp2, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
                { cusparseDestroyDnVec(vpx); message = word("PolyCreateVecYFail"); return false; }
                cusparseStatus_t spmvStat =
                    cusparseSpMV
                    (
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &oneD,
                        matA,
                        vpx,
                        &zeroD,
                        vpy,
                        CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        spmvBuffer
                    );
                cusparseDestroyDnVec(vpx);
                cusparseDestroyDnVec(vpy);
                if (spmvStat != CUSPARSE_STATUS_SUCCESS)
                { message = word("PolySpMVFail"); return false; }
                if (computeStream && cudaStreamSynchronize(computeStream) != cudaSuccess)
                { message = word("PolyStreamSyncFail"); return false; }

                // d_preTmp = rhs - t
                if (cublasDcopy(cublasHandle, rows, rhs, 1, d_preTmp, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("PolyTmpCopyFail");
                    return false;
                }
                const double minusOne = -1.0;
                if (cublasDaxpy(cublasHandle, rows, &minusOne, d_preTmp2, 1, d_preTmp, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("PolyAxpyFail");
                    return false;
                }

                // d_preTmp = omega*D^{-1} d_preTmp
                if
                (
                    cublasDdgmm
                    (
                        cublasHandle,
                        CUBLAS_SIDE_LEFT,
                        rows,
                        1,
                        d_preTmp,
                        rows,
                        reinterpret_cast<const double*>(d_diagInv),
                        1,
                        d_preTmp,
                        rows
                    ) != CUBLAS_STATUS_SUCCESS
                )
                {
                    message = word("PolyDiagScale2Fail");
                    return false;
                }
                if (cublasDscal(cublasHandle, rows, &omega0, d_preTmp, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("PolyScale2OmegaFail");
                    return false;
                }

                // out += d_preTmp
                if (cublasDaxpy(cublasHandle, rows, &oneD, d_preTmp, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("PolyUpdateFail");
                    return false;
                }
            }
            // No descriptor state to restore when using transient DnVecs
            return true;
        }
        else if (useComposite && icReady && (colourOperational && colourReady))
        {
            // Composite: a few coloured sweeps, then IC0
            const label sweeps = std::max<label>(1, dict.lookupOrDefault<label>("compositeColourSweeps", 2));
            // First sweep: rhs -> out
            if (!applyColourDevice(rhs, out, "compColour1"))
            {
                // If colour failed, fall back to IC only
            }
            else
            {
                // Additional sweeps in-place on 'out'
                for (label s = 1; s < sweeps; ++s)
                {
                    if (!applyColourDevice(out, out, "compColourN"))
                    {
                        break;
                    }
                }
            }
            // Apply IC on the (possibly smoothed) vector in 'out'
            const double oneD = 1.0;
            cusparseDnVecSetValues(preIn, out);
            cusparseDnVecSetValues(preOut, d_preTmp);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &oneD,
                    matLchol,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvLowerChol
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("ICLowerFail");
                return false;
            }
            cusparseDnVecSetValues(preIn, d_preTmp);
            cusparseDnVecSetValues(preOut, out);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &oneD,
                    matLchol,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvUpperTChol
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("ICUpperTFail");
                return false;
            }
            return true;
        }
        else if (icReady)
        {
            // Incomplete Cholesky: solve L y = rhs; solve L^T out = y
            const double oneD = 1.0;
            if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("ICCopyFail");
                return false;
            }
            cusparseDnVecSetValues(preIn, out);
            cusparseDnVecSetValues(preOut, d_preTmp);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &oneD,
                    matLchol,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvLowerChol
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("ICLowerFail");
                return false;
            }
            cusparseDnVecSetValues(preIn, d_preTmp);
            cusparseDnVecSetValues(preOut, out);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &oneD,
                    matLchol,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvUpperTChol
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("ICUpperTFail");
                return false;
            }
            return true;
        }
        else if (iluReady)
        {
            // ILU0: solve L y = rhs; solve U out = y
            const double oneD = 1.0;
            if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("ILUCopyFail");
                return false;
            }
            cusparseDnVecSetValues(preIn, out);
            cusparseDnVecSetValues(preOut, d_preTmp);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &oneD,
                    matL,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvLowerIlu
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("ILULowerFail");
                return false;
            }
            cusparseDnVecSetValues(preIn, d_preTmp);
            cusparseDnVecSetValues(preOut, out);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &oneD,
                    matU,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvUpperIlu
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("ILUUpperFail");
                return false;
            }
            return true;
        }
        else if (sgsReady)
        {
            // out = M^{-1} rhs via SGS: solve (D+L) y = rhs; solve (D+L)^T out = y
            const double oneD = 1.0;
            // First, copy rhs into out then into preTmp
            if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("PrecondCopyFail");
                return false;
            }
            cusparseDnVecSetValues(preIn, out);
            cusparseDnVecSetValues(preOut, d_preTmp);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &oneD,
                    matDL,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvLower
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("SpSV_lower_fail");
                return false;
            }
            // Middle D^{-1} scaling for SSOR: y := D^{-1} y
            if
            (
                cublasDdgmm
                (
                    cublasHandle,
                    CUBLAS_SIDE_LEFT,
                    rows,
                    1,
                    reinterpret_cast<double*>(d_preTmp),
                    rows,
                    reinterpret_cast<const double*>(d_diagInv),
                    1,
                    reinterpret_cast<double*>(d_preTmp),
                    rows
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                message = word("SSORDiagScaleFail");
                return false;
            }

            cusparseDnVecSetValues(preIn, d_preTmp);
            cusparseDnVecSetValues(preOut, out);
            if
            (
                cusparseSpSV_solve
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &oneD,
                    matDL,
                    preIn,
                    preOut,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvUpperT
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                message = word("SpSV_upperT_fail");
                return false;
            }
            return true;
        }
        else if (colourOperational && colourReady)
        {
            if (applyColourDevice(rhs, out, stage))
            {
                return true;
            }
            // applyColourDevice disables the coloured path on failure – fall through to diagonal
        }

        return applyDiagonalDevice(rhs, out);
    };

    // Emit one-line SGS status when requested
    if (reportStats && useSGS)
    {
        if (sgsReady)
        {
            Info<< "cudaPCG(" << fieldName_ << "): SGS preconditioner active (rows="
                << rows << ", nnz=" << preNnz << ")" << nl;
        }
        else
        {
            Info<< "cudaPCG(" << fieldName_ << "): SGS preconditioner disabled ("
                << (sgsDisableReason.size() ? sgsDisableReason : std::string("buildFailure"))
                << ")" << nl;
        }
    }
    if (reportStats && (useILU || useIC))
    {
        if (useILU)
        {
            Info<< "cudaPCG(" << fieldName_ << "): ILU0 preconditioner "
                << (iluReady ? "active" : "disabled")
                << (iluReady ? "" : (std::string(" (") + (iluDisableReason.size()?iluDisableReason:"buildFailure") + ")"))
                << nl;
        }
        if (useIC)
        {
            Info<< "cudaPCG(" << fieldName_ << "): IC0 preconditioner "
                << (icReady ? "active" : "disabled")
                << (icReady ? "" : (std::string(" (") + (iluDisableReason.size()?iluDisableReason:"buildFailure") + ")"))
                << nl;
        }
    }

    const double alpha = 1.0;
    const double zero = 0.0;
    const double negOne = -1.0;

    size_t bufferSize = 0;
    if
    (
        cusparseSpMV_bufferSize
        (
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            &zero,
            vecY,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize
        ) != CUSPARSE_STATUS_SUCCESS
    )
    {
        return fail("cusparseSpMV_bufferSize");
    }

    if (bufferSize > 0)
    {
        if (cudaMalloc(&spmvBuffer, bufferSize) != cudaSuccess)
        {
            return fail("cudaMalloc(spmvBuffer)");
        }
    }

    // Optional: auto-tune poly-Jacobi parameters using power iteration on D^{-1}A
    if (usePolyJacobi && polyAuto)
    {
        // allocate temporary vectors
        DeviceScalar *d_powV = nullptr, *d_powW = nullptr;
        bool powerReady =
            cudaMalloc(reinterpret_cast<void**>(&d_powV), sizeof(DeviceScalar)*rows) == cudaSuccess
         && cudaMalloc(reinterpret_cast<void**>(&d_powW), sizeof(DeviceScalar)*rows) == cudaSuccess;

        if (powerReady)
        {
            // initialize v to ones
            Foam::List<DeviceScalar> onesHost(rows, DeviceScalar(1));
            cudaMemcpy(d_powV, onesHost.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice);

            const int iters = 8;
            double vNorm = 0.0;
            double lambda = 0.0;

            for (int k = 0; k < iters; ++k)
            {
                // w = A*v
                cusparseDnVecDescr_t vDesc = nullptr, wDesc = nullptr;
                if (cusparseCreateDnVec(&vDesc, rows, d_powV, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
                { powerReady = false; break; }
                if (cusparseCreateDnVec(&wDesc, rows, d_powW, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
                { cusparseDestroyDnVec(vDesc); powerReady = false; break; }
                const double oneD = 1.0, zeroD = 0.0;
                cusparseStatus_t st =
                    cusparseSpMV
                    (
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &oneD,
                        matA,
                        vDesc,
                        &zeroD,
                        wDesc,
                        CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        spmvBuffer
                    );
                cusparseDestroyDnVec(vDesc);
                cusparseDestroyDnVec(wDesc);
                if (st != CUSPARSE_STATUS_SUCCESS)
                { powerReady = false; break; }

                // w = D^{-1} w
                if
                (
                    cublasDdgmm
                    (
                        cublasHandle,
                        CUBLAS_SIDE_LEFT,
                        rows,
                        1,
                        reinterpret_cast<double*>(d_powW),
                        rows,
                        reinterpret_cast<const double*>(d_diagInv),
                        1,
                        reinterpret_cast<double*>(d_powW),
                        rows
                    ) != CUBLAS_STATUS_SUCCESS
                )
                { powerReady = false; break; }

                // Rayleigh quotient approx: lambda ≈ (v^T w)/(v^T v)
                double vTw = 0.0, vTv = 0.0;
                if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_powV), 1, reinterpret_cast<const double*>(d_powW), 1, &vTw) != CUBLAS_STATUS_SUCCESS)
                { powerReady = false; break; }
                if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_powV), 1, reinterpret_cast<const double*>(d_powV), 1, &vTv) != CUBLAS_STATUS_SUCCESS)
                { powerReady = false; break; }
                if (vTv > 0)
                {
                    lambda = vTw / vTv;
                }

                // normalize w -> v
                if (cublasDnrm2(cublasHandle, rows, reinterpret_cast<const double*>(d_powW), 1, &vNorm) != CUBLAS_STATUS_SUCCESS)
                { powerReady = false; break; }
                if (vNorm > 0)
                {
                    const double invNorm = 1.0 / vNorm;
                    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_powW), 1, reinterpret_cast<double*>(d_powV), 1) != CUBLAS_STATUS_SUCCESS)
                    { powerReady = false; break; }
                    if (cublasDscal(cublasHandle, rows, &invNorm, reinterpret_cast<double*>(d_powV), 1) != CUBLAS_STATUS_SUCCESS)
                    { powerReady = false; break; }
                }
            }

            // choose omega and sweeps heuristically
            double lambdaAbs = std::fabs(lambda);
            if (lambdaAbs <= 1e-12) lambdaAbs = 1.0; // fallback
            jacOmegaEff = std::min(0.95, std::max(0.3, 0.8/lambdaAbs));
            if (polySweepsEff <= 0)
            {
                polySweepsEff = (rows > 50000 ? 3 : 2);
            }

            if (reportStats)
            {
                Info<< "cudaPCG: polyJacobi auto omega=" << jacOmegaEff
                    << ", sweeps=" << polySweepsEff << nl;
            }

            cudaFree(d_powV);
            cudaFree(d_powW);
        }
        if (!powerReady)
        {
            // default heuristic
            jacOmegaEff = 0.7;
            if (polySweepsEff <= 0) polySweepsEff = 2;
        }
        // ensure poly-Jacobi will run
        usePolyJacobi = true;
    }

    // r = b - A*x
    std::chrono::steady_clock::time_point setupStart;
    if (logIterationStats)
    {
        setupStart = std::chrono::steady_clock::now();
    }

    if
    (
        cusparseSpMV
        (
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            &zero,
            vecY,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            spmvBuffer
        ) != CUSPARSE_STATUS_SUCCESS
    )
    {
        return fail("cusparseSpMV (A*x)");
    }
    if (!syncStream())
    {
        return false;
    }
    if (logIterationStats)
    {
        setupTimeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setupStart).count();
    }

    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_b), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas copy b->r");
    if (cublasDaxpy(cublasHandle, rows, reinterpret_cast<const double*>(&negOne), reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas axpy r -= Ap");

    if (!applyPreconditionerVec(reinterpret_cast<const double*>(d_r), reinterpret_cast<double*>(d_z), "initial"))
        return fail(message);

    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas copy z->p");

    scalarField wA(nCells, Zero);
    scalarField pA(nCells, Zero);

    cudaMemcpy(wA.begin(), d_Ap, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(pA.begin(), d_p, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

    double rzDevice = 0.0;
    if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, reinterpret_cast<const double*>(d_z), 1, &rzDevice) != CUBLAS_STATUS_SUCCESS)
    {
        cleanup();
        return fail("cublas dot(r,z)");
    }
    if (!syncStream())
    {
        return false;
    }
    scalar rz = returnReduce(static_cast<scalar>(rzDevice), sumOp<scalar>());

    scalar normFactor = this->normFactor(psi, source, wA, pA);

    double sumAbsDevice = 0.0;
    if (cublasDasum(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, &sumAbsDevice) != CUBLAS_STATUS_SUCCESS)
    {
        cleanup();
        return fail("cublas dasum(r)");
    }
    if (!syncStream())
    {
        return false;
    }
    scalar sumAbs = returnReduce(static_cast<scalar>(sumAbsDevice), sumOp<scalar>());

    solverPerf.initialResidual() = sumAbs/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();
    bestResidualSoFar = solverPerf.initialResidual();
    if (earlyAbortOnStall)
    {
        residualWindow.clear();
        residualWindow.push_back(solverPerf.initialResidual());
        stallInitCounter = 0;
        stallWindowCounter = 0;
    }

    if (logResidualTrajectory)
    {
        residualIterHistory.push_back(0);
        residualValueHistory.push_back(solverPerf.initialResidual());
    }

    if
    (
        minIter_ <= 0
     && solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        cudaMemcpy(psi.begin(), d_x, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
        if (equilibrateMatrix && scaleHost.size())
        {
            for (int i = 0; i < rows; ++i)
            {
                psi[i] = static_cast<scalar>(static_cast<double>(psi[i]) * static_cast<double>(scaleHost[i]));
            }
        }
        cleanup();
        writeTelemetry(true, word::null);
        if (useColour && autoTuneEnabled(dict))
        {
            colourAutoTuneRecordSuccess(fieldName_);
        }
        return true;
    }

    label iter = 0;

    while
    (
        (
          iter < maxIter_
        && !solverPerf.checkConvergence(tolerance_, relTol_)
        )
     || iter < minIter_
    )
    {
        std::chrono::steady_clock::time_point iterStart;
        if (logIterationStats)
        {
            iterStart = std::chrono::steady_clock::now();
        }
        if (cusparseDnVecSetValues(vecX, d_p) != CUSPARSE_STATUS_SUCCESS)
        {
            return fail("cusparseSetVec(p)");
        }
        if (cusparseDnVecSetValues(vecY, d_Ap) != CUSPARSE_STATUS_SUCCESS)
        {
            return fail("cusparseSetVec(Ap)");
        }

        bool launchedFromGraph = false;
        if (useCudaGraph)
        {
            cudaStream_t activeStream = computeStream ? computeStream : nullptr;
            if (!spmvGraphReady && iter >= cudaGraphWarmup)
            {
                if (cudaStreamBeginCapture(activeStream, cudaStreamCaptureModeGlobal) != cudaSuccess)
                {
                    return fail("cudaStreamBeginCapture");
                }
                cusparseStatus_t capStatus =
                    cusparseSpMV
                    (
                        cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        matA,
                        vecX,
                        &zero,
                        vecY,
                        CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        spmvBuffer
                    );
                if (capStatus != CUSPARSE_STATUS_SUCCESS)
                {
                    cudaStreamEndCapture(activeStream, &spmvGraph);
                    return fail("cusparseSpMV (capture)");
                }
                if (cudaStreamEndCapture(activeStream, &spmvGraph) != cudaSuccess)
                {
                    return fail("cudaStreamEndCapture");
                }
                if (cudaGraphInstantiate(&spmvGraphExec, spmvGraph, nullptr, nullptr, 0) != cudaSuccess)
                {
                    return fail("cudaGraphInstantiate");
                }
                spmvGraphReady = true;
                if (cudaGraphLaunch(spmvGraphExec, activeStream) != cudaSuccess)
                {
                    return fail("cudaGraphLaunch");
                }
                launchedFromGraph = true;
            }
            else if (spmvGraphReady)
            {
                cudaStream_t launchStream = computeStream ? computeStream : nullptr;
                if (cudaGraphLaunch(spmvGraphExec, launchStream) != cudaSuccess)
                {
                    return fail("cudaGraphLaunch");
                }
                launchedFromGraph = true;
            }
        }

        if (!launchedFromGraph)
        {
            if
            (
                cusparseSpMV
                (
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    matA,
                    vecX,
                    &zero,
                    vecY,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT,
                    spmvBuffer
                ) != CUSPARSE_STATUS_SUCCESS
            )
            {
                return fail("cusparseSpMV (A*p)");
            }
        }

        if (!syncStream())
        {
            return false;
        }

        // Restore vecX to x for completeness
        cusparseDnVecSetValues(vecX, d_x);
        cusparseDnVecSetValues(vecY, d_Ap);

        double pApDevice = 0.0;
        if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_p), 1, reinterpret_cast<const double*>(d_Ap), 1, &pApDevice) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dot(p,Ap)");
        }
        if (!syncStream())
        {
            return false;
        }
        scalar pAp = returnReduce(static_cast<scalar>(pApDevice), sumOp<scalar>());

        if (solverPerf.checkSingularity(mag(pAp)/normFactor))
        {
            break;
        }

        const scalar alphaStep = rz/pAp;
        const double alphaDevice = static_cast<double>(alphaStep);
        const double negAlphaDevice = -alphaDevice;

        if (cublasDaxpy(cublasHandle, rows, &alphaDevice, reinterpret_cast<const double*>(d_p), 1, reinterpret_cast<double*>(d_x), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas axpy x += alpha*p");
        if (cublasDaxpy(cublasHandle, rows, &negAlphaDevice, reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas axpy r -= alpha*Ap");

        if (cublasDasum(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, &sumAbsDevice) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dasum(r iter)");
        }
        if (!syncStream())
        {
            return false;
        }
        sumAbs = returnReduce(static_cast<scalar>(sumAbsDevice), sumOp<scalar>());

        solverPerf.finalResidual() = sumAbs/normFactor;

        if (earlyAbortOnStall)
        {
            const scalar initRes = solverPerf.initialResidual();
            if (initRes > VSMALL)
            {
                const scalar ratioInit = solverPerf.finalResidual()/initRes;
                if (!std::isfinite(ratioInit))
                {
                    return fail("stallDetected");
                }
                if (ratioInit > stallRatioTol)
                {
                    ++stallInitCounter;
                }
                else
                {
                    stallInitCounter = 0;
                }

                if (stallInitCounter >= stallWindow)
                {
                    return fail("stallDetected");
                }
            }

            if (solverPerf.finalResidual() < bestResidualSoFar)
            {
                bestResidualSoFar = solverPerf.finalResidual();
            }

            if (!residualWindow.empty())
            {
                scalar windowMin = residualWindow.front();
                for (const scalar val : residualWindow)
                {
                    if (val < windowMin)
                    {
                        windowMin = val;
                    }
                }

                if (windowMin > VSMALL)
                {
                    const scalar ratioWindow = solverPerf.finalResidual()/windowMin;
                    if (!std::isfinite(ratioWindow))
                    {
                        return fail("stallDetected");
                    }
                    if (ratioWindow > stallBestRatioTol)
                    {
                        ++stallWindowCounter;
                    }
                    else
                    {
                        stallWindowCounter = 0;
                    }

                    if (stallWindowCounter >= stallBestWindow)
                    {
                        return fail("stallDetected");
                    }
                }
            }

            residualWindow.push_back(solverPerf.finalResidual());
            if (residualWindow.size() > static_cast<std::size_t>(stallBestWindow))
            {
                residualWindow.pop_front();
            }
        }

        ++iter;
        solverPerf.nIterations() = iter;

        if (logResidualTrajectory && (iter % residualLogEvery == 0))
        {
            residualIterHistory.push_back(iter);
            residualValueHistory.push_back(solverPerf.finalResidual());
        }

        if
        (
            iter >= maxIter_
         && solverPerf.checkConvergence(tolerance_, relTol_)
        )
        {
            break;
        }

        if (!applyPreconditionerVec(reinterpret_cast<const double*>(d_r), reinterpret_cast<double*>(d_z), "iter"))
            return fail(message);

        double rzNewDevice = 0.0;
        if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, reinterpret_cast<const double*>(d_z), 1, &rzNewDevice) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dot(r,z) iter");
        }
        if (!syncStream())
        {
            return false;
        }

        const scalar rzOld = rz;
        rz = returnReduce(static_cast<scalar>(rzNewDevice), sumOp<scalar>());

        const scalar betaStep = rzOld != 0 ? rz/rzOld : 0;
        const double betaDevice = static_cast<double>(betaStep);

        if (usePipelinedCG)
        {
            if (cublasDscal(cublasHandle, rows, &betaDevice, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            {
                return fail("cublas scal p*=beta");
            }
            const double minusAlphaBetaDevice = -alphaDevice * betaDevice;
            if (cublasDaxpy(cublasHandle, rows, &minusAlphaBetaDevice, reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            {
                return fail("cublas axpy p-=alphaBetaAp");
            }
            if (cublasDaxpy(cublasHandle, rows, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            {
                return fail("cublas axpy p+=z");
            }
        }
        else
        {
            if (cublasDscal(cublasHandle, rows, &betaDevice, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            {
                return fail("cublas scal p*=beta");
            }
            if (cublasDaxpy(cublasHandle, rows, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            {
                return fail("cublas axpy p+=z");
            }
        }

        if (logIterationStats)
        {
            const double iterSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - iterStart).count();
            iterationAccumSeconds += iterSeconds;
            if (iterSeconds > iterationMaxSeconds)
            {
                iterationMaxSeconds = iterSeconds;
            }
            ++iterationCountStats;
        }
    }

    cudaMemcpy(psi.begin(), d_x, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    if (equilibrateMatrix && scaleHost.size())
    {
        for (int i = 0; i < rows; ++i)
        {
            psi[i] = static_cast<scalar>(static_cast<double>(psi[i]) * static_cast<double>(scaleHost[i]));
        }
    }
    cleanup();
    if (logResidualTrajectory)
    {
        const label lastRecordedIter =
            residualIterHistory.empty()
          ? -1
          : residualIterHistory.back();
        if (solverPerf.nIterations() > lastRecordedIter)
        {
            residualIterHistory.push_back(solverPerf.nIterations());
            residualValueHistory.push_back(solverPerf.finalResidual());
        }
    }
    writeTelemetry(true, word::null);
    if (logIterationStats && Pstream::master())
    {
        double avgIterSeconds = iterationCountStats > 0 ? iterationAccumSeconds/static_cast<double>(iterationCountStats) : 0.0;
        Info<< "cudaPCG(" << fieldName_ << "): timing setup=" << setupTimeSeconds
            << " s, iterations=" << iterationCountStats
            << ", avgIter=" << avgIterSeconds << " s"
            << ", maxIter=" << iterationMaxSeconds << " s" << nl;
    }
    if (useColour && autoTuneEnabled(dict))
    {
        colourAutoTuneRecordSuccess(fieldName_);
    }
    return true;
}

#endif
