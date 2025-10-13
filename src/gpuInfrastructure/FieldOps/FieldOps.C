#include "FieldOps.H"
#include "error.H"

#include <string>
#include <sstream>

#ifdef FOAM_USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <nvrtc.h>
#endif

namespace Foam
{
namespace gpu
{

#ifdef FOAM_USE_CUDA
namespace
{

std::string nvrtcErrorString(const nvrtcResult code)
{
    return std::string(nvrtcGetErrorString(code))
        + " (" + Foam::name(static_cast<int>(code)) + ')';
}


std::string cudaDriverErrorString(const CUresult code)
{
    const char* msg = nullptr;
    cuGetErrorString(code, &msg);
    if (msg)
    {
        return std::string(msg)
            + " (" + Foam::name(static_cast<int>(code)) + ')';
    }
    return "cuda driver error (" + Foam::name(static_cast<int>(code)) + ')';
}


bool compileSurfaceSubtractKernel(CUmodule& module, CUfunction& function, word& error)
{
    static bool initialised = false;
    static CUmodule cachedModule = nullptr;
    static CUfunction cachedFunction = nullptr;

    if (initialised)
    {
        module = cachedModule;
        function = cachedFunction;
        error.clear();
        return cachedModule != nullptr && cachedFunction != nullptr;
    }

    initialised = true;

    const char* source = R"(
extern "C" __global__
void surfaceSubtract(double* dst, const double* a, const double* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = a[idx] - b[idx];
    }
}
)";

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source,
        "surfaceSubtract.cu",
        0,
        nullptr,
        nullptr
    );

    if (nvStatus != NVRTC_SUCCESS)
    {
        error = "nvrtcCreateProgram failed: " + nvrtcErrorString(nvStatus);
        return false;
    }

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    std::ostringstream arch;
    arch<< "--gpu-architecture=compute_" << prop.major << prop.minor;
    const std::string archStr = arch.str();
    const char* options[] = { "--std=c++14", archStr.c_str() };
    nvStatus = nvrtcCompileProgram(prog, 2, options);

    if (nvStatus != NVRTC_SUCCESS)
    {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        if (logSize > 1)
        {
            nvrtcGetProgramLog(prog, &log[0]);
        }

        error = "nvrtcCompileProgram failed: " + nvrtcErrorString(nvStatus);
        if (!log.empty())
        {
            error += " :: " + log;
        }
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    CUresult cuStatus = cuModuleLoadData(&cachedModule, ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        cachedModule = nullptr;
        cachedFunction = nullptr;
        return false;
    }

    cuStatus = cuModuleGetFunction(&cachedFunction, cachedModule, "surfaceSubtract");
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cachedModule = nullptr;
        cachedFunction = nullptr;
        return false;
    }

    module = cachedModule;
    function = cachedFunction;
    error.clear();
    return true;
}

} // namespace
#endif


bool subtractSurfaceScalarFields
(
    Context& ctx,
    DeviceField<surfaceScalarField>& dst,
    const DeviceField<surfaceScalarField>& a,
    const DeviceField<surfaceScalarField>& b,
    word& error,
    OperationStats* stats
)
{
#ifdef FOAM_USE_CUDA
    if (!ctx.ready())
    {
        if (!ctx.initialise(0, error))
        {
            return false;
        }
    }

    if (!dst.allocated())
    {
        if (!dst.syncHostToDevice(ctx, error))
        {
            return false;
        }
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!compileSurfaceSubtractKernel(module, function, error))
    {
        return false;
    }

    const label n = dst.size();
    if (n == 0)
    {
        error.clear();
        return true;
    }

    double* dstPtr = reinterpret_cast<double*>(dst.devicePointer());
    double* aPtr = const_cast<double*>
    (
        reinterpret_cast<const double*>(a.devicePointer())
    );
    double* bPtr = const_cast<double*>
    (
        reinterpret_cast<const double*>(b.devicePointer())
    );

    if (!dstPtr || !aPtr || !bPtr)
    {
        error = "Device pointers are null";
        return false;
    }

    int nCells = static_cast<int>(n);
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (nCells + blockSize - 1)/blockSize;

    void* args[] = { &dstPtr, &aPtr, &bPtr, &nCells };

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    if (stats)
    {
        cudaEventCreateWithFlags(&startEvent, cudaEventDefault);
        cudaEventCreateWithFlags(&stopEvent, cudaEventDefault);
        cudaEventRecord(startEvent, ctx.stream());
    }

    CUresult status = cuLaunchKernel
    (
        function,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0,
        reinterpret_cast<CUstream>(ctx.stream()),
        args,
        nullptr
    );

    if (status != CUDA_SUCCESS)
    {
        error = "cuLaunchKernel failed: " + cudaDriverErrorString(status);
        return false;
    }

    if (stats)
    {
        cudaEventRecord(stopEvent, ctx.stream());
        cudaEventSynchronize(stopEvent);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);

        stats->gpuMilliseconds = static_cast<scalar>(ms);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    else
    {
        status = cuStreamSynchronize(reinterpret_cast<CUstream>(ctx.stream()));
        if (status != CUDA_SUCCESS)
        {
            error = "cuStreamSynchronize failed: " + cudaDriverErrorString(status);
            return false;
        }
    }

    error.clear();
    return true;
#else
    (void)ctx;
    (void)dst;
    (void)a;
    (void)b;
    error = "CUDA support not available";
    return false;
#endif
}

#ifdef FOAM_USE_CUDA
bool compileVelocityCorrectorKernel(CUmodule& module, CUfunction& function, word& error)
{
    static bool initialised = false;
    static CUmodule cachedModule = nullptr;
    static CUfunction cachedFunction = nullptr;

    if (initialised)
    {
        module = cachedModule;
        function = cachedFunction;
        error.clear();
        return cachedModule != nullptr && cachedFunction != nullptr;
    }

    initialised = true;

    const char* source = R"(
extern "C" __global__
void velocityCorrector(double* u, const double* h, const double* grad, const double* r, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        double rVal = r[idx];
        int base = idx * 3;
        u[base]     = h[base]     - rVal * grad[base];
        u[base + 1] = h[base + 1] - rVal * grad[base + 1];
        u[base + 2] = h[base + 2] - rVal * grad[base + 2];
    }
}
)";

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source,
        "velocityCorrector.cu",
        0,
        nullptr,
        nullptr
    );

    if (nvStatus != NVRTC_SUCCESS)
    {
        error = "nvrtcCreateProgram failed: " + nvrtcErrorString(nvStatus);
        return false;
    }

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    std::ostringstream arch;
    arch<< "--gpu-architecture=compute_" << prop.major << prop.minor;
    const std::string archStr = arch.str();
    const char* options[] = { "--std=c++14", archStr.c_str() };
    nvStatus = nvrtcCompileProgram(prog, 2, options);

    if (nvStatus != NVRTC_SUCCESS)
    {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        if (logSize > 1)
        {
            nvrtcGetProgramLog(prog, &log[0]);
        }

        error = "nvrtcCompileProgram failed: " + nvrtcErrorString(nvStatus);
        if (!log.empty())
        {
            error += " :: " + log;
        }
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    CUresult cuStatus = cuModuleLoadData(&cachedModule, ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        cachedModule = nullptr;
        cachedFunction = nullptr;
        return false;
    }

    cuStatus = cuModuleGetFunction(&cachedFunction, cachedModule, "velocityCorrector");
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cachedModule = nullptr;
        cachedFunction = nullptr;
        return false;
    }

    module = cachedModule;
    function = cachedFunction;
    error.clear();
    return true;
}
#endif


bool velocityCorrector
(
    Context& ctx,
    DeviceField<volVectorField>& U,
    const DeviceField<volVectorField>& HbyA,
    const DeviceField<volScalarField>& rAtU,
    const DeviceField<volVectorField>& gradP,
    word& error,
    OperationStats* stats
)
{
#ifdef FOAM_USE_CUDA
    if (!ctx.ready())
    {
        if (!ctx.initialise(0, error))
        {
            return false;
        }
    }

    if (!U.allocated())
    {
        if (!U.syncHostToDevice(ctx, error))
        {
            return false;
        }
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!compileVelocityCorrectorKernel(module, function, error))
    {
        return false;
    }

    const label n = U.size();
    if (n == 0)
    {
        error.clear();
        return true;
    }

    double* uPtr = reinterpret_cast<double*>(U.devicePointer());
    double* hPtr = const_cast<double*>
    (
        reinterpret_cast<const double*>(HbyA.devicePointer())
    );
    double* gradPtr = const_cast<double*>
    (
        reinterpret_cast<const double*>(gradP.devicePointer())
    );
    double* rPtr = const_cast<double*>
    (
        reinterpret_cast<const double*>(rAtU.devicePointer())
    );

    if (!uPtr || !hPtr || !gradPtr || !rPtr)
    {
        error = "Device pointers are null";
        return false;
    }

    int nCells = static_cast<int>(n);
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (nCells + blockSize - 1)/blockSize;

    void* args[] = { &uPtr, &hPtr, &gradPtr, &rPtr, &nCells };

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    if (stats)
    {
        cudaEventCreateWithFlags(&startEvent, cudaEventDefault);
        cudaEventCreateWithFlags(&stopEvent, cudaEventDefault);
        cudaEventRecord(startEvent, ctx.stream());
    }

    CUresult status = cuLaunchKernel
    (
        function,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0,
        reinterpret_cast<CUstream>(ctx.stream()),
        args,
        nullptr
    );

    if (status != CUDA_SUCCESS)
    {
        error = "cuLaunchKernel failed: " + cudaDriverErrorString(status);
        return false;
    }

    if (stats)
    {
        cudaEventRecord(stopEvent, ctx.stream());
        cudaEventSynchronize(stopEvent);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);

        stats->gpuMilliseconds = static_cast<scalar>(ms);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    else
    {
        status = cuStreamSynchronize(reinterpret_cast<CUstream>(ctx.stream()));
        if (status != CUDA_SUCCESS)
        {
            error = "cuStreamSynchronize failed: " + cudaDriverErrorString(status);
            return false;
        }
    }

    error.clear();
    return true;
#else
    (void)ctx;
    (void)U;
    (void)HbyA;
    (void)rAtU;
    (void)gradP;
    (void)stats;
    error = "CUDA support not available";
    return false;
#endif
}

#ifdef FOAM_USE_CUDA
namespace
{

bool compileNutKernel(CUmodule& module, CUfunction& function, word& error)
{
    static bool initialised = false;
    static CUmodule cachedModule = nullptr;
    static CUfunction cachedFunction = nullptr;

    if (initialised)
    {
        module = cachedModule;
        function = cachedFunction;
        error.clear();
        return cachedModule != nullptr && cachedFunction != nullptr;
    }

    initialised = true;

    const char* source = R"(
extern "C" __global__
void computeNut(double* nut, const double* k, const double* epsilon,
                const double Cmu, const double kMin, const double epsilonMin,
                const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    double kval = k[idx];
    if (kval < kMin)
    {
        kval = kMin;
    }

    double eps = epsilon[idx];
    if (eps < epsilonMin)
    {
        eps = epsilonMin;
    }

    const double value = Cmu * kval * kval / eps;
    nut[idx] = value;
}
)";

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source,
        "computeNut.cu",
        0,
        nullptr,
        nullptr
    );

    if (nvStatus != NVRTC_SUCCESS)
    {
        error = "nvrtcCreateProgram failed: " + nvrtcErrorString(nvStatus);
        return false;
    }

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    std::ostringstream arch;
    arch<< "--gpu-architecture=compute_" << prop.major << prop.minor;
    const std::string archStr = arch.str();
    const char* options[] = { "--std=c++14", archStr.c_str() };
    nvStatus = nvrtcCompileProgram(prog, 2, options);

    if (nvStatus != NVRTC_SUCCESS)
    {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        if (logSize > 1)
        {
            nvrtcGetProgramLog(prog, &log[0]);
        }

        error = "nvrtcCompileProgram failed: " + nvrtcErrorString(nvStatus);
        if (!log.empty())
        {
            error += " :: " + log;
        }
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    CUresult cuStatus = cuModuleLoadData(&cachedModule, ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        cachedModule = nullptr;
        cachedFunction = nullptr;
        return false;
    }

    cuStatus = cuModuleGetFunction(&cachedFunction, cachedModule, "computeNut");
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cuModuleUnload(cachedModule);
        cachedModule = nullptr;
        cachedFunction = nullptr;
        return false;
    }

    module = cachedModule;
    function = cachedFunction;
    error.clear();
    return true;
}

} // namespace
#endif


bool computeNutFromKEpsilon
(
    Context& ctx,
    DeviceField<volScalarField>& nut,
    const DeviceField<volScalarField>& k,
    const DeviceField<volScalarField>& epsilon,
    const scalar Cmu,
    const scalar kMin,
    const scalar epsilonMin,
    word& error,
    OperationStats* stats
)
{
#ifdef FOAM_USE_CUDA
    if (!ctx.ready())
    {
        if (!ctx.initialise(0, error))
        {
            return false;
        }
    }

    const label nCells = nut.size();
    if (nCells == 0)
    {
        error.clear();
        return true;
    }

    if (k.size() != nCells || epsilon.size() != nCells)
    {
        error = "computeNut: field sizes do not match";
        return false;
    }

    DeviceField<volScalarField>& nutMutable = nut;
    DeviceField<volScalarField>& kMutable =
        const_cast<DeviceField<volScalarField>&>(k);
    DeviceField<volScalarField>& epsilonMutable =
        const_cast<DeviceField<volScalarField>&>(epsilon);

    nutMutable.markHostDirty();
    kMutable.markHostDirty();
    epsilonMutable.markHostDirty();

    if
    (
        !nutMutable.syncHostToDevice(ctx, error)
     || !kMutable.syncHostToDevice(ctx, error)
     || !epsilonMutable.syncHostToDevice(ctx, error)
    )
    {
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!compileNutKernel(module, function, error))
    {
        return false;
    }

    double* nutPtr = reinterpret_cast<double*>(nutMutable.devicePointer());
    double* kPtr = reinterpret_cast<double*>(kMutable.devicePointer());
    double* epsPtr = reinterpret_cast<double*>(epsilonMutable.devicePointer());

    if (!nutPtr || !kPtr || !epsPtr)
    {
        error = "computeNut: null device pointer";
        return false;
    }

    double CmuVal = static_cast<double>(Cmu);
    double kMinVal = static_cast<double>(kMin);
    double epsMinVal = static_cast<double>(epsilonMin);
    int cells = static_cast<int>(nCells);

    void* args[] =
    {
        &nutPtr,
        &kPtr,
        &epsPtr,
        &CmuVal,
        &kMinVal,
        &epsMinVal,
        &cells
    };

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    if (stats)
    {
        cudaEventCreateWithFlags(&startEvent, cudaEventDefault);
        cudaEventCreateWithFlags(&stopEvent, cudaEventDefault);
        cudaEventRecord(startEvent, ctx.stream());
    }

    CUresult status = cuLaunchKernel
    (
        function,
        (cells + 255)/256, 1, 1,
        256, 1, 1,
        0,
        reinterpret_cast<CUstream>(ctx.stream()),
        args,
        nullptr
    );

    if (status != CUDA_SUCCESS)
    {
        error = "cuLaunchKernel(computeNut) failed: "
          + cudaDriverErrorString(status);
        return false;
    }

    if (stats)
    {
        cudaEventRecord(stopEvent, ctx.stream());
        cudaEventSynchronize(stopEvent);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        stats->gpuMilliseconds = static_cast<scalar>(ms);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    else
    {
        status = cuStreamSynchronize(reinterpret_cast<CUstream>(ctx.stream()));
        if (status != CUDA_SUCCESS)
        {
            error = "cuStreamSynchronize failed: "
              + cudaDriverErrorString(status);
            return false;
        }
    }

    nutMutable.markDeviceDirty();
    if (!nutMutable.syncDeviceToHost(ctx, error))
    {
        return false;
    }

    error.clear();
    return true;
#else
    (void)ctx;
    (void)nut;
    (void)k;
    (void)epsilon;
    (void)Cmu;
    (void)kMin;
    (void)epsilonMin;
    (void)stats;
    error = "CUDA support not available";
    return false;
#endif
}

} // namespace gpu
} // namespace Foam
