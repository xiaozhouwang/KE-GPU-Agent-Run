#include "FvOperators.H"
#include "GpuContext.H"
#include "DeviceField.H"
#include "gaussConvectionScheme.H"
#include "gaussLaplacianScheme.H"
#include "convectionScheme.H"
#include "laplacianScheme.H"
#include "surfaceInterpolationScheme.H"
#include "fvcSurfaceIntegrate.H"
#include "fvcDiv.H"
#include "fvc.H"
#include "correctedSnGrad.H"
#include "IStringStream.H"
#include "surfaceInterpolate.H"
#include "fvMesh.H"
#include "fvMatrices.H"
#include "Time.H"
#include "pTraits.H"

#include <sstream>
#include <map>

#ifdef FOAM_USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <nvrtc.h>
#endif

namespace Foam
{
namespace gpu
{
namespace fvm
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


struct KernelEntry
{
    CUmodule module{nullptr};
    CUfunction function{nullptr};
};


bool compileEulerDdtKernel
(
    const int components,
    CUmodule& module,
    CUfunction& function,
    word& error
)
{
    static std::map<int, KernelEntry> cache;

    const auto iter = cache.find(components);
    if (iter != cache.end())
    {
        module = iter->second.module;
        function = iter->second.function;
        error.clear();
        return module != nullptr && function != nullptr;
    }

    std::ostringstream source;
    source
        << "#define COMPONENTS " << components << '\n'
        << "extern \"C\" __global__\n"
        << "void assembleEulerDdt(\n"
        << "    double* diag,\n"
        << "    double* source,\n"
        << "    const double* volumeDiag,\n"
        << "    const double* volumeSource,\n"
        << "    const double* fieldOld,\n"
        << "    const double rDeltaT,\n"
        << "    const int nCells)\n"
        << "{\n"
        << "    const int cell = blockIdx.x * blockDim.x + threadIdx.x;\n"
        << "    if (cell >= nCells)\n"
        << "    {\n"
        << "        return;\n"
        << "    }\n"
        << "    const double diagVal = rDeltaT * volumeDiag[cell];\n"
        << "    diag[cell] = diagVal;\n"
        << "    const double coeff = rDeltaT * volumeSource[cell];\n"
        << "    const double* uOld = fieldOld + cell*COMPONENTS;\n"
        << "    double* src = source + cell*COMPONENTS;\n"
        << "    #pragma unroll\n"
        << "    for (int c = 0; c < COMPONENTS; ++c)\n"
        << "    {\n"
        << "        src[c] = coeff * uOld[c];\n"
        << "    }\n"
        << "}\n";

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source.str().c_str(),
        "assembleEulerDdt.cu",
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

    KernelEntry entry;

    CUresult cuStatus = cuModuleLoadData(&entry.module, ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        return false;
    }

    cuStatus = cuModuleGetFunction
    (
        &entry.function,
        entry.module,
        "assembleEulerDdt"
    );

    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cuModuleUnload(entry.module);
        entry = KernelEntry{};
        return false;
    }

    cache.emplace(components, entry);
    module = entry.module;
    function = entry.function;
    error.clear();
    return true;
}


bool compileGaussDivKernel(CUmodule& module, CUfunction& function, word& error)
{
    static bool initialised = false;
    static KernelEntry cache;

    if (initialised)
    {
        module = cache.module;
        function = cache.function;
        error.clear();
        return cache.module != nullptr && cache.function != nullptr;
    }

    initialised = true;

    const char* source = R"(
extern "C" __global__
void assembleGaussDiv(double* lower, double* upper, const double* faceFlux, int nFaces)
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nFaces)
    {
        return;
    }

    const double phi = faceFlux[face];
    const double lowerVal = phi > 0.0 ? -phi : 0.0;
    const double upperVal = lowerVal + phi;

    lower[face] = lowerVal;
    upper[face] = upperVal;
}
)";

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source,
        "assembleGaussDiv.cu",
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

    CUresult cuStatus = cuModuleLoadData(&cache.module, ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        cache = KernelEntry{};
        return false;
    }

    cuStatus = cuModuleGetFunction
    (
        &cache.function,
        cache.module,
        "assembleGaussDiv"
    );

    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cuModuleUnload(cache.module);
        cache = KernelEntry{};
        return false;
    }

    module = cache.module;
    function = cache.function;
    error.clear();
    return true;
}

bool compileGaussLaplacianKernel
(
    CUmodule& module,
    CUfunction& function,
    word& error
)
{
    static bool initialised = false;
    static KernelEntry cache;

    if (initialised)
    {
        module = cache.module;
        function = cache.function;
        error.clear();
        return cache.module != nullptr && cache.function != nullptr;
    }

    initialised = true;

    const char* source = R"(
extern "C" __global__
void assembleGaussLaplacian(double* upper, const double* gammaMagSf, const double* deltaCoeffs, int nFaces)
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nFaces)
    {
        return;
    }

    upper[face] = gammaMagSf[face]*deltaCoeffs[face];
}
)";

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source,
        "assembleGaussLaplacian.cu",
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

    CUresult cuStatus = cuModuleLoadData(&cache.module, ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        cache = KernelEntry{};
        return false;
    }

    cuStatus = cuModuleGetFunction
    (
        &cache.function,
        cache.module,
        "assembleGaussLaplacian"
    );

    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cuModuleUnload(cache.module);
        cache = KernelEntry{};
        return false;
    }

    module = cache.module;
    function = cache.function;
    error.clear();
    return true;
}

class DeviceBuffer
{
    void* ptr_;

public:
    DeviceBuffer() : ptr_(nullptr) {}

    ~DeviceBuffer()
    {
        if (ptr_)
        {
            cudaFree(ptr_);
        }
    }

    void* get() const { return ptr_; }

    template<class T>
    T* as() const { return reinterpret_cast<T*>(ptr_); }

    bool allocate(std::size_t bytes, word& error)
    {
        if (ptr_)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }

        if (!bytes)
        {
            return true;
        }

        cudaError_t status = cudaMalloc(&ptr_, bytes);
        if (status != cudaSuccess)
        {
            error = "cudaMalloc failed";
            ptr_ = nullptr;
            return false;
        }

        return true;
    }
};

} // namespace
#endif


template<class Type>
tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> gammaSnGradCorrection
(
    const surfaceVectorField& SfGammaCorr,
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    const fvMesh& mesh = vf.mesh();

    tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> tCorr
    (
        GeometricField<Type, fvsPatchField, surfaceMesh>::New
        (
            "gammaSnGradCorr(" + vf.name() + ')',
            mesh,
            SfGammaCorr.dimensions()
           *vf.dimensions()
           *mesh.deltaCoeffs().dimensions()
        )
    );

    GeometricField<Type, fvsPatchField, surfaceMesh>& corr = tCorr.ref();

    for (direction cmpt = 0; cmpt < pTraits<Type>::nComponents; ++cmpt)
    {
        corr.replace
        (
            cmpt,
            fvc::dotInterpolate
            (
                SfGammaCorr,
                fvc::grad(vf.component(cmpt))
            )
        );
    }

    return tCorr;
}


template<class Type>
tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> correctedSnGradCorrection
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    const fvMesh& mesh = vf.mesh();

    tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> tCorr
    (
        GeometricField<Type, fvsPatchField, surfaceMesh>::New
        (
            "snGradCorr(" + vf.name() + ')',
            mesh,
            vf.dimensions()*mesh.nonOrthDeltaCoeffs().dimensions()
        )
    );

    GeometricField<Type, fvsPatchField, surfaceMesh>& corr = tCorr.ref();

    for (direction cmpt = 0; cmpt < pTraits<Type>::nComponents; ++cmpt)
    {
        typedef typename pTraits<Type>::cmptType CmptType;
        IStringStream emptyStream("");
        fv::correctedSnGrad<CmptType> scheme(mesh, emptyStream);
        corr.replace
        (
            cmpt,
            scheme.fullGradCorrection(vf.component(cmpt))
        );
    }

    return tCorr;
}


template<class Type>
tmp<fvMatrix<Type>> ddt
(
    Context& ctx,
    FieldRegistry& registry,
    GeometricField<Type, fvPatchField, volMesh>& vf,
    bool& usedGpu,
    word& error,
    OperationStats* stats
)
{
    usedGpu = false;
    error.clear();

#ifndef FOAM_USE_CUDA
    error = "CUDA support not available";
    return tmp<fvMatrix<Type>>();
#else
    const fvMesh& mesh = vf.mesh();
    const word ddtKey("ddt(" + vf.name() + ')');
    const word schemeName(mesh.schemes().ddt(ddtKey));

    if (schemeName != "Euler")
    {
        error = "GPU ddt supports Euler scheme only (found " + schemeName + ")";
        return tmp<fvMatrix<Type>>();
    }

    if (!ctx.ready())
    {
        if (!ctx.initialise(0, error))
        {
            return tmp<fvMatrix<Type>>();
        }
    }

    DeviceField<GeometricField<Type, fvPatchField, volMesh>>* vfDevicePtr =
        nullptr;

    DeviceField<GeometricField<Type, fvPatchField, volMesh>>* vfOldDevicePtr =
        nullptr;

    try
    {
        vfDevicePtr = &registry.getOrCreate(vf);
        auto& vfOld =
            const_cast<GeometricField<Type, fvPatchField, volMesh>&>
            (
                vf.oldTime()
            );
        vfOldDevicePtr = &registry.getOrCreate(vfOld);
    }
    catch (...)
    {
        error = "Failed to access device field wrappers for ddt";
        return tmp<fvMatrix<Type>>();
    }

    DeviceField<GeometricField<Type, fvPatchField, volMesh>>& vfDevice =
        *vfDevicePtr;
    DeviceField<GeometricField<Type, fvPatchField, volMesh>>& vfOldDevice =
        *vfOldDevicePtr;

    if (!vfDevice.syncHostToDevice(ctx, error))
    {
        return tmp<fvMatrix<Type>>();
    }

    if (!vfOldDevice.syncHostToDevice(ctx, error))
    {
        return tmp<fvMatrix<Type>>();
    }

    const label nCells = vf.internalField().size();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            vf.dimensions()*dimVol/dimTime
        )
    );
    fvMatrix<Type>& mat = tfvm.ref();

    scalarField& diag = mat.diag();
    Field<Type>& source = mat.source();

    if (!nCells)
    {
        usedGpu = true;
        return tfvm;
    }

    const scalar rDeltaT = 1.0/mesh.time().deltaTValue();
    const scalarField& volumeDiag = mesh.Vsc();
    const scalarField& volumeSource =
        mesh.moving() ? mesh.Vsc0() : mesh.Vsc();

    DeviceBuffer diagDev;
    DeviceBuffer sourceDev;
    DeviceBuffer volDiagDev;
    DeviceBuffer volSrcDev;

    const std::size_t diagBytes = sizeof(scalar)*nCells;
    const std::size_t nComp = pTraits<Type>::nComponents;
    const std::size_t sourceBytes = sizeof(scalar)*nCells*nComp;

    if
    (
        !diagDev.allocate(diagBytes, error)
     || !sourceDev.allocate(sourceBytes, error)
     || !volDiagDev.allocate(diagBytes, error)
     || !volSrcDev.allocate(diagBytes, error)
    )
    {
        return tmp<fvMatrix<Type>>();
    }

    if (cudaMemcpy
        (
            volDiagDev.as<scalar>(),
            volumeDiag.begin(),
            diagBytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess)
    {
        error = "cudaMemcpy(volDiag host->device) failed";
        return tmp<fvMatrix<Type>>();
    }

    if (cudaMemcpy
        (
            volSrcDev.as<scalar>(),
            volumeSource.begin(),
            diagBytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess)
    {
        error = "cudaMemcpy(volSrc host->device) failed";
        return tmp<fvMatrix<Type>>();
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;

    if
    (
        !compileEulerDdtKernel(static_cast<int>(nComp), module, function, error)
    )
    {
        return tmp<fvMatrix<Type>>();
    }

    scalar* diagPtr = reinterpret_cast<scalar*>(diagDev.get());
    scalar* sourcePtr = reinterpret_cast<scalar*>(sourceDev.get());
    scalar* volDiagPtr = reinterpret_cast<scalar*>(volDiagDev.get());
    scalar* volSrcPtr = reinterpret_cast<scalar*>(volSrcDev.get());

    scalar* vfOldPtr = reinterpret_cast<scalar*>
    (
        vfOldDevice.devicePointer()
    );

    if (!vfOldPtr)
    {
        error = "ddt: device pointer for old field is null";
        return tmp<fvMatrix<Type>>();
    }

    const unsigned int blockSize = 256;
    const unsigned int gridSize = (nCells + blockSize - 1)/blockSize;

    scalar coeff = rDeltaT;
    int cells = static_cast<int>(nCells);

    void* args[] =
    {
        &diagPtr,
        &sourcePtr,
        &volDiagPtr,
        &volSrcPtr,
        &vfOldPtr,
        &coeff,
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

    CUresult launchStatus = cuLaunchKernel
    (
        function,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0,
        reinterpret_cast<CUstream>(ctx.stream()),
        args,
        nullptr
    );

    if (launchStatus != CUDA_SUCCESS)
    {
        error = "cuLaunchKernel(assembleEulerDdt) failed: "
            + cudaDriverErrorString(launchStatus);
        return tmp<fvMatrix<Type>>();
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
        launchStatus = cuStreamSynchronize
        (
            reinterpret_cast<CUstream>(ctx.stream())
        );

        if (launchStatus != CUDA_SUCCESS)
        {
            error = "cuStreamSynchronize failed: "
              + cudaDriverErrorString(launchStatus);
            return tmp<fvMatrix<Type>>();
        }
    }

    if
    (
        cudaMemcpy
        (
            diag.begin(),
            diagPtr,
            diagBytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    )
    {
        error = "cudaMemcpy(diag device->host) failed";
        return tmp<fvMatrix<Type>>();
    }

    scalar* sourceHost = reinterpret_cast<scalar*>(source.begin());
    if
    (
        cudaMemcpy
        (
            sourceHost,
            sourcePtr,
            sourceBytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    )
    {
        error = "cudaMemcpy(source device->host) failed";
        return tmp<fvMatrix<Type>>();
    }

    usedGpu = true;
    error.clear();
    return tfvm;
#endif
}


template<class Type>
tmp<fvMatrix<Type>> laplacian
(
    Context& ctx,
    FieldRegistry& registry,
    const GeometricField<scalar, fvPatchField, volMesh>& gamma,
    GeometricField<Type, fvPatchField, volMesh>& vf,
    bool& usedGpu,
    word& error,
    OperationStats* stats
)
{
    (void)registry;
    usedGpu = false;
    error.clear();

#ifndef FOAM_USE_CUDA
    error = "CUDA support not available";
    return tmp<fvMatrix<Type>>();
#else
    const fvMesh& mesh = vf.mesh();
    const word lapName("laplacian(" + gamma.name() + ',' + vf.name() + ')');

    ITstream& schemeStream = mesh.schemes().laplacian(lapName);
    word schemeType(schemeStream);
    if (schemeType != "Gauss")
    {
        error = "GPU laplacian supports Gauss schemes only (found "
          + schemeType + ')';
        return tmp<fvMatrix<Type>>();
    }

    word interpolationName("linear");
    if (!schemeStream.eof())
    {
        interpolationName = word(schemeStream);
    }

    word snGradName("corrected");
    if (!schemeStream.eof())
    {
        snGradName = word(schemeStream);
    }

    if (interpolationName != "linear")
    {
        error = "GPU laplacian supports Gauss linear only (found "
          + interpolationName + ')';
        return tmp<fvMatrix<Type>>();
    }

    if
    (
        snGradName != "corrected"
     && snGradName != "uncorrected"
    )
    {
        error = "GPU laplacian supports corrected/uncorrected snGrad only"
            " (found " + snGradName + ')';
        return tmp<fvMatrix<Type>>();
    }

    if (!ctx.ready())
    {
        if (!ctx.initialise(0, error))
        {
            return tmp<fvMatrix<Type>>();
        }
    }

    tmp<surfaceScalarField> tGammaFace = fvc::interpolate(gamma);
    const surfaceScalarField& gammaFace = tGammaFace();
    const surfaceScalarField& magSf = mesh.magSf();

    tmp<surfaceScalarField> tGammaMagSf(gammaFace*magSf);
    const surfaceScalarField& gammaMagSf = tGammaMagSf();

    const surfaceScalarField& deltaCoeffsRef =
        (snGradName == "corrected")
      ? mesh.nonOrthDeltaCoeffs()
      : mesh.deltaCoeffs();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            deltaCoeffsRef.dimensions()*gammaMagSf.dimensions()*vf.dimensions()
        )
    );
    fvMatrix<Type>& mat = tfvm.ref();

    scalarField& upper = mat.upper();

    const label nFaces = mesh.nInternalFaces();

    if (nFaces)
    {
        DeviceBuffer gammaDev;
        DeviceBuffer deltaDev;
        DeviceBuffer upperDev;

        const std::size_t bytes = sizeof(scalar)*nFaces;

        if
        (
            !gammaDev.allocate(bytes, error)
         || !deltaDev.allocate(bytes, error)
         || !upperDev.allocate(bytes, error)
        )
        {
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaMemcpy
            (
                gammaDev.get(),
                gammaMagSf.primitiveField().begin(),
                bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpy(gammaMagSf host->device) failed";
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaMemcpy
            (
                deltaDev.get(),
                deltaCoeffsRef.primitiveField().begin(),
                bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpy(deltaCoeffs host->device) failed";
            return tmp<fvMatrix<Type>>();
        }

        CUmodule module = nullptr;
        CUfunction function = nullptr;

        if (!compileGaussLaplacianKernel(module, function, error))
        {
            return tmp<fvMatrix<Type>>();
        }

        double* upperPtr = reinterpret_cast<double*>(upperDev.get());
        double* gammaPtr = reinterpret_cast<double*>(gammaDev.get());
        double* deltaPtr = reinterpret_cast<double*>(deltaDev.get());

        int faces = static_cast<int>(nFaces);
        void* args[] = { &upperPtr, &gammaPtr, &deltaPtr, &faces };

        cudaEvent_t startEvent = nullptr;
        cudaEvent_t stopEvent = nullptr;

        if (stats)
        {
            cudaEventCreateWithFlags(&startEvent, cudaEventDefault);
            cudaEventCreateWithFlags(&stopEvent, cudaEventDefault);
            cudaEventRecord(startEvent, ctx.stream());
        }

        CUresult launchStatus = cuLaunchKernel
        (
            function,
            (faces + 255)/256, 1, 1,
            256, 1, 1,
            0,
            reinterpret_cast<CUstream>(ctx.stream()),
            args,
            nullptr
        );

        if (launchStatus != CUDA_SUCCESS)
        {
            error = "cuLaunchKernel(assembleGaussLaplacian) failed: "
              + cudaDriverErrorString(launchStatus);
            return tmp<fvMatrix<Type>>();
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
            launchStatus = cuStreamSynchronize
            (
                reinterpret_cast<CUstream>(ctx.stream())
            );

            if (launchStatus != CUDA_SUCCESS)
            {
                error = "cuStreamSynchronize failed: "
                  + cudaDriverErrorString(launchStatus);
                return tmp<fvMatrix<Type>>();
            }
        }

        if
        (
            cudaMemcpy
            (
                upper.begin(),
                upperDev.get(),
                bytes,
                cudaMemcpyDeviceToHost
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpy(upper device->host) failed";
            return tmp<fvMatrix<Type>>();
        }
    }
    else if (stats)
    {
        stats->gpuMilliseconds = 0;
    }

    mat.negSumDiag();

    forAll(vf.boundaryField(), patchi)
    {
        const fvPatchField<Type>& pvf = vf.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDelta =
            deltaCoeffsRef.boundaryField()[patchi];

        if (pvf.coupled())
        {
            mat.internalCoeffs()[patchi] =
                pGamma*pvf.gradientInternalCoeffs(pDelta);
            mat.boundaryCoeffs()[patchi] =
               -pGamma*pvf.gradientBoundaryCoeffs(pDelta);
        }
        else
        {
            mat.internalCoeffs()[patchi] = pGamma*pvf.gradientInternalCoeffs();
            mat.boundaryCoeffs()[patchi] =
               -pGamma*pvf.gradientBoundaryCoeffs();
        }
    }

    const surfaceVectorField& Sf = mesh.Sf();
    const surfaceVectorField Sn = Sf/mesh.magSf();

    tmp<surfaceVectorField> tSfGamma(Sf*gammaFace);
    const surfaceVectorField& SfGamma = tSfGamma();

    tmp<surfaceScalarField> tSfGammaSn = SfGamma & Sn;
    const surfaceScalarField& SfGammaSn = tSfGammaSn();

    surfaceVectorField SfGammaCorr(SfGamma - Sn*SfGammaSn);

    tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> tFaceFluxCorrection =
        gammaSnGradCorrection(SfGammaCorr, vf);

    if (snGradName == "corrected")
    {
        tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> tSnGradCorr =
            correctedSnGradCorrection(vf);
        tFaceFluxCorrection.ref() += SfGammaSn*tSnGradCorr();
    }

    mat.source() -=
        mesh.V()
       *fvc::div(tFaceFluxCorrection())().primitiveField();

    if (mesh.schemes().fluxRequired(vf.name()))
    {
        mat.faceFluxCorrectionPtr() = tFaceFluxCorrection.ptr();
    }

    usedGpu = true;
    error.clear();
    return tfvm;
#endif
}


template<class Type>
tmp<fvMatrix<Type>> div
(
    Context& ctx,
    FieldRegistry& registry,
    const surfaceScalarField& faceFlux,
    GeometricField<Type, fvPatchField, volMesh>& vf,
    bool& usedGpu,
    word& error,
    OperationStats* stats
)
{
    usedGpu = false;
    error.clear();

#ifndef FOAM_USE_CUDA
    error = "CUDA support not available";
    return tmp<fvMatrix<Type>>();
#else
    const fvMesh& mesh = vf.mesh();
    const word divName("div(" + faceFlux.name() + ',' + vf.name() + ')');

    tmp<fv::convectionScheme<Type>> tScheme =
        fv::convectionScheme<Type>::New
        (
            mesh,
            faceFlux,
            mesh.schemes().div(divName)
        );

    fv::convectionScheme<Type>& scheme = tScheme.ref();

    const fv::gaussConvectionScheme<Type>* gaussScheme =
        dynamic_cast<fv::gaussConvectionScheme<Type>*>(&scheme);

    if (!gaussScheme)
    {
        error =
            "GPU div supports Gauss schemes only (found "
          + scheme.type() + ')';
        return tmp<fvMatrix<Type>>();
    }

    const surfaceInterpolationScheme<Type>& interp = gaussScheme->interpScheme();
    tmp<surfaceScalarField> tweights = interp.weights(vf);
    const surfaceScalarField& weights = tweights();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            faceFlux.dimensions()*vf.dimensions()
        )
    );
    fvMatrix<Type>& mat = tfvm.ref();

    scalarField& lower = mat.lower();
    scalarField& upper = mat.upper();

    const label nFaces = mesh.nInternalFaces();

    if (nFaces)
    {
        DeviceField<surfaceScalarField>& fluxDev =
            registry.getOrCreate(const_cast<surfaceScalarField&>(faceFlux));

        if (!fluxDev.syncHostToDevice(ctx, error))
        {
            return tmp<fvMatrix<Type>>();
        }

        DeviceBuffer lowerDev;
        DeviceBuffer upperDev;

        if
        (
            !lowerDev.allocate(sizeof(scalar)*nFaces, error)
         || !upperDev.allocate(sizeof(scalar)*nFaces, error)
        )
        {
            return tmp<fvMatrix<Type>>();
        }

        CUmodule module = nullptr;
        CUfunction function = nullptr;

        if (!compileGaussDivKernel(module, function, error))
        {
            return tmp<fvMatrix<Type>>();
        }

        double* lowerPtr = reinterpret_cast<double*>(lowerDev.get());
        double* upperPtr = reinterpret_cast<double*>(upperDev.get());
        double* fluxPtr = reinterpret_cast<double*>(fluxDev.devicePointer());

        if (!fluxPtr)
        {
            error = "GPU div: device pointer for face flux is null";
            return tmp<fvMatrix<Type>>();
        }

        int faces = static_cast<int>(nFaces);
        void* args[] = { &lowerPtr, &upperPtr, &fluxPtr, &faces };

        cudaEvent_t startEvent = nullptr;
        cudaEvent_t stopEvent = nullptr;

        if (stats)
        {
            cudaEventCreateWithFlags(&startEvent, cudaEventDefault);
            cudaEventCreateWithFlags(&stopEvent, cudaEventDefault);
            cudaEventRecord(startEvent, ctx.stream());
        }

        CUresult launchStatus = cuLaunchKernel
        (
            function,
            (faces + 255)/256, 1, 1,
            256, 1, 1,
            0,
            reinterpret_cast<CUstream>(ctx.stream()),
            args,
            nullptr
        );

        if (launchStatus != CUDA_SUCCESS)
        {
            error = "cuLaunchKernel(assembleGaussDiv) failed: "
              + cudaDriverErrorString(launchStatus);
            return tmp<fvMatrix<Type>>();
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
            launchStatus = cuStreamSynchronize
            (
                reinterpret_cast<CUstream>(ctx.stream())
            );

            if (launchStatus != CUDA_SUCCESS)
            {
                error = "cuStreamSynchronize failed: "
                  + cudaDriverErrorString(launchStatus);
                return tmp<fvMatrix<Type>>();
            }
        }

        if
        (
            cudaMemcpy
            (
                lower.begin(),
                lowerDev.get(),
                sizeof(scalar)*nFaces,
                cudaMemcpyDeviceToHost
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpy(lower device->host) failed";
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaMemcpy
            (
                upper.begin(),
                upperDev.get(),
                sizeof(scalar)*nFaces,
                cudaMemcpyDeviceToHost
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpy(upper device->host) failed";
            return tmp<fvMatrix<Type>>();
        }
    }
    else if (stats)
    {
        stats->gpuMilliseconds = 0;
    }

    mat.negSumDiag();

    forAll(vf.boundaryField(), patchi)
    {
        const fvPatchField<Type>& psf = vf.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = faceFlux.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];

        mat.internalCoeffs()[patchi] =
            patchFlux*psf.valueInternalCoeffs(pw);
        mat.boundaryCoeffs()[patchi] =
           -patchFlux*psf.valueBoundaryCoeffs(pw);
    }

    if (interp.corrected())
    {
        tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> tcorr =
            interp.correction(vf);

        mat += fvc::surfaceIntegrate(faceFlux*tcorr());
    }

    usedGpu = true;
    error.clear();
    return tfvm;
#endif
}

// Explicit instantiations for scalar and vector fields used in pimpleFoamGPU.
template tmp<fvMatrix<scalar>> ddt
(
    Context&,
    FieldRegistry&,
    GeometricField<scalar, fvPatchField, volMesh>&,
    bool&,
    word&,
    OperationStats*
);

template tmp<fvMatrix<vector>> ddt
(
    Context&,
    FieldRegistry&,
    GeometricField<vector, fvPatchField, volMesh>&,
    bool&,
    word&,
    OperationStats*
);

template tmp<fvMatrix<vector>> div
(
    Context&,
    FieldRegistry&,
    const surfaceScalarField&,
    GeometricField<vector, fvPatchField, volMesh>&,
    bool&,
    word&,
    OperationStats*
);

template tmp<fvMatrix<scalar>> laplacian
(
    Context&,
    FieldRegistry&,
    const GeometricField<scalar, fvPatchField, volMesh>&,
    GeometricField<scalar, fvPatchField, volMesh>&,
    bool&,
    word&,
    OperationStats*
);

} // namespace fvm
} // namespace gpu
} // namespace Foam
