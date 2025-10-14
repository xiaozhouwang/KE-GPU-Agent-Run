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
#include "localEulerDdtScheme.H"
#include "fvmLaplacian.H"
#include "fvMesh.H"
#include "fvMatrices.H"
#include "Time.H"
#include "pTraits.H"
#include "KernelCache.H"

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
namespace fvm
{

#ifdef FOAM_USE_CUDA
namespace
{

word makeEulerDdtKey(const int components)
{
    return word("assembleEulerDdt_" + Foam::name(components));
}


std::string makeEulerDdtSource(const int components)
{
    std::ostringstream source;
    source
        << "#define COMPONENTS " << components << '\n'
        << "extern \"C\" __global__\n"
        << "void assembleEulerDdt(\n"
        << "    double* diag,\n"
        << "    double* source,\n"
        << "    const double* volumeDiag,\n"
        << "    const double* volumeSource,\n"
        << "    const double* rDeltaT,\n"
        << "    const double* fieldOld,\n"
        << "    const int nCells)\n"
        << "{\n"
        << "    const int cell = blockIdx.x * blockDim.x + threadIdx.x;\n"
        << "    if (cell >= nCells)\n"
        << "    {\n"
        << "        return;\n"
        << "    }\n"
        << "    const double rDeltaTCell = rDeltaT[cell];\n"
        << "    const double diagVal = rDeltaTCell * volumeDiag[cell];\n"
        << "    diag[cell] = diagVal;\n"
        << "    const double coeff = rDeltaTCell * volumeSource[cell];\n"
        << "    const double* uOld = fieldOld + cell*COMPONENTS;\n"
        << "    double* src = source + cell*COMPONENTS;\n"
        << "    #pragma unroll\n"
        << "    for (int c = 0; c < COMPONENTS; ++c)\n"
        << "    {\n"
        << "        src[c] = coeff * uOld[c];\n"
        << "    }\n"
        << "}\n";
    return source.str();
}


const std::string& gaussDivSource()
{
    static const std::string src = R"(
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
    return src;
}


const std::string& gaussLaplacianSource()
{
    static const std::string src = R"(
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
    return src;
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

    // Ensure the old-time field values are uploaded for this step since
    // OpenFOAM rotates them on the host each time step.
    vfOldDevice.markHostDirty();

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

    const scalarField& volumeDiag = mesh.Vsc();
    const scalarField& volumeSource =
        mesh.moving() ? mesh.Vsc0() : mesh.Vsc();

    scalarField rDeltaTHost
    (
        volumeDiag.size(),
        1.0/mesh.time().deltaTValue()
    );

    if (fv::localEulerDdt::enabled(mesh))
    {
        rDeltaTHost = fv::localEulerDdt::localRDeltaT(mesh).primitiveField();
    }

    DeviceBuffer diagDev;
    DeviceBuffer sourceDev;
    DeviceBuffer volDiagDev;
    DeviceBuffer volSrcDev;
    DeviceBuffer rDeltaTDev;

    const std::size_t diagBytes = sizeof(scalar)*nCells;
    const std::size_t nComp = pTraits<Type>::nComponents;
    const std::size_t sourceBytes = sizeof(scalar)*nCells*nComp;

    if
    (
        !diagDev.allocate(diagBytes, error)
     || !sourceDev.allocate(sourceBytes, error)
     || !volDiagDev.allocate(diagBytes, error)
     || !volSrcDev.allocate(diagBytes, error)
     || !rDeltaTDev.allocate(diagBytes, error)
    )
    {
        return tmp<fvMatrix<Type>>();
    }

    if
    (
        cudaMemcpyAsync
        (
            volDiagDev.as<scalar>(),
            volumeDiag.begin(),
            diagBytes,
            cudaMemcpyHostToDevice,
            ctx.transferStream()
        ) != cudaSuccess
    )
    {
        error = "cudaMemcpyAsync(volDiag host->device) failed";
        return tmp<fvMatrix<Type>>();
    }

    if
    (
        cudaMemcpyAsync
        (
            volSrcDev.as<scalar>(),
            volumeSource.begin(),
            diagBytes,
            cudaMemcpyHostToDevice,
            ctx.transferStream()
        ) != cudaSuccess
    )
    {
        error = "cudaMemcpyAsync(volSrc host->device) failed";
        return tmp<fvMatrix<Type>>();
    }

    if
    (
        cudaMemcpyAsync
        (
            rDeltaTDev.as<scalar>(),
            rDeltaTHost.begin(),
            diagBytes,
            cudaMemcpyHostToDevice,
            ctx.transferStream()
        ) != cudaSuccess
    )
    {
        error = "cudaMemcpyAsync(rDeltaT host->device) failed";
        return tmp<fvMatrix<Type>>();
    }

    if
    (
        cudaStreamSynchronize(ctx.transferStream()) != cudaSuccess
    )
    {
        error = "cudaStreamSynchronize(volCoeff host->device) failed";
        return tmp<fvMatrix<Type>>();
    }

    scalar* diagPtr = reinterpret_cast<scalar*>(diagDev.get());
    scalar* sourcePtr = reinterpret_cast<scalar*>(sourceDev.get());
    scalar* volDiagPtr = reinterpret_cast<scalar*>(volDiagDev.get());
    scalar* volSrcPtr = reinterpret_cast<scalar*>(volSrcDev.get());
    scalar* rDeltaTPtr = reinterpret_cast<scalar*>(rDeltaTDev.get());

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

    int cells = static_cast<int>(nCells);

    void* args[] =
    {
        &diagPtr,
        &sourcePtr,
        &volDiagPtr,
        &volSrcPtr,
        &rDeltaTPtr,
        &vfOldPtr,
        &cells
    };

    const int components = static_cast<int>(nComp);
    const word kernelKey = makeEulerDdtKey(components);
    const std::string kernelSource = makeEulerDdtSource(components);

    CompiledKernel kernel;
    if
    (
        !KernelCache::instance().getOrCompile
        (
            ctx,
            kernelKey,
            kernelSource,
            {},
            "assembleEulerDdt",
            kernel,
            error
        )
    )
    {
        return tmp<fvMatrix<Type>>();
    }

    scalar elapsedMs = 0;

    if
    (
        !GraphLaunchCache::instance().launch
        (
            ctx,
            kernelKey,
            kernel,
            StreamCategory::compute,
            dim3(gridSize, 1, 1),
            dim3(blockSize, 1, 1),
            args,
            sizeof(args)/sizeof(args[0]),
            0,
            error,
            stats != nullptr,
            elapsedMs
        )
    )
    {
        return tmp<fvMatrix<Type>>();
    }

    if (stats)
    {
        stats->gpuMilliseconds = elapsedMs;
    }

    cudaEvent_t kernelDone = nullptr;
    if (cudaEventCreateWithFlags(&kernelDone, cudaEventDisableTiming) != cudaSuccess)
    {
        error = "cudaEventCreate(kernelDone) failed";
        return tmp<fvMatrix<Type>>();
    }

    if (cudaEventRecord(kernelDone, ctx.computeStream()) != cudaSuccess)
    {
        cudaEventDestroy(kernelDone);
        error = "cudaEventRecord(kernelDone) failed";
        return tmp<fvMatrix<Type>>();
    }

    if (cudaStreamWaitEvent(ctx.transferStream(), kernelDone, 0) != cudaSuccess)
    {
        cudaEventDestroy(kernelDone);
        error = "cudaStreamWaitEvent(kernelDone) failed";
        return tmp<fvMatrix<Type>>();
    }

    bool copyOk = true;

    if
    (
        cudaMemcpyAsync
        (
            diag.begin(),
            diagPtr,
            diagBytes,
            cudaMemcpyDeviceToHost,
            ctx.transferStream()
        ) != cudaSuccess
    )
    {
        error = "cudaMemcpyAsync(diag device->host) failed";
        copyOk = false;
    }

    scalar* sourceHost = reinterpret_cast<scalar*>(source.begin());
    if (copyOk)
    {
        if
        (
            cudaMemcpyAsync
            (
                sourceHost,
                sourcePtr,
                sourceBytes,
                cudaMemcpyDeviceToHost,
                ctx.transferStream()
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpyAsync(source device->host) failed";
            copyOk = false;
        }
    }

    cudaEventDestroy(kernelDone);

    if (!copyOk)
    {
        return tmp<fvMatrix<Type>>();
    }

    if
    (
        cudaStreamSynchronize(ctx.transferStream()) != cudaSuccess
    )
    {
        error = "cudaStreamSynchronize(ddt copies) failed";
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

    bool copiedUpper = false;

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
            cudaMemcpyAsync
            (
                gammaDev.get(),
                gammaMagSf.primitiveField().begin(),
                bytes,
                cudaMemcpyHostToDevice,
                ctx.transferStream()
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpyAsync(gammaMagSf host->device) failed";
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaMemcpyAsync
            (
                deltaDev.get(),
                deltaCoeffsRef.primitiveField().begin(),
                bytes,
                cudaMemcpyHostToDevice,
                ctx.transferStream()
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpyAsync(deltaCoeffs host->device) failed";
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaStreamSynchronize(ctx.transferStream()) != cudaSuccess
        )
        {
            error = "cudaStreamSynchronize(Laplacian coeff copies) failed";
            return tmp<fvMatrix<Type>>();
        }

        double* upperPtr = reinterpret_cast<double*>(upperDev.get());
        double* gammaPtr = reinterpret_cast<double*>(gammaDev.get());
        double* deltaPtr = reinterpret_cast<double*>(deltaDev.get());

        int faces = static_cast<int>(nFaces);
        void* args[] = { &upperPtr, &gammaPtr, &deltaPtr, &faces };

        static const word kernelKey("assembleGaussLaplacian");

        CompiledKernel kernel;
        if
        (
            !KernelCache::instance().getOrCompile
            (
                ctx,
                kernelKey,
                gaussLaplacianSource(),
                {},
                "assembleGaussLaplacian",
                kernel,
                error
            )
        )
        {
            return tmp<fvMatrix<Type>>();
        }

        scalar elapsedMs = 0;

        if
        (
            !GraphLaunchCache::instance().launch
            (
                ctx,
                kernelKey,
                kernel,
                StreamCategory::compute,
                dim3((faces + 255)/256, 1, 1),
                dim3(256, 1, 1),
                args,
                sizeof(args)/sizeof(args[0]),
                0,
                error,
                stats != nullptr,
                elapsedMs
            )
        )
        {
            return tmp<fvMatrix<Type>>();
        }

        if (stats)
        {
            stats->gpuMilliseconds = elapsedMs;
        }

        cudaEvent_t kernelDone = nullptr;
        if (cudaEventCreateWithFlags(&kernelDone, cudaEventDisableTiming) != cudaSuccess)
        {
            error = "cudaEventCreate(laplacian kernel) failed";
            return tmp<fvMatrix<Type>>();
        }

        if (cudaEventRecord(kernelDone, ctx.computeStream()) != cudaSuccess)
        {
            cudaEventDestroy(kernelDone);
            error = "cudaEventRecord(laplacian kernel) failed";
            return tmp<fvMatrix<Type>>();
        }

        if (cudaStreamWaitEvent(ctx.transferStream(), kernelDone, 0) != cudaSuccess)
        {
            cudaEventDestroy(kernelDone);
            error = "cudaStreamWaitEvent(laplacian kernel) failed";
            return tmp<fvMatrix<Type>>();
        }

        bool copyOk = true;

        if
        (
            cudaMemcpyAsync
            (
                upper.begin(),
                upperDev.get(),
                bytes,
                cudaMemcpyDeviceToHost,
                ctx.transferStream()
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpyAsync(upper device->host) failed";
            copyOk = false;
        }

        if (!copyOk)
        {
            cudaEventDestroy(kernelDone);
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaStreamSynchronize(ctx.transferStream()) != cudaSuccess
        )
        {
            cudaEventDestroy(kernelDone);
            error = "cudaStreamSynchronize(laplacian copies) failed";
            return tmp<fvMatrix<Type>>();
        }

        cudaEventDestroy(kernelDone);

        copiedUpper = true;
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

    if (stats)
    {
        tmp<fvMatrix<Type>> tCpuLapCheck = Foam::fvm::laplacian(gamma, vf);
        const scalarField& gpuSrc = mat.source();
        const scalarField& cpuSrc = tCpuLapCheck().source();
        scalar diffSq = 0.0;
        scalar refSq = 0.0;
        forAll(gpuSrc, celli)
        {
            const scalar diff = gpuSrc[celli] - cpuSrc[celli];
            const scalar w = mesh.V()[celli];
            diffSq += w*diff*diff;
            refSq += w*cpuSrc[celli]*cpuSrc[celli];
        }
        refSq = max(refSq, VSMALL);
        Info<< "GPU laplacian source mismatch (pre-solve): "
            << Foam::sqrt(diffSq/refSq) << nl;
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

    bool copiedCoeffs = false;

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

        static const word kernelKey("assembleGaussDiv");

        CompiledKernel kernel;
        if
        (
            !KernelCache::instance().getOrCompile
            (
                ctx,
                kernelKey,
                gaussDivSource(),
                {},
                "assembleGaussDiv",
                kernel,
                error
            )
        )
        {
            return tmp<fvMatrix<Type>>();
        }

        scalar elapsedMs = 0;

        if
        (
            !GraphLaunchCache::instance().launch
            (
                ctx,
                kernelKey,
                kernel,
                StreamCategory::compute,
                dim3((faces + 255)/256, 1, 1),
                dim3(256, 1, 1),
                args,
                sizeof(args)/sizeof(args[0]),
                0,
                error,
                stats != nullptr,
                elapsedMs
            )
        )
        {
            return tmp<fvMatrix<Type>>();
        }

        if (stats)
        {
            stats->gpuMilliseconds = elapsedMs;
        }

        cudaEvent_t kernelDone = nullptr;
        if (cudaEventCreateWithFlags(&kernelDone, cudaEventDisableTiming) != cudaSuccess)
        {
            error = "cudaEventCreate(div kernel) failed";
            return tmp<fvMatrix<Type>>();
        }

        if (cudaEventRecord(kernelDone, ctx.computeStream()) != cudaSuccess)
        {
            cudaEventDestroy(kernelDone);
            error = "cudaEventRecord(div kernel) failed";
            return tmp<fvMatrix<Type>>();
        }

        if (cudaStreamWaitEvent(ctx.transferStream(), kernelDone, 0) != cudaSuccess)
        {
            cudaEventDestroy(kernelDone);
            error = "cudaStreamWaitEvent(div kernel) failed";
            return tmp<fvMatrix<Type>>();
        }

        const std::size_t coeffBytes = sizeof(scalar)*nFaces;
        bool copyOk = true;

        if
        (
            cudaMemcpyAsync
            (
                lower.begin(),
                lowerDev.get(),
                coeffBytes,
                cudaMemcpyDeviceToHost,
                ctx.transferStream()
            ) != cudaSuccess
        )
        {
            error = "cudaMemcpyAsync(lower device->host) failed";
            copyOk = false;
        }

        if (copyOk)
        {
            if
            (
                cudaMemcpyAsync
                (
                    upper.begin(),
                    upperDev.get(),
                    coeffBytes,
                    cudaMemcpyDeviceToHost,
                    ctx.transferStream()
                ) != cudaSuccess
            )
            {
                error = "cudaMemcpyAsync(upper device->host) failed";
                copyOk = false;
            }
        }

        if (!copyOk)
        {
            cudaEventDestroy(kernelDone);
            return tmp<fvMatrix<Type>>();
        }

        if
        (
            cudaStreamSynchronize(ctx.transferStream()) != cudaSuccess
        )
        {
            cudaEventDestroy(kernelDone);
            error = "cudaStreamSynchronize(div copies) failed";
            return tmp<fvMatrix<Type>>();
        }

        cudaEventDestroy(kernelDone);

        copiedCoeffs = true;
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
