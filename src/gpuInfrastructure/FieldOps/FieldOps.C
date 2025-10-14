#include "FieldOps.H"
#include "error.H"
#include "KernelCache.H"
#include "OSspecific.H"

#include <cctype>
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

const std::string& surfaceSubtractSource()
{
    static const std::string src = R"(
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
    return src;
}


const std::string& velocityCorrectorAoSSource()
{
    static const std::string src = R"(
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
    return src;
}


const std::string& velocityCorrectorSoASource()
{
    static const std::string src = R"(
extern "C" __global__
void velocityCorrectorSoA
(
    double* u,
    const double* hx,
    const double* hy,
    const double* hz,
    const double* gx,
    const double* gy,
    const double* gz,
    const double* r,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        double rVal = r[idx];
        int base = idx * 3;
        u[base]     = hx[idx] - rVal * gx[idx];
        u[base + 1] = hy[idx] - rVal * gy[idx];
        u[base + 2] = hz[idx] - rVal * gz[idx];
    }
}
)";
    return src;
}


const std::string& computeNutKernelSource()
{
    static const std::string src = R"(
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
    return src;
}


bool preferVectorSoA()
{
    static const bool prefer = []
    {
        word value = getEnv("FOAM_GPU_PREFER_SOA");
        for (char& c : value)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return value == "1"
            || value == "true"
            || value == "on"
            || value == "yes";
    }();
    return prefer;
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

    CompiledKernel kernel;
    static const word kernelKey("surfaceSubtract");
    if
    (
        !KernelCache::instance().getOrCompile
        (
            ctx,
            kernelKey,
            surfaceSubtractSource(),
            {},
            "surfaceSubtract",
            kernel,
            error
        )
    )
    {
        return false;
    }

    scalar elapsedMs = 0;

    if
    (
        !GraphLaunchCache::instance().launch
        (
            ctx,
            kernelKey,
            kernel,
            StreamCategory::aux,
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
        return false;
    }

    if (stats)
    {
        stats->gpuMilliseconds = elapsedMs;
    }

    dst.markDeviceDirty();
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

    scalar elapsedMs = 0;

    const bool useSoA =
        preferVectorSoA()
     && HbyA.supportsSoA()
     && gradP.supportsSoA();

    CompiledKernel kernel;

    if (useSoA)
    {
        DeviceField<volVectorField>& hMutable =
            const_cast<DeviceField<volVectorField>&>(HbyA);
        DeviceField<volVectorField>& gradMutable =
            const_cast<DeviceField<volVectorField>&>(gradP);

        if
        (
            !hMutable.ensureSoALayout(ctx, error)
         || !gradMutable.ensureSoALayout(ctx, error)
        )
        {
            return false;
        }

        double* hSoA =
            const_cast<double*>
            (
                reinterpret_cast<const double*>(hMutable.soaPointer())
            );
        double* gradSoA =
            const_cast<double*>
            (
                reinterpret_cast<const double*>(gradMutable.soaPointer())
            );

        if (!hSoA || !gradSoA)
        {
            error = "velocityCorrector: SoA pointers are null";
            return false;
        }

        double* hx = hSoA;
        double* hy = hSoA + n;
        double* hz = hSoA + n*2;

        double* gx = gradSoA;
        double* gy = gradSoA + n;
        double* gz = gradSoA + n*2;

        void* args[] =
        {
            &uPtr,
            &hx,
            &hy,
            &hz,
            &gx,
            &gy,
            &gz,
            &rPtr,
            &nCells
        };

        static const word kernelKey("velocityCorrectorSoA");

        if
        (
            !KernelCache::instance().getOrCompile
            (
                ctx,
                kernelKey,
                velocityCorrectorSoASource(),
                {},
                "velocityCorrectorSoA",
                kernel,
                error
            )
        )
        {
            return false;
        }

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
            return false;
        }
    }
    else
    {
        void* args[] = { &uPtr, &hPtr, &gradPtr, &rPtr, &nCells };

        static const word kernelKey("velocityCorrectorAoS");

        if
        (
            !KernelCache::instance().getOrCompile
            (
                ctx,
                kernelKey,
                velocityCorrectorAoSSource(),
                {},
                "velocityCorrector",
                kernel,
                error
            )
        )
        {
            return false;
        }

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
            return false;
        }
    }

    if (stats)
    {
        stats->gpuMilliseconds = elapsedMs;
    }

    U.markDeviceDirty();
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

    static const word kernelKey("computeNut");
    CompiledKernel kernel;

    if
    (
        !KernelCache::instance().getOrCompile
        (
            ctx,
            kernelKey,
            computeNutKernelSource(),
            {},
            "computeNut",
            kernel,
            error
        )
    )
    {
        return false;
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
            dim3((cells + 255)/256, 1, 1),
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
        return false;
    }

    if (stats)
    {
        stats->gpuMilliseconds = elapsedMs;
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
