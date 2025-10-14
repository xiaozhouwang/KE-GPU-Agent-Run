#include "GpuContext.H"
#include "error.H"

#ifdef FOAM_USE_CUDA
    #include <cuda_runtime_api.h>
#endif

namespace Foam
{
namespace gpu
{

Context::Context()
#ifdef FOAM_USE_CUDA
:
    initialised_(false),
    deviceId_(-1),
    computeStream_(nullptr),
    transferStream_(nullptr),
    auxStream_(nullptr),
    deviceProps_(),
    archTag_()
#endif
{}


Context::~Context()
{
    reset();
}


bool Context::initialise(int deviceId, word& errMessage)
{
#ifdef FOAM_USE_CUDA
    if (initialised_)
    {
        return true;
    }

    int count = 0;
    cudaError_t status = ::cudaGetDeviceCount(&count);

    if (status != cudaSuccess || count == 0)
    {
        errMessage = "cudaGetDeviceCount failed";
        return false;
    }

    if (deviceId < 0 || deviceId >= count)
    {
        deviceId = 0;
    }

    status = ::cudaSetDevice(deviceId);
    if (status != cudaSuccess)
    {
        errMessage = "cudaSetDevice failed";
        return false;
    }

    status = ::cudaGetDeviceProperties(&deviceProps_, deviceId);
    if (status != cudaSuccess)
    {
        errMessage = "cudaGetDeviceProperties failed";
        return false;
    }

    archTag_ =
        word
        (
            "compute_"
          + Foam::name(deviceProps_.major)
          + Foam::name(deviceProps_.minor)
        );

    status = ::cudaStreamCreateWithFlags(&computeStream_, cudaStreamNonBlocking);
    if (status != cudaSuccess)
    {
        errMessage = "cudaStreamCreateWithFlags (compute) failed";
        computeStream_ = nullptr;
        return false;
    }

    status = ::cudaStreamCreateWithFlags(&transferStream_, cudaStreamNonBlocking);
    if (status != cudaSuccess)
    {
        errMessage = "cudaStreamCreateWithFlags (transfer) failed";
        ::cudaStreamDestroy(computeStream_);
        computeStream_ = nullptr;
        transferStream_ = nullptr;
        return false;
    }

    status = ::cudaStreamCreateWithFlags(&auxStream_, cudaStreamNonBlocking);
    if (status != cudaSuccess)
    {
        errMessage = "cudaStreamCreateWithFlags (aux) failed";
        ::cudaStreamDestroy(computeStream_);
        ::cudaStreamDestroy(transferStream_);
        computeStream_ = nullptr;
        transferStream_ = nullptr;
        auxStream_ = nullptr;
        return false;
    }

    initialised_ = true;
    deviceId_ = deviceId;
    errMessage.clear();
    return true;
#else
    (void)deviceId;
    errMessage = "CUDA support not available in this build";
    return false;
#endif
}


void Context::reset()
{
#ifdef FOAM_USE_CUDA
    if (initialised_)
    {
        if (computeStream_)
        {
            ::cudaStreamDestroy(computeStream_);
            computeStream_ = nullptr;
        }
        if (transferStream_)
        {
            ::cudaStreamDestroy(transferStream_);
            transferStream_ = nullptr;
        }
        if (auxStream_)
        {
            ::cudaStreamDestroy(auxStream_);
            auxStream_ = nullptr;
        }
        initialised_ = false;
        deviceId_ = -1;
        archTag_.clear();
        deviceProps_ = cudaDeviceProp{};
    }
#endif
}


bool Context::ready() const
{
#ifdef FOAM_USE_CUDA
    return initialised_;
#else
    return false;
#endif
}


int Context::deviceId() const
{
#ifdef FOAM_USE_CUDA
    return deviceId_;
#else
    return -1;
#endif
}


cudaStream_t Context::stream() const
{
#ifdef FOAM_USE_CUDA
    return computeStream_;
#else
    return nullptr;
#endif
}


cudaStream_t Context::computeStream() const
{
#ifdef FOAM_USE_CUDA
    return computeStream_;
#else
    return nullptr;
#endif
}


cudaStream_t Context::transferStream() const
{
#ifdef FOAM_USE_CUDA
    return transferStream_;
#else
    return nullptr;
#endif
}


cudaStream_t Context::auxStream() const
{
#ifdef FOAM_USE_CUDA
    return auxStream_;
#else
    return nullptr;
#endif
}


const cudaDeviceProp& Context::deviceProperties() const
{
#ifdef FOAM_USE_CUDA
    return deviceProps_;
#else
    static cudaDeviceProp dummy{};
    return dummy;
#endif
}


const word& Context::architectureTag() const
{
#ifdef FOAM_USE_CUDA
    return archTag_;
#else
    static const word empty;
    return empty;
#endif
}


int Context::capabilityMajor() const
{
#ifdef FOAM_USE_CUDA
    return initialised_ ? deviceProps_.major : -1;
#else
    return -1;
#endif
}


int Context::capabilityMinor() const
{
#ifdef FOAM_USE_CUDA
    return initialised_ ? deviceProps_.minor : -1;
#else
    return nullptr;
#endif
}


bool available()
{
    return globalContext().ready();
}


Context& globalContext()
{
    static Context ctx;
    if (!ctx.ready())
    {
        word err;
        ctx.initialise(0, err);
        if (!err.empty())
        {
            WarningIn("Foam::gpu::globalContext") << err << nl;
        }
    }
    return ctx;
}

} // namespace gpu
} // namespace Foam
