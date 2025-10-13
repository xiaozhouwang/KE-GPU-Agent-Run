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
    stream_(nullptr)
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

    status = ::cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess)
    {
        errMessage = "cudaStreamCreateWithFlags failed";
        stream_ = nullptr;
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
        if (stream_)
        {
            ::cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        initialised_ = false;
        deviceId_ = -1;
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
    return stream_;
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
