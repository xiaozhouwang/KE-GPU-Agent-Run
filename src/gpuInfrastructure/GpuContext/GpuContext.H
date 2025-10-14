#ifndef Foam_gpu_GpuContext_H
#define Foam_gpu_GpuContext_H

#include "Switch.H"
#include "word.H"

#ifdef FOAM_USE_CUDA
    #include <cuda_runtime.h>
#else
    typedef void* cudaStream_t;
#endif

namespace Foam
{
namespace gpu
{

class Context
{
#ifdef FOAM_USE_CUDA
    bool initialised_;
    int deviceId_;
    cudaStream_t computeStream_;
    cudaStream_t transferStream_;
    cudaStream_t auxStream_;
    cudaDeviceProp deviceProps_;
    word archTag_;
#endif

public:

    Context();
    ~Context();

    //- Disable copy
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    //- Initialise the context on the requested device.
    //  Returns true on success and populates err on failure.
    bool initialise(int deviceId, word& errMessage);

    //- Destroy the context resources if initialised.
    void reset();

    //- True when the context owns a CUDA stream and device.
    bool ready() const;

    //- Return the active device id (or -1 if not initialised).
    int deviceId() const;

    //- Return the default compute stream (nullptr if not initialised or CUDA disabled).
    cudaStream_t stream() const;

    //- Return the compute stream used for kernel launches.
    cudaStream_t computeStream() const;

    //- Stream dedicated to host<->device transfers.
    cudaStream_t transferStream() const;

    //- Auxiliary stream for optional concurrent kernels.
    cudaStream_t auxStream() const;

    //- Access cached device properties (undefined if not initialised).
    const cudaDeviceProp& deviceProperties() const;

    //- Architecture tag in the form compute_XY (empty if not initialised).
    const word& architectureTag() const;

    //- Major compute capability (returns -1 if not initialised).
    int capabilityMajor() const;

    //- Minor compute capability (returns -1 if not initialised).
    int capabilityMinor() const;
};

//- Return true when CUDA support is enabled at compile time and the runtime
//  initialisation succeeded.
bool available();

//- Obtain a global singleton context. The first call initialises it on device 0.
Context& globalContext();

} // namespace gpu
} // namespace Foam

#endif
