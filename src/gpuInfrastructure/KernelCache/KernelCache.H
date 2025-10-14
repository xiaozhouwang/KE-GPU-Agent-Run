#ifndef Foam_gpu_KernelCache_H
#define Foam_gpu_KernelCache_H

#include "GpuContext.H"
#include "scalar.H"
#include "word.H"

#include <string>
#include <vector>

#ifdef FOAM_USE_CUDA
    #include <cuda.h>
    #include <nvrtc.h>
#else
    typedef void* CUmodule;
    typedef void* CUfunction;
    typedef void* CUgraph;
    typedef void* CUgraphExec;
    typedef void* CUgraphNode;
#endif

namespace Foam
{
namespace gpu
{

enum class StreamCategory
{
    compute,
    aux
};

struct CompiledKernel
{
#ifdef FOAM_USE_CUDA
    CUmodule module{nullptr};
    CUfunction function{nullptr};
#else
    CUmodule module{nullptr};
    CUfunction function{nullptr};
#endif
};

class KernelCache
{
public:
    static KernelCache& instance();

    bool getOrCompile
    (
        Context& ctx,
        const word& lookupKey,
        const std::string& source,
        const std::vector<std::string>& extraOptions,
        const char* kernelName,
        CompiledKernel& kernel,
        word& error
    );

    void clear();

private:
    KernelCache() = default;
    ~KernelCache();
    KernelCache(const KernelCache&) = delete;
    KernelCache& operator=(const KernelCache&) = delete;
};

class GraphLaunchCache
{
public:
    static GraphLaunchCache& instance();

    bool launch
    (
        Context& ctx,
        const word& kernelKey,
        const CompiledKernel& kernel,
        StreamCategory streamCategory,
#ifdef FOAM_USE_CUDA
        dim3 grid,
        dim3 block,
#else
        int grid,
        int block,
#endif
        void** args,
        std::size_t numArgs,
        std::size_t sharedMemBytes,
        word& error,
        bool captureStats,
        scalar& elapsedMs
    );

    void clear();

private:
    GraphLaunchCache() = default;
    ~GraphLaunchCache();
    GraphLaunchCache(const GraphLaunchCache&) = delete;
    GraphLaunchCache& operator=(const GraphLaunchCache&) = delete;
};

#ifdef FOAM_USE_CUDA
bool graphsSupported();
void setCudaGraphsRequested(bool enabled);
bool graphsEnabled();
std::string nvrtcErrorString(const nvrtcResult code);
std::string cudaDriverErrorString(const CUresult code);
#endif

} // namespace gpu
} // namespace Foam

#endif
