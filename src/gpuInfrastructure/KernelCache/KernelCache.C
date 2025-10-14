#include "KernelCache.H"
#include "error.H"
#include "OSspecific.H"
#include "fileName.H"
#include "SHA1.H"

#include <array>
#include <atomic>
#include <cctype>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <vector>

#ifdef FOAM_USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
#endif

#if defined(FOAM_USE_CUDA)
    #if CUDA_VERSION >= 10020
        #define FOAM_GPU_CUDA_GRAPHS_AVAILABLE 1
    #else
        #define FOAM_GPU_CUDA_GRAPHS_AVAILABLE 0
    #endif
#else
    #define FOAM_GPU_CUDA_GRAPHS_AVAILABLE 0
#endif

namespace Foam
{
namespace gpu
{

#ifdef FOAM_USE_CUDA
namespace
{

struct KernelCacheEntry
{
    CUmodule module{nullptr};
    CUfunction function{nullptr};
    std::string ptx;
};

#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
struct GraphEntry
{
    CUgraph graph{nullptr};
    CUgraphExec exec{nullptr};
    CUgraphNode node{nullptr};
    dim3 grid{};
    dim3 block{};
    std::size_t sharedMem{0};
    CUfunction function{nullptr};
    std::vector<std::array<unsigned char, sizeof(void*)>> paramBuffers;
    std::vector<void*> initialParams;
};
#endif

const char* ptxExtension()
{
    return ".ptx";
}

bool envSwitchOn(const std::string& value)
{
    if (value.empty())
    {
        return false;
    }

    std::string lowered(value);
    for (char& c : lowered)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    return lowered == "1"
        || lowered == "true"
        || lowered == "on"
        || lowered == "yes";
}

Foam::fileName kernelCacheRoot()
{
    static const Foam::fileName dir = []
    {
        const std::string explicitDir = Foam::getEnv("FOAM_GPU_CACHE_DIR");
        if (!explicitDir.empty())
        {
            Foam::fileName path(explicitDir);
            Foam::mkDir(path);
            return path;
        }

        const std::string xdg = Foam::getEnv("XDG_CACHE_HOME");
        if (!xdg.empty())
        {
            Foam::fileName base(xdg);
            Foam::mkDir(base);

            Foam::fileName ofDir(base/"openfoam");
            Foam::mkDir(ofDir);

            Foam::fileName nvrtcDir(ofDir/"nvrtc");
            Foam::mkDir(nvrtcDir);
            return nvrtcDir;
        }

        Foam::fileName homeDir = Foam::home();
        if (homeDir.empty())
        {
            return Foam::fileName::null;
        }

        Foam::fileName cacheDir(homeDir/".cache");
        Foam::mkDir(cacheDir);

        Foam::fileName ofDir(cacheDir/"openfoam");
        Foam::mkDir(ofDir);

        Foam::fileName nvrtcDir(ofDir/"nvrtc");
        Foam::mkDir(nvrtcDir);
        return nvrtcDir;
    }();

    return dir;
}

Foam::fileName cacheDirectoryForArch(const word& archTag)
{
    const Foam::fileName root = kernelCacheRoot();
    if (root.empty())
    {
        return Foam::fileName::null;
    }

    Foam::fileName dir(root/Foam::fileName(archTag));
    Foam::mkDir(dir);
    return dir;
}

std::string makeKernelCacheKey
(
    const word& archTag,
    const word& kernelKey,
    const std::string& source,
    const std::vector<std::string>& options
)
{
    std::ostringstream os;
    os << archTag << '|' << kernelKey << '|'
       << std::hash<std::string>{}(source);
    for (const auto& opt : options)
    {
        os << '|' << opt;
    }
    return os.str();
}

#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
std::string makeGraphCacheKey
(
    const word& archTag,
    const word& kernelKey,
    const dim3 grid,
    const dim3 block,
    const std::size_t sharedMem
)
{
    std::ostringstream os;
    os << archTag << '|' << kernelKey << '|'
       << grid.x << ',' << grid.y << ',' << grid.z << '|'
       << block.x << ',' << block.y << ',' << block.z << '|'
       << sharedMem;
    return os.str();
}
#endif

std::string makeKernelDiskTag
(
    const word& lookupKey,
    const std::string& source,
    const std::vector<std::string>& options
)
{
    Foam::SHA1 sha;
    sha.append(lookupKey);
    sha.append(source);

    for (const auto& opt : options)
    {
        sha.append(opt);
    }

    sha.append(std::string("cuda:") + std::to_string(CUDA_VERSION));

#ifdef NVRTC_VERSION
    sha.append(std::string("nvrtc:") + std::to_string(NVRTC_VERSION));
#endif

    return sha.digest().str(false);
}

bool readFileContents(const Foam::fileName& path, std::string& data)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in)
    {
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    if (end < 0)
    {
        return false;
    }

    const std::size_t size = static_cast<std::size_t>(end);
    data.resize(size);

    in.seekg(0, std::ios::beg);
    if (size)
    {
        in.read(&data[0], static_cast<std::streamsize>(size));
    }

    return in.good();
}

bool writeFileContents(const Foam::fileName& path, const std::string& data)
{
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out)
    {
        return false;
    }

    if (!data.empty())
    {
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    return out.good();
}

bool loadModuleFromDisk
(
    const word& archTag,
    const word& lookupKey,
    const std::string& source,
    const std::vector<std::string>& options,
    const char* kernelName,
    KernelCacheEntry& entry
)
{
    const Foam::fileName dir = cacheDirectoryForArch(archTag);
    if (dir.empty())
    {
        return false;
    }

    const std::string tag = makeKernelDiskTag(lookupKey, source, options);
    const Foam::fileName ptxPath(dir/(tag + ptxExtension()));

    if (!Foam::isFile(ptxPath))
    {
        return false;
    }

    std::string ptx;
    if (!readFileContents(ptxPath, ptx))
    {
        return false;
    }

    CUresult status = cuModuleLoadData(&entry.module, ptx.c_str());
    if (status != CUDA_SUCCESS)
    {
        entry.module = nullptr;
        return false;
    }

    status = cuModuleGetFunction(&entry.function, entry.module, kernelName);
    if (status != CUDA_SUCCESS)
    {
        cuModuleUnload(entry.module);
        entry.module = nullptr;
        entry.function = nullptr;
        return false;
    }

    entry.ptx = std::move(ptx);
    return true;
}

void storeModuleToDisk
(
    const word& archTag,
    const word& lookupKey,
    const std::string& source,
    const std::vector<std::string>& options,
    const std::string& ptx
)
{
    if (ptx.empty())
    {
        return;
    }

    const Foam::fileName dir = cacheDirectoryForArch(archTag);
    if (dir.empty())
    {
        return;
    }

    const std::string tag = makeKernelDiskTag(lookupKey, source, options);
    const Foam::fileName ptxPath(dir/(tag + ptxExtension()));
    writeFileContents(ptxPath, ptx);
}

std::unordered_map<std::string, KernelCacheEntry>& kernelCacheStorage()
{
    static std::unordered_map<std::string, KernelCacheEntry> cache;
    return cache;
}

std::mutex& kernelCacheMutex()
{
    static std::mutex mutex;
    return mutex;
}

#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
std::unordered_map<std::string, GraphEntry>& graphCacheStorage()
{
    static std::unordered_map<std::string, GraphEntry> cache;
    return cache;
}

std::mutex& graphCacheMutex()
{
    static std::mutex mutex;
    return mutex;
}
#endif

bool defaultGraphsRequested()
{
#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
    const std::string envValue = Foam::getEnv("FOAM_GPU_ENABLE_CUDA_GRAPHS");
    if (!envValue.empty())
    {
        return envSwitchOn(envValue);
    }
#endif
    return false;
}

std::atomic<bool>& graphsRequestedFlag()
{
    static std::atomic<bool> flag(defaultGraphsRequested());
    return flag;
}

}
#endif

KernelCache& KernelCache::instance()
{
    static KernelCache cache;
    return cache;
}


KernelCache::~KernelCache()
{
}


bool KernelCache::getOrCompile
(
    Context& ctx,
    const word& lookupKey,
    const std::string& source,
    const std::vector<std::string>& extraOptions,
    const char* kernelName,
    CompiledKernel& kernel,
    word& error
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

    const word& arch = ctx.architectureTag();
    if (arch.empty())
    {
        error = "CUDA context architecture unknown";
        return false;
    }

    std::vector<std::string> options;
    options.reserve(2 + extraOptions.size());
    options.emplace_back("--std=c++14");
    options.emplace_back("--gpu-architecture=" + arch);
    options.insert(options.end(), extraOptions.begin(), extraOptions.end());

    const std::string cacheKey = makeKernelCacheKey(arch, lookupKey, source, options);

    {
        std::lock_guard<std::mutex> guard(kernelCacheMutex());
        auto& cache = kernelCacheStorage();
        const auto iter = cache.find(cacheKey);
        if (iter != cache.end())
        {
            kernel.module = iter->second.module;
            kernel.function = iter->second.function;
            error.clear();
            return kernel.module != nullptr && kernel.function != nullptr;
        }
    }

    KernelCacheEntry cachedEntry;
    if
    (
        loadModuleFromDisk
        (
            arch,
            lookupKey,
            source,
            options,
            kernelName,
            cachedEntry
        )
    )
    {
        std::lock_guard<std::mutex> guard(kernelCacheMutex());
        auto& cache = kernelCacheStorage();
        const auto iter = cache.find(cacheKey);
        if (iter != cache.end())
        {
            kernel.module = iter->second.module;
            kernel.function = iter->second.function;
        }
        else
        {
            auto inserted = cache.emplace(cacheKey, std::move(cachedEntry));
            if (!inserted.second)
            {
                if (cachedEntry.module)
                {
                    cuModuleUnload(cachedEntry.module);
                    cachedEntry.module = nullptr;
                }
                kernel.module = inserted.first->second.module;
                kernel.function = inserted.first->second.function;
            }
            else
            {
                kernel.module = inserted.first->second.module;
                kernel.function = inserted.first->second.function;
            }
        }

        error.clear();
        return kernel.module != nullptr && kernel.function != nullptr;
    }

    nvrtcProgram prog;
    nvrtcResult nvStatus = nvrtcCreateProgram
    (
        &prog,
        source.c_str(),
        (lookupKey + ".cu").c_str(),
        0,
        nullptr,
        nullptr
    );

    if (nvStatus != NVRTC_SUCCESS)
    {
        error = "nvrtcCreateProgram failed: " + nvrtcErrorString(nvStatus);
        return false;
    }

    std::vector<const char*> optionPtrs;
    optionPtrs.reserve(options.size());
    for (const auto& opt : options)
    {
        optionPtrs.push_back(opt.c_str());
    }

    nvStatus = nvrtcCompileProgram
    (
        prog,
        static_cast<int>(optionPtrs.size()),
        optionPtrs.data()
    );

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

    KernelCacheEntry entry;
    entry.ptx = ptx;

    CUresult cuStatus = cuModuleLoadData(&entry.module, entry.ptx.c_str());
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleLoadData failed: " + cudaDriverErrorString(cuStatus);
        return false;
    }

    cuStatus = cuModuleGetFunction(&entry.function, entry.module, kernelName);
    if (cuStatus != CUDA_SUCCESS)
    {
        error = "cuModuleGetFunction failed: " + cudaDriverErrorString(cuStatus);
        cuModuleUnload(entry.module);
        return false;
    }

    storeModuleToDisk(arch, lookupKey, source, options, entry.ptx);

    {
        std::lock_guard<std::mutex> guard(kernelCacheMutex());
        auto& cache = kernelCacheStorage();
        auto inserted = cache.emplace(cacheKey, entry);
        if (!inserted.second)
        {
            // Another thread inserted concurrently. Use stored copy.
            cuModuleUnload(entry.module);
            kernel.module = inserted.first->second.module;
            kernel.function = inserted.first->second.function;
        }
        else
        {
            kernel.module = inserted.first->second.module;
            kernel.function = inserted.first->second.function;
        }
    }

    error.clear();
    return true;
#else
    (void)ctx;
    (void)lookupKey;
    (void)source;
    (void)extraOptions;
    (void)kernelName;
    (void)kernel;
    error = "CUDA support not available";
    return false;
#endif
}


void KernelCache::clear()
{
#ifdef FOAM_USE_CUDA
    std::lock_guard<std::mutex> guard(kernelCacheMutex());
    auto& cache = kernelCacheStorage();
    for (auto& pair : cache)
    {
        if (pair.second.module)
        {
            cuModuleUnload(pair.second.module);
            pair.second.module = nullptr;
            pair.second.function = nullptr;
        }
    }
    cache.clear();
#endif
}


GraphLaunchCache& GraphLaunchCache::instance()
{
    static GraphLaunchCache cache;
    return cache;
}


GraphLaunchCache::~GraphLaunchCache()
{
}


bool GraphLaunchCache::launch
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
)
{
#ifdef FOAM_USE_CUDA
    elapsedMs = 0;

    if (!kernel.function)
    {
        error = "Kernel function handle is null";
        return false;
    }

    if (!ctx.ready())
    {
        if (!ctx.initialise(0, error))
        {
            return false;
        }
    }

    cudaStream_t stream =
        streamCategory == StreamCategory::aux
      ? ctx.auxStream()
      : ctx.computeStream();

    if (!stream)
    {
        error = "CUDA stream unavailable";
        return false;
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    if (captureStats)
    {
        if
        (
            cudaEventCreateWithFlags(&startEvent, cudaEventDefault) != cudaSuccess
         || cudaEventCreateWithFlags(&stopEvent, cudaEventDefault) != cudaSuccess
        )
        {
            if (startEvent)
            {
                cudaEventDestroy(startEvent);
            }
            error = "cudaEventCreate failed";
            return false;
        }
        cudaEventRecord(startEvent, stream);
    }

    bool launched = false;

#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
    if (graphsEnabled())
    {
        const word& arch = ctx.architectureTag();
        if (arch.empty())
        {
            if (captureStats)
            {
                cudaEventDestroy(startEvent);
                cudaEventDestroy(stopEvent);
            }
            error = "CUDA context architecture unknown";
            return false;
        }

        const std::string cacheKey =
            makeGraphCacheKey(arch, kernelKey, grid, block, sharedMemBytes);

        GraphEntry* entryPtr = nullptr;
        {
            std::lock_guard<std::mutex> guard(graphCacheMutex());
            auto& cache = graphCacheStorage();
            auto iter = cache.find(cacheKey);
            if (iter == cache.end())
            {
                GraphEntry entry;
                entry.grid = grid;
                entry.block = block;
                entry.sharedMem = sharedMemBytes;
                entry.function = kernel.function;
                entry.paramBuffers.resize(numArgs);
                entry.initialParams.resize(numArgs);
                for (std::size_t i = 0; i < numArgs; ++i)
                {
                    entry.initialParams[i] = entry.paramBuffers[i].data();
                }

                CUresult status = cuGraphCreate(&entry.graph, 0);
                if (status != CUDA_SUCCESS)
                {
                    if (captureStats)
                    {
                        cudaEventDestroy(startEvent);
                        cudaEventDestroy(stopEvent);
                    }
                    error = "cuGraphCreate failed: "
                      + cudaDriverErrorString(status);
                    return false;
                }

                CUDA_KERNEL_NODE_PARAMS params{};
                params.func = kernel.function;
                params.gridDimX = grid.x;
                params.gridDimY = grid.y;
                params.gridDimZ = grid.z;
                params.blockDimX = block.x;
                params.blockDimY = block.y;
                params.blockDimZ = block.z;
                params.sharedMemBytes = sharedMemBytes;
                params.kernelParams = entry.initialParams.empty()
                    ? nullptr
                    : entry.initialParams.data();
                params.extra = nullptr;

                status = cuGraphAddKernelNode
                (
                    &entry.node,
                    entry.graph,
                    nullptr,
                    0,
                    &params
                );

                if (status != CUDA_SUCCESS)
                {
                    if (captureStats)
                    {
                        cudaEventDestroy(startEvent);
                        cudaEventDestroy(stopEvent);
                    }
                    error = "cuGraphAddKernelNode failed: "
                      + cudaDriverErrorString(status);
                    cuGraphDestroy(entry.graph);
                    return false;
                }

            #if CUDA_VERSION >= 11040
                status = cuGraphInstantiate(&entry.exec, entry.graph, 0);
            #else
                status = cuGraphInstantiate
                (
                    &entry.exec,
                    entry.graph,
                    nullptr,
                    nullptr,
                    0
                );
            #endif

                if (status != CUDA_SUCCESS)
                {
                    if (captureStats)
                    {
                        cudaEventDestroy(startEvent);
                        cudaEventDestroy(stopEvent);
                    }
                    error = "cuGraphInstantiate failed: "
                      + cudaDriverErrorString(status);
                    cuGraphDestroy(entry.graph);
                    return false;
                }

                auto inserted = cache.emplace(cacheKey, std::move(entry));
                entryPtr = &inserted.first->second;
            }
            else
            {
                entryPtr = &iter->second;
            }
        }

        CUDA_KERNEL_NODE_PARAMS params{};
        params.func = kernel.function;
        params.gridDimX = grid.x;
        params.gridDimY = grid.y;
        params.gridDimZ = grid.z;
        params.blockDimX = block.x;
        params.blockDimY = block.y;
        params.blockDimZ = block.z;
        params.sharedMemBytes = sharedMemBytes;
        params.kernelParams = args;
        params.extra = nullptr;

        CUresult status =
            cuGraphExecKernelNodeSetParams(entryPtr->exec, entryPtr->node, &params);

        if (status != CUDA_SUCCESS)
        {
            if (captureStats)
            {
                cudaEventDestroy(startEvent);
                cudaEventDestroy(stopEvent);
            }
            error = "cuGraphExecKernelNodeSetParams failed: "
              + cudaDriverErrorString(status);
            return false;
        }

        status = cuGraphLaunch(entryPtr->exec, reinterpret_cast<CUstream>(stream));
        if (status != CUDA_SUCCESS)
        {
            if (captureStats)
            {
                cudaEventDestroy(startEvent);
                cudaEventDestroy(stopEvent);
            }
            error = "cuGraphLaunch failed: " + cudaDriverErrorString(status);
            return false;
        }

        launched = true;
    }
#endif

    if (!launched)
    {
        CUresult status = cuLaunchKernel
        (
            kernel.function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            static_cast<unsigned int>(sharedMemBytes),
            reinterpret_cast<CUstream>(stream),
            args,
            nullptr
        );

        if (status != CUDA_SUCCESS)
        {
            if (captureStats)
            {
                cudaEventDestroy(startEvent);
                cudaEventDestroy(stopEvent);
            }
            error = "cuLaunchKernel failed: " + cudaDriverErrorString(status);
            return false;
        }
    }

    if (captureStats)
    {
        cudaEventRecord(stopEvent, stream);
        cudaEventSynchronize(stopEvent);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        elapsedMs = static_cast<scalar>(ms);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    else
    {
        elapsedMs = 0;
    }

    error.clear();
    return true;
#else
    (void)ctx;
    (void)kernelKey;
    (void)kernel;
    (void)streamCategory;
    (void)args;
    (void)sharedMemBytes;
    (void)captureStats;
    (void)grid;
    (void)block;
    elapsedMs = 0;
    error = "CUDA support not available";
    return false;
#endif
}


void GraphLaunchCache::clear()
{
#if defined(FOAM_USE_CUDA) && FOAM_GPU_CUDA_GRAPHS_AVAILABLE
    std::lock_guard<std::mutex> guard(graphCacheMutex());
    auto& cache = graphCacheStorage();
    for (auto& pair : cache)
    {
        if (pair.second.exec)
        {
            cuGraphExecDestroy(pair.second.exec);
            pair.second.exec = nullptr;
        }
        if (pair.second.graph)
        {
            cuGraphDestroy(pair.second.graph);
            pair.second.graph = nullptr;
        }
        pair.second.node = nullptr;
        pair.second.function = nullptr;
    }
    cache.clear();
#endif
}


#ifdef FOAM_USE_CUDA
bool graphsSupported()
{
#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
    return true;
#else
    return false;
#endif
}

void setCudaGraphsRequested(const bool enabled)
{
#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
    graphsRequestedFlag().store(enabled, std::memory_order_relaxed);
#else
    (void)enabled;
#endif
}

bool graphsEnabled()
{
#if FOAM_GPU_CUDA_GRAPHS_AVAILABLE
    return graphsRequestedFlag().load(std::memory_order_relaxed);
#else
    return false;
#endif
}

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
#endif

} // namespace gpu
} // namespace Foam
