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
#include "IStringStream.H"

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
#include <memory>
#include <climits>
#include <cstring>
#include <array>
#include <utility>
#include <tuple>
#include <random>
#include <numeric>
#include <unordered_set>
#include "bandCompression.H"

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

#if defined(ULLONG_MAX)
static constexpr std::size_t hashMagic = 0x9e3779b97f4a7c15ULL;
#else
static constexpr std::size_t hashMagic = 0x9e3779b9;
#endif

inline std::size_t hashCombine(std::size_t seed, std::size_t value)
{
    seed ^= value + hashMagic + (seed << 6) + (seed >> 2);
    return seed;
}

inline std::size_t hashDouble(double value)
{
    std::size_t hashBits = 0;
    static_assert(sizeof(hashBits) >= sizeof(value), "size_t must be at least as large as double");
    std::memcpy(&hashBits, &value, sizeof(value));
    return hashBits;
}

template<class Iterator>
inline std::size_t hashSequence(std::size_t seed, Iterator begin, Iterator end)
{
    for (Iterator it = begin; it != end; ++it)
    {
        seed = hashCombine(seed, static_cast<std::size_t>(*it));
    }
    return seed;
}

static void computeRuizScale
(
    const Foam::List<int>& rowPtr,
    const Foam::List<int>& colInd,
    const Foam::List<double>& values,
    std::vector<double>& scaleOut,
    const int sweeps = 2
)
{
    const std::size_t rows = rowPtr.size() ? rowPtr.size() - 1 : 0;
    scaleOut.assign(rows, 1.0);
    std::vector<double> tmpScale(rows, 1.0);
    const double eps = 1e-12;
    for (int sweep = 0; sweep < sweeps; ++sweep)
    {
        for (std::size_t i = 0; i < rows; ++i)
        {
            double sum = 0.0;
            const int start = rowPtr[i];
            const int end = rowPtr[i + 1];
            for (int idx = start; idx < end; ++idx)
            {
                const int j = colInd[idx];
                const double a = std::abs(static_cast<double>(values[idx]));
                sum += a * scaleOut[j];
            }
            sum *= std::max(scaleOut[i], eps);
            double rowNorm = std::sqrt(std::max(sum, eps));
            if (rowNorm <= eps)
            {
                rowNorm = 1.0;
            }
            tmpScale[i] = scaleOut[i]/rowNorm;
        }
        scaleOut.swap(tmpScale);
    }
}

enum class PreconCacheKind : int
{
    Diagonal = 0,
    IC0 = 1,
    ILU0 = 2,
    SGS = 3,
    Colour = 4,
    Chebyshev = 5,
    AMG = 6
};

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

struct PreconCacheKey
{
    std::size_t matrixPatternId = 0;
    std::size_t scaleId = 0;
    std::size_t permutationId = 0;
    int deviceId = -1;
    PreconCacheKind kind = PreconCacheKind::Diagonal;

    bool operator==(const PreconCacheKey& other) const noexcept
    {
        return matrixPatternId == other.matrixPatternId
            && scaleId == other.scaleId
            && permutationId == other.permutationId
            && deviceId == other.deviceId
            && kind == other.kind;
    }
};

struct PreconCacheKeyHash
{
    std::size_t operator()(const PreconCacheKey& key) const noexcept
    {
        std::size_t seed = 0;
        seed = hashCombine(seed, key.matrixPatternId);
        seed = hashCombine(seed, key.scaleId);
        seed = hashCombine(seed, key.permutationId);
        seed = hashCombine(seed, static_cast<std::size_t>(key.deviceId));
        seed = hashCombine(seed, static_cast<std::size_t>(key.kind));
        return seed;
    }
};

struct PreconCacheEntry
{
    Foam::label capacity = 0;
    double* d_sqrtDiag = nullptr;
    double* d_invSqrtDiag = nullptr;
    double* d_scaledResidual = nullptr;
    double* d_scaledDirection = nullptr;
    double* d_scaledAp = nullptr;
    double* d_scaledTmp = nullptr;
    double* d_ruizScale = nullptr;
    double* d_ruizInvScale = nullptr;
    Foam::label icCacheKeyId = 0;
    Foam::label flooredCount = 0;
    Foam::label invSqrtClampCount = 0;
    double diagFloorApplied = 0.0;
    double invSqrtClampLimit = 0.0;
    double lambdaMin = 0.0;
    double lambdaMax = 0.0;
    bool lambdaReady = false;
    Foam::label icCapacityNnz = 0;
    double* d_icVals = nullptr;
    csric02Info_t icInfo = nullptr;
    bool icAnalysisReady = false;
    size_t icBufferSize = 0;
    void* icBuffer = nullptr;
    Foam::label icShiftCounter = 0;
    Foam::label icAnalysisReuseHits = 0;
    Foam::label icFactorReuseHits = 0;
    Foam::label icRefactorCount = 0;
    bool icFactorValid = false;
    bool icHadZeroPivot = false;
    std::size_t cachedMatrixPatternId = 0;
    std::size_t cachedScaleId = 0;
    std::size_t cachedPermutationId = 0;
    PreconCacheKind cachedKind = PreconCacheKind::Diagonal;
    double cachedMatvecSign = 1.0;
    double icFactorScale = 1.0;
    double icApplyScale = 1.0;
    Foam::label permutationSize = 0;
    int* d_perm = nullptr;
    bool permutationReady = false;
    Foam::label icPatternRows = 0;
    Foam::label icPatternNnz = 0;
    int* d_icRowPtr = nullptr;
    int* d_icColInd = nullptr;
    bool icPatternReady = false;
    bool permutationValidated = false;
    bool permutationStructureValidated = false;
    Foam::label icLastIterations = 0;
    Foam::scalar icRollingMedian = 0;
    std::deque<Foam::label> icIterationHistory;
    std::vector<std::pair<double, Foam::label>> icShiftTrace;
    bool ruizReady = false;
    double ruizClampRatio = 0.0;
    double ruizScaleMin = 1.0;
    double ruizScaleMax = 1.0;
    bool icUsedDiagonalScaling = false;
    bool icUsedRuizScaling = false;
    Foam::autoPtr<Foam::lduMatrix::solver> amgSolver;
    Foam::scalarField amgHostR;
    Foam::scalarField amgHostZ;
    Foam::label amgApplyCount = 0;
    Foam::scalar amgSetupMs = 0;
    Foam::scalar amgApplyMs = 0;

    ~PreconCacheEntry()
    {
        release();
    }

    void release()
    {
        if (d_sqrtDiag)
        {
            cudaFree(d_sqrtDiag);
            d_sqrtDiag = nullptr;
        }
        if (d_invSqrtDiag)
        {
            cudaFree(d_invSqrtDiag);
            d_invSqrtDiag = nullptr;
        }
        if (d_scaledResidual)
        {
            cudaFree(d_scaledResidual);
            d_scaledResidual = nullptr;
        }
        if (d_scaledDirection)
        {
            cudaFree(d_scaledDirection);
            d_scaledDirection = nullptr;
        }
        if (d_scaledAp)
        {
            cudaFree(d_scaledAp);
            d_scaledAp = nullptr;
        }
        if (d_scaledTmp)
        {
            cudaFree(d_scaledTmp);
            d_scaledTmp = nullptr;
        }
        if (d_ruizScale)
        {
            cudaFree(d_ruizScale);
            d_ruizScale = nullptr;
        }
        if (d_ruizInvScale)
        {
            cudaFree(d_ruizInvScale);
            d_ruizInvScale = nullptr;
        }
        if (d_perm)
        {
            cudaFree(d_perm);
            d_perm = nullptr;
        }
        if (d_icRowPtr)
        {
            cudaFree(d_icRowPtr);
            d_icRowPtr = nullptr;
        }
        if (d_icColInd)
        {
            cudaFree(d_icColInd);
            d_icColInd = nullptr;
        }
        capacity = 0;
        flooredCount = 0;
        invSqrtClampCount = 0;
        diagFloorApplied = 0.0;
        invSqrtClampLimit = 0.0;
        lambdaMin = 0.0;
        lambdaMax = 0.0;
        lambdaReady = false;
        if (d_icVals)
        {
            cudaFree(d_icVals);
            d_icVals = nullptr;
        }
        if (icBuffer)
        {
            cudaFree(icBuffer);
            icBuffer = nullptr;
        }
        if (icInfo)
        {
            cusparseDestroyCsric02Info(icInfo);
            icInfo = nullptr;
        }
        icCapacityNnz = 0;
        icAnalysisReady = false;
        icBufferSize = 0;
        icShiftCounter = 0;
        icAnalysisReuseHits = 0;
        icFactorReuseHits = 0;
        icRefactorCount = 0;
        icFactorValid = false;
        icHadZeroPivot = false;
        icFactorScale = 1.0;
        icApplyScale = 1.0;
        cachedMatrixPatternId = 0;
        cachedScaleId = 0;
        cachedPermutationId = 0;
        cachedKind = PreconCacheKind::Diagonal;
        cachedMatvecSign = 1.0;
        permutationSize = 0;
        permutationReady = false;
        icPatternRows = 0;
        icPatternNnz = 0;
        icPatternReady = false;
        permutationValidated = false;
        permutationStructureValidated = false;
        icLastIterations = 0;
        icRollingMedian = 0;
        icIterationHistory.clear();
        icShiftTrace.clear();
        ruizReady = false;
        ruizClampRatio = 0.0;
        ruizScaleMin = 1.0;
        ruizScaleMax = 1.0;
        amgSolver.clear();
        amgHostR.clear();
        amgHostZ.clear();
        amgApplyCount = 0;
        amgSetupMs = 0;
        amgApplyMs = 0;
    }

    bool ensureCapacity(Foam::label n)
    {
        if (n < 0)
        {
            n = 0;
        }
        if
        (
            capacity == n
         && d_sqrtDiag
         && d_invSqrtDiag
         && d_scaledResidual
         && d_scaledDirection
         && d_scaledAp
         && d_scaledTmp
         && d_ruizScale
         && d_ruizInvScale
        )
        {
            return true;
        }
        release();
        if (!n)
        {
            return true;
        }
        if
        (
            cudaMalloc(reinterpret_cast<void**>(&d_sqrtDiag), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_invSqrtDiag), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_scaledResidual), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_scaledDirection), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_scaledAp), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_scaledTmp), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_ruizScale), sizeof(double)*n) != cudaSuccess
         || cudaMalloc(reinterpret_cast<void**>(&d_ruizInvScale), sizeof(double)*n) != cudaSuccess
        )
        {
            release();
            return false;
        }
        capacity = n;
        return true;
    }

    bool ensureIcCapacity(Foam::label nnz)
    {
        if (nnz < 0)
        {
            nnz = 0;
        }
        if (icCapacityNnz == nnz && d_icVals)
        {
            return true;
        }
        if (d_icVals)
        {
            cudaFree(d_icVals);
            d_icVals = nullptr;
        }
        icCapacityNnz = 0;
        if (!nnz)
        {
            return true;
        }
        if
        (
            cudaMalloc
            (
                reinterpret_cast<void**>(&d_icVals),
                sizeof(double)*nnz
            ) != cudaSuccess
        )
        {
            d_icVals = nullptr;
            return false;
        }
        icCapacityNnz = nnz;
        return true;
    }

    bool ensureIcBuffer(size_t bytes)
    {
        if (icBuffer && icBufferSize >= bytes)
        {
            return true;
        }
        if (icBuffer)
        {
            cudaFree(icBuffer);
            icBuffer = nullptr;
            icBufferSize = 0;
        }
        if (!bytes)
        {
            return true;
        }
        if (cudaMalloc(&icBuffer, bytes) != cudaSuccess)
        {
            icBuffer = nullptr;
            return false;
        }
        icBufferSize = bytes;
        return true;
    }

    void updateKeyMetadata
    (
        std::size_t matrixPatternId,
        std::size_t scaleId,
        std::size_t permutationId,
        PreconCacheKind kindIn
    )
    {
        cachedMatrixPatternId = matrixPatternId;
        cachedScaleId = scaleId;
        cachedPermutationId = permutationId;
        cachedKind = kindIn;
    }

    void resetIcReuseState()
    {
        icFactorValid = false;
        icHadZeroPivot = false;
        icAnalysisReady = false;
        icAnalysisReuseHits = 0;
        icFactorReuseHits = 0;
        icRefactorCount = 0;
        icFactorScale = 1.0;
        icApplyScale = 1.0;
        permutationReady = false;
        icPatternReady = false;
        permutationValidated = false;
        permutationStructureValidated = false;
        icLastIterations = 0;
        icRollingMedian = 0;
        icIterationHistory.clear();
        icShiftTrace.clear();
    }

    bool setPermutationData(const Foam::labelList& permutation)
    {
        permutationReady = false;
        permutationValidated = false;
        permutationStructureValidated = false;

        if (permutation.empty())
        {
            if (d_perm)
            {
                cudaFree(d_perm);
                d_perm = nullptr;
            }
            permutationSize = 0;
            return true;
        }

        const Foam::label n = permutation.size();
        Foam::List<int> permInt(n);
        forAll(permInt, i)
        {
            permInt[i] = static_cast<int>(permutation[i]);
        }

        if (permutationSize != n || !d_perm)
        {
            if (d_perm)
            {
                cudaFree(d_perm);
                d_perm = nullptr;
            }
            if (cudaMalloc(reinterpret_cast<void**>(&d_perm), sizeof(int)*n) != cudaSuccess)
            {
                d_perm = nullptr;
                permutationSize = 0;
                return false;
            }
            permutationSize = n;
        }

        if
        (
            cudaMemcpy
            (
                d_perm,
                permInt.begin(),
                sizeof(int)*n,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        )
        {
            permutationReady = false;
            return false;
        }
        permutationReady = true;
        return true;
    }

    bool setIcPatternData
    (
        const Foam::labelList& rowPtrHost,
        const Foam::labelList& colIndHost
    )
    {
        icPatternReady = false;
        permutationStructureValidated = false;

        const Foam::label rows = rowPtrHost.size() ? rowPtrHost.size() - 1 : 0;
        const Foam::label nnz = colIndHost.size();

        if (!rows || !nnz)
        {
            if (d_icRowPtr)
            {
                cudaFree(d_icRowPtr);
                d_icRowPtr = nullptr;
            }
            if (d_icColInd)
            {
                cudaFree(d_icColInd);
                d_icColInd = nullptr;
            }
            icPatternRows = 0;
            icPatternNnz = 0;
            return true;
        }

        Foam::List<int> rowPtrInt(rows + 1);
        for (Foam::label i = 0; i <= rows; ++i)
        {
            rowPtrInt[i] = static_cast<int>(rowPtrHost[i]);
        }

        Foam::List<int> colIndInt(nnz);
        for (Foam::label i = 0; i < nnz; ++i)
        {
            colIndInt[i] = static_cast<int>(colIndHost[i]);
        }

        if (icPatternRows != rows || !d_icRowPtr)
        {
            if (d_icRowPtr)
            {
                cudaFree(d_icRowPtr);
                d_icRowPtr = nullptr;
            }
            if
            (
                cudaMalloc
                (
                    reinterpret_cast<void**>(&d_icRowPtr),
                    sizeof(int)*(rows + 1)
                ) != cudaSuccess
            )
            {
                d_icRowPtr = nullptr;
                icPatternRows = 0;
                icPatternNnz = 0;
                return false;
            }
            icPatternRows = rows;
        }

        if (icPatternNnz != nnz || !d_icColInd)
        {
            if (d_icColInd)
            {
                cudaFree(d_icColInd);
                d_icColInd = nullptr;
            }
            if
            (
                cudaMalloc
                (
                    reinterpret_cast<void**>(&d_icColInd),
                    sizeof(int)*nnz
                ) != cudaSuccess
            )
            {
                if (d_icRowPtr && icPatternRows != rows)
                {
                    cudaFree(d_icRowPtr);
                    d_icRowPtr = nullptr;
                    icPatternRows = 0;
                }
                d_icColInd = nullptr;
                icPatternNnz = 0;
                return false;
            }
            icPatternNnz = nnz;
        }

        if
        (
            cudaMemcpy
            (
                d_icRowPtr,
                rowPtrInt.begin(),
                sizeof(int)*(rows + 1),
                cudaMemcpyHostToDevice
            ) != cudaSuccess
         || cudaMemcpy
            (
                d_icColInd,
                colIndInt.begin(),
                sizeof(int)*nnz,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        )
        {
            icPatternReady = false;
            return false;
        }

        icPatternReady = true;
        return true;
    }

    Foam::scalar computeMedian() const
    {
        if (icIterationHistory.empty())
        {
            return 0;
        }
        std::vector<Foam::label> sorted(icIterationHistory.begin(), icIterationHistory.end());
        std::sort(sorted.begin(), sorted.end());
        const std::size_t mid = sorted.size()/2;
        if (sorted.size() % 2 == 0)
        {
            if (mid == 0 || mid >= sorted.size())
            {
                return static_cast<Foam::scalar>(sorted.front());
            }
            return static_cast<Foam::scalar>(sorted[mid - 1] + sorted[mid])*Foam::scalar(0.5);
        }
        return static_cast<Foam::scalar>(sorted[mid]);
    }

    void recordIterations(Foam::label iterations)
    {
        if (iterations <= 0)
        {
            return;
        }
        constexpr std::size_t window = 8;
        icIterationHistory.push_back(iterations);
        if (icIterationHistory.size() > window)
        {
            icIterationHistory.pop_front();
        }
        icLastIterations = iterations;
        icRollingMedian = computeMedian();
    }
};

using PreconCacheMap = std::unordered_map
<
    PreconCacheKey,
    std::unique_ptr<PreconCacheEntry>,
    PreconCacheKeyHash
>;

static std::mutex preconCacheMutex;
static PreconCacheMap preconCache;

static PreconCacheEntry& getPreconCacheEntry(const PreconCacheKey& key)
{
    std::lock_guard<std::mutex> guard(preconCacheMutex);
    static std::size_t nextCacheKeyId = 1;
    auto iter = preconCache.find(key);
    if (iter == preconCache.end())
    {
        auto inserted = preconCache.emplace
        (
            key,
            std::unique_ptr<PreconCacheEntry>(new PreconCacheEntry())
        );
        PreconCacheEntry& entry = *(inserted.first->second);
        entry.icCacheKeyId = static_cast<Foam::label>(nextCacheKeyId++);
        entry.updateKeyMetadata(key.matrixPatternId, key.scaleId, key.permutationId, key.kind);
        return entry;
    }
    PreconCacheEntry& entry = *(iter->second);
    entry.updateKeyMetadata(key.matrixPatternId, key.scaleId, key.permutationId, key.kind);
    return entry;
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

static const char* permutationKernelSource = R"(
extern "C" __global__
void permutationGather
(
    int n,
    const int* __restrict__ perm,
    const double* __restrict__ inVec,
    double* __restrict__ outVec
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    outVec[idx] = inVec[perm[idx]];
}

extern "C" __global__
void permutationScatter
(
    int n,
    const int* __restrict__ perm,
    const double* __restrict__ inVec,
    double* __restrict__ outVec
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    outVec[perm[idx]] = inVec[idx];
}
)";

struct PermutationKernelCacheEntry
{
    int major = 0;
    int minor = 0;
    CUmodule module = nullptr;
    CUfunction gather = nullptr;
    CUfunction scatter = nullptr;
    bool ok = false;
};

static std::mutex permutationKernelMutex;
static std::vector<PermutationKernelCacheEntry> permutationKernelCache;

bool compilePermutationKernels
(
    const cudaDeviceProp& prop,
    PermutationKernelCacheEntry& entry,
    bool verbose
)
{
    entry.major = prop.major;
    entry.minor = prop.minor;
    entry.module = nullptr;
    entry.gather = nullptr;
    entry.scatter = nullptr;
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
    nvrtcResult nvRes = nvrtcCreateProgram
    (
        &prog,
        permutationKernelSource,
        "permutationKernel.cu",
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

    std::string archStr = std::string("--gpu-architecture=compute_")
        + std::to_string(prop.major)
        + std::to_string(prop.minor);

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
                << "nvrtcCompileProgram failed (" << nvrtcStatusName(nvRes) << ")"
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

    cuRes = cuModuleGetFunction(&entry.gather, entry.module, "permutationGather");
    if (cuRes != CUDA_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "cuModuleGetFunction gather failed: " << cuStatusName(cuRes) << Foam::endl;
        }
        cuModuleUnload(entry.module);
        entry.module = nullptr;
        return false;
    }

    cuRes = cuModuleGetFunction(&entry.scatter, entry.module, "permutationScatter");
    if (cuRes != CUDA_SUCCESS)
    {
        if (verbose)
        {
            WarningInFunction << "cuModuleGetFunction scatter failed: " << cuStatusName(cuRes) << Foam::endl;
        }
        cuModuleUnload(entry.module);
        entry.module = nullptr;
        entry.gather = nullptr;
        return false;
    }

    entry.ok = true;
    return true;
}

bool getPermutationKernels
(
    const int deviceId,
    CUfunction& gather,
    CUfunction& scatter,
    bool verbose
)
{
    std::lock_guard<std::mutex> guard(permutationKernelMutex);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    {
        if (verbose)
        {
            WarningInFunction << "cudaGetDeviceProperties failed for device " << deviceId << Foam::endl;
        }
        return false;
    }

    for (const PermutationKernelCacheEntry& entry : permutationKernelCache)
    {
        if (entry.major == prop.major && entry.minor == prop.minor && entry.ok)
        {
            gather = entry.gather;
            scatter = entry.scatter;
            return true;
        }
    }

    PermutationKernelCacheEntry entry;
    if (!compilePermutationKernels(prop, entry, verbose))
    {
        return false;
    }
    permutationKernelCache.push_back(entry);
    gather = entry.gather;
    scatter = entry.scatter;
    return true;
}

inline bool launchPermutationKernel
(
    CUfunction kernel,
    int n,
    CUdeviceptr perm,
    CUdeviceptr inVec,
    CUdeviceptr outVec,
    CUstream stream
)
{
    const unsigned blockSize = 256;
    const unsigned gridSize = static_cast<unsigned>((n + blockSize - 1)/blockSize);
    void* args[] =
    {
        &n,
        &perm,
        &inVec,
        &outVec
    };
    CUresult cuRes =
        cuLaunchKernel
        (
            kernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0,
            stream,
            args,
            nullptr
        );
    return cuRes == CUDA_SUCCESS;
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

    const label diagHistogramBinsCfg =
        std::max<label>(0, dict.lookupOrDefault<label>("diagHistogramBins", 0));
    fileName diagHistogramFile =
        dict.lookupOrDefault<fileName>("diagHistogramFile", fileName::null);

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
    std::vector<scalar> diagHistogramQuantiles;
    bool colourFallbackTriggered = false;
    bool colourSpdMode = false;
    PreconCacheEntry* preconCacheEntry = nullptr;
    bool chebyshevRequested = false;
    bool chebyshevForced = false;
    bool chebyshevSpdApplied = false;
    bool chebyshevGuardrailTriggered = false;
    bool chebyshevRestartAttempted = false;
    bool chebyshevRestarted = false;
    bool chebyshevDotGuardTriggered = false;
    bool chebyshevClampGuardTriggered = false;
    bool chebyshevClampGate = false;
    bool icForced = false;
    bool icAppliedThisSolve = false;
    bool icDotGuardTriggered = false;
    bool icEffectComputed = false;
    double icPrecondEffect0 = 0.0;
    scalar chebyshevLambdaMinUsed = 0;
    scalar chebyshevLambdaMaxUsed = 0;
    scalar chebyshevLambdaInflateUsed = scalar(1);
    label chebyshevDegreeUsed = 0;
    scalar chebyshevPrevResidual = -1;
    label chebyshevGrowthCounter = 0;

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

    const Switch icUseDiagonalScalingSwitch =
        dict.lookupOrDefault<Switch>("icUseDiagonalScaling", Switch(true));
    const Switch icUseRuizScalingSwitch =
        dict.lookupOrDefault<Switch>("icUseRuizScaling", Switch(true));
    const Switch icUseRCMSwitch =
        dict.lookupOrDefault<Switch>("icUseRCM", Switch(false));
    const Switch icNormalizeFactorSwitch =
        dict.lookupOrDefault<Switch>("icNormalizeFactor", Switch(true));
    const Switch icDebugDump =
        dict.lookupOrDefault<Switch>("icDebugDump", Switch(false));
    const label icDebugIterLog =
        dict.lookupOrDefault<label>("icDebugIterLog", label(3));
    const bool icUseDiagonalScaling = icUseDiagonalScalingSwitch;
    const bool icUseRuizScaling = icUseRuizScalingSwitch;
    const bool icUseRCM = icUseRCMSwitch;
    const bool icNormalizeFactor = icNormalizeFactorSwitch;
    const scalar icNormalizeClampMinCfg =
        dict.lookupOrDefault<scalar>("icNormalizeClampMin", scalar(1e-2));
    const scalar icNormalizeClampMaxCfg =
        dict.lookupOrDefault<scalar>("icNormalizeClampMax", scalar(10));
    const scalar icNormalizeTargetEffectCfg =
        dict.lookupOrDefault<scalar>("icNormalizeTargetEffect", scalar(1));
    const scalar icDiagBoostFractionCfg =
        dict.lookupOrDefault<scalar>("icDiagBoostFraction", scalar(0));
    const scalar icDiagBoostMinCfg =
        dict.lookupOrDefault<scalar>("icDiagBoostMin", scalar(0));
    const scalar icDiagBoostMaxCfg =
        dict.lookupOrDefault<scalar>("icDiagBoostMax", scalar(0));



    std::size_t permutationId = 0;
    Foam::labelList permutation;
    Foam::labelList permutationInverse;
    bool permutationActive = false;
    if (icUseRCM)
    {
        Foam::labelList offsets(rows + 1);
        for (int i = 0; i <= rows; ++i)
        {
            offsets[i] = rowPtr[i];
        }
        Foam::labelList adjacency(nnz);
        for (int i = 0; i < nnz; ++i)
        {
            adjacency[i] = colInd[i];
        }

        Foam::labelList rcmOrder = bandCompression(adjacency, offsets);
        if (rcmOrder.size() == rows)
        {
            permutation.setSize(rows);
            permutationInverse.setSize(rows);
            permutationActive = false;
            for (Foam::label newIdx = 0; newIdx < rows; ++newIdx)
            {
                const Foam::label origIdx = rcmOrder[newIdx];
                permutation[newIdx] = origIdx;
                permutationInverse[origIdx] = newIdx;
                if (!permutationActive && origIdx != newIdx)
                {
                    permutationActive = true;
                }
            }
            if (!permutationActive)
            {
                permutation.clear();
                permutationInverse.clear();
            }
            else
            {
                permutationId = hashSequence(static_cast<std::size_t>(0), permutation.begin(), permutation.end());
            }
        }
    }

    std::size_t matrixPatternId = hashSequence(static_cast<std::size_t>(rows), rowPtr.begin(), rowPtr.end());
    matrixPatternId = hashSequence(matrixPatternId, colInd.begin(), colInd.end());

    const scalarField& diag = matrix_.diag();
    const scalar diagFloorAbsCfg =
        dict.lookupOrDefault<scalar>("preconditionerDiagFloor", scalar(1e-20));
    const scalar diagFloorRelCfg =
        dict.lookupOrDefault<scalar>("preconditionerDiagRelFloor", scalar(1e-10));
    const scalar invSqrtClampCfg =
        dict.lookupOrDefault<scalar>("preconditionerInvSqrtClamp", scalar(1e4));
    const scalar diagQuantileFracCfg =
        max(scalar(0), dict.lookupOrDefault<scalar>("preconditionerDiagQuantile", scalar(0.01)));
    const scalar diagQuantileAlphaCfg =
        dict.lookupOrDefault<scalar>("preconditionerDiagQuantileAlpha", scalar(0.5));
    const scalar invSqrtClampSafetyCfg =
        dict.lookupOrDefault<scalar>("preconditionerInvSqrtClampSafety", scalar(1));
    const scalar invSqrtClampMaxCfg =
        dict.lookupOrDefault<scalar>("preconditionerInvSqrtClampMax", scalar(1e5));
    const Switch diagSnapshotEnable =
        dict.lookupOrDefault<Switch>("preconditionerDiagSnapshot", Switch(false));
    const scalar diagSnapshotClampRatioCfg =
        dict.lookupOrDefault<scalar>("preconditionerDiagSnapshotClampRatio", scalar(0.01));
    const fileName diagSnapshotFileCfg =
        dict.lookupOrDefault<fileName>("preconditionerDiagSnapshotFile", fileName::null);

    scalar minDiag = std::numeric_limits<scalar>::max();
    scalar maxDiag = -std::numeric_limits<scalar>::max();
    scalar maxAbsDiag = 0;
    std::vector<scalar> absDiagVals;
    absDiagVals.reserve(diag.size());
    for (const scalar d : diag)
    {
        minDiag = std::min(minDiag, d);
        maxDiag = std::max(maxDiag, d);
        const scalar absVal = mag(d);
        maxAbsDiag = std::max(maxAbsDiag, absVal);
        absDiagVals.push_back(absVal);
    }

    scalar medianAbsDiag = 0;
    if (!absDiagVals.empty())
    {
        std::sort(absDiagVals.begin(), absDiagVals.end());
        const label midIdx = static_cast<label>(absDiagVals.size()/2);
        if (absDiagVals.size() % 2 == 0)
        {
            const scalar upperMid = absDiagVals[midIdx];
            const scalar lowerMid =
                midIdx > 0 ? absDiagVals[midIdx - 1] : absDiagVals[midIdx];
            medianAbsDiag = scalar(0.5)*(upperMid + lowerMid);
        }
        else
        {
            medianAbsDiag = absDiagVals[midIdx];
        }
    }

    const scalar matvecSignCfg = dict.lookupOrDefault<scalar>("matvecSign", scalar(-1));
    const scalar matvecSign = matvecSignCfg >= scalar(0) ? scalar(1) : scalar(-1);
    const double matvecSignD = static_cast<double>(matvecSign);

    scalar diagFloorCfg = diagFloorAbsCfg;
    scalar diagQuantileValue = 0;
    if (!absDiagVals.empty() && diagQuantileFracCfg > scalar(0))
    {
        const scalar clampedFrac = std::min(std::max(diagQuantileFracCfg, scalar(0)), scalar(0.5));
        const scalar position = clampedFrac * scalar(absDiagVals.size() - 1);
        const label lowerIdx = static_cast<label>(std::max(scalar(0), std::floor(position)));
        const label upperIdx = static_cast<label>(std::min(scalar(absDiagVals.size() - 1), std::ceil(position)));
        const scalar weight = position - scalar(lowerIdx);
        const scalar lowerVal = absDiagVals[lowerIdx];
        const scalar upperVal = absDiagVals[upperIdx];
        diagQuantileValue = (1 - weight)*lowerVal + weight*upperVal;
    }
    if (diagHistogramBinsCfg > 0 && !absDiagVals.empty())
    {
        const label bins = diagHistogramBinsCfg;
        diagHistogramQuantiles.clear();
        diagHistogramQuantiles.reserve(bins + 1);
        for (label i = 0; i <= bins; ++i)
        {
            const scalar fraction = bins ? scalar(i)/scalar(bins) : scalar(0);
            const scalar position = fraction * scalar(absDiagVals.size() - 1);
            const label lowerIdx = static_cast<label>(std::max(scalar(0), std::floor(position)));
            const label upperIdx = static_cast<label>(std::min(scalar(absDiagVals.size() - 1), std::ceil(position)));
            const scalar weight = position - scalar(lowerIdx);
            const scalar lowerVal = absDiagVals[lowerIdx];
            const scalar upperVal = absDiagVals[upperIdx];
            diagHistogramQuantiles.push_back((1 - weight)*lowerVal + weight*upperVal);
        }
    }
    if (diagFloorRelCfg > scalar(0))
    {
        const scalar relReference =
            (medianAbsDiag > VSMALL) ? medianAbsDiag : maxAbsDiag;
        diagFloorCfg = std::max(diagFloorCfg, diagFloorRelCfg*relReference);
    }
    if (diagQuantileValue > scalar(0) && diagQuantileAlphaCfg > scalar(0))
    {
        diagFloorCfg = std::max(diagFloorCfg, diagQuantileAlphaCfg*diagQuantileValue);
    }
    if (diagFloorCfg <= scalar(0))
    {
        diagFloorCfg = std::max(diagFloorAbsCfg, scalar(1e-20));
    }

    const labelUList& upperAddr = matrix_.lduAddr().upperAddr();
    const labelUList& lowerAddr = matrix_.lduAddr().lowerAddr();
    const scalarField& upper = matrix_.upper();

    scalarField dicDiag(diag);
    label numFlooredDiag = 0;
    forAll(dicDiag, i)
    {
        scalar& d = dicDiag[i];
        const scalar absVal = mag(d);
        if (absVal < diagFloorCfg)
        {
            ++numFlooredDiag;
        }
        d = std::max(absVal, diagFloorCfg);
    }

    scalar dicDiagMedianPreBoost = scalar(0);
    if
    (
        (icDiagBoostFractionCfg > scalar(0))
     || (icDiagBoostMinCfg > scalar(0))
     || (icDiagBoostMaxCfg > scalar(0))
    )
    {
        std::vector<scalar> dicDiagAbs;
        dicDiagAbs.reserve(dicDiag.size());
        forAll(dicDiag, i)
        {
            const scalar absVal = mag(dicDiag[i]);
            if (absVal > VSMALL)
            {
                dicDiagAbs.push_back(absVal);
            }
        }
        if (!dicDiagAbs.empty())
        {
            const auto computeMedianScalar = [](std::vector<scalar>& data) -> scalar
            {
                const std::size_t mid = data.size()/2;
                std::nth_element(data.begin(), data.begin() + mid, data.end());
                scalar median = data[mid];
                if (data.size() % 2 == 0 && mid > 0)
                {
                    std::nth_element(data.begin(), data.begin() + mid - 1, data.end());
                    median = scalar(0.5)*(median + data[mid - 1]);
                }
                return median;
            };
            dicDiagMedianPreBoost = computeMedianScalar(dicDiagAbs);
            scalar boostTarget = scalar(0);
            if (icDiagBoostFractionCfg > scalar(0) && dicDiagMedianPreBoost > VSMALL)
            {
                boostTarget = icDiagBoostFractionCfg * dicDiagMedianPreBoost;
            }
            scalar diagBoost = std::max(icDiagBoostMinCfg, boostTarget);
            if (icDiagBoostMaxCfg > scalar(0))
            {
                diagBoost = std::min(diagBoost, icDiagBoostMaxCfg);
            }
            if (diagBoost > scalar(0))
            {
                forAll(dicDiag, i)
                {
                    dicDiag[i] += diagBoost;
                }
                if (reportStats && (!Pstream::parRun() || Pstream::master()))
                {
                    scalar minPre = dicDiagAbs.front();
                    scalar maxPre = dicDiagAbs.front();
                    for (scalar val : dicDiagAbs)
                    {
                        minPre = std::min(minPre, val);
                        maxPre = std::max(maxPre, val);
                    }
                    Info<< "cudaPCG(" << fieldName_ << "): IC diag boost delta="
                        << diagBoost
                        << " medianPre=" << dicDiagMedianPreBoost
                        << " rangePre=[" << minPre << ',' << maxPre << ']'
                        << nl;
                }
            }
        }
        else if (reportStats && (!Pstream::parRun() || Pstream::master()))
        {
            Info<< "cudaPCG(" << fieldName_ << "): IC diag boost skipped (no positive entries)"
                << nl;
        }
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
    label invSqrtClamped = 0;
    double invSqrtClampUsed = static_cast<double>(invSqrtClampCfg);
    double icDiagMedianScaled = std::max(static_cast<double>(diagFloorCfg), static_cast<double>(medianAbsDiag));
    if (invSqrtClampUsed > 0 && diagFloorCfg > scalar(0))
    {
        const double diagClamp = 1.0/std::sqrt(static_cast<double>(diagFloorCfg));
        invSqrtClampUsed = std::max
        (
            invSqrtClampUsed,
            static_cast<double>(invSqrtClampSafetyCfg) * diagClamp
        );
    }
    if (diagQuantileValue > scalar(0) && invSqrtClampUsed > 0)
    {
        const double quantClamp =
            static_cast<double>(invSqrtClampSafetyCfg)
          / std::sqrt(std::max(static_cast<double>(diagQuantileValue), double(VSMALL)));
        invSqrtClampUsed = std::max(invSqrtClampUsed, quantClamp);
    }
    if (invSqrtClampMaxCfg > scalar(0))
    {
        invSqrtClampUsed = std::min
        (
            invSqrtClampUsed,
            static_cast<double>(invSqrtClampMaxCfg)
        );
    }
    scalar invSqrtClampRatio = 0;

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

    if (reportStats && (!Pstream::parRun() || Pstream::master()))
    {
        auto computeMedianLocal = [](std::vector<double>& data) -> double
        {
            if (data.empty())
            {
                return 0.0;
            }
            std::size_t mid = data.size()/2;
            std::nth_element(data.begin(), data.begin() + mid, data.end());
            double median = data[mid];
            if (data.size() % 2 == 0 && mid > 0)
            {
                std::nth_element(data.begin(), data.begin() + mid - 1, data.end());
                median = 0.5*(median + data[mid - 1]);
            }
            return median;
        };

        double dicMin = std::numeric_limits<double>::max();
        double dicMax = -std::numeric_limits<double>::max();
        std::vector<double> dicPos;
        dicPos.reserve(precondDiag.size());
        forAll(precondDiag, i)
        {
            const double val = static_cast<double>(precondDiag[i]);
            dicMin = std::min(dicMin, val);
            dicMax = std::max(dicMax, val);
            if (val > 0.0)
            {
                dicPos.push_back(val);
            }
        }
        const double dicMedian = computeMedianLocal(dicPos);
        const double flooredFrac =
            rows > 0 ? static_cast<double>(numFlooredDiag)/static_cast<double>(rows) : 0.0;
        Info<< "cudaPCG(" << fieldName_ << "): dicDiag stats "
            << "min=" << dicMin
            << " max=" << dicMax
            << " median=" << dicMedian
            << " floored=" << numFlooredDiag
            << " flooredFrac=" << flooredFrac
            << " invSqrtClampUsed=" << invSqrtClampUsed
            << nl;
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
    bool d_iluValsFromCache = false;
    bool icInfoFromCache = false;
    bool icBufferFromCache = false;
    List<DeviceScalar> sqrtDiagHost;
    List<DeviceScalar> invSqrtDiagHost;
    double scaledRowSumMax = 0.0;
    bool scaledRowSumReady = false;
    Foam::label amgApplyCountTotal = 0;
    Foam::scalar amgSetupMsTotal = 0;
    Foam::scalar amgApplyMsTotal = 0;

    // Preconditioner configuration
    const word preconditionerWord = dict.lookupOrDefault<word>("preconditioner", word("diagonal"));
    const std::string precondLower = toLower(preconditionerWord);

    bool useColour = false;
    bool useSGS = false;
    bool useILU = false;
    bool useIC = false;
    bool useComposite = false;
    bool useChebyshev = false;
    bool useAmg = false;

    const word forcePreconditionerWord =
        dict.lookupOrDefault<word>("forcePreconditioner", word::null);
    const std::string forcePreconLower = toLower(forcePreconditionerWord);

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
    else if (precondLower == "chebyshev" || precondLower == "cheb")
    {
        useChebyshev = true;
        chebyshevRequested = true;
    }
    else if (precondLower == "amg")
    {
        useAmg = true;
    }
    if (precondLower == "composite" || precondLower.find("colour+ic") != std::string::npos || precondLower.find("color+ic") != std::string::npos)
    {
        useComposite = true;
    }

    if (forcePreconLower == "sgs_spd")
    {
        useColour = true;
        useSGS = false;
        useILU = false;
        useIC = false;
        useComposite = false;
        colourSpdMode = true;
    }
    else if (forcePreconLower == "cheb_spd")
    {
        useColour = false;
        useSGS = false;
        useILU = false;
        useIC = false;
        useComposite = false;
        useChebyshev = true;
        chebyshevRequested = true;
        chebyshevForced = true;
    }
    else if
    (
        forcePreconLower == "ic0"
     || forcePreconLower == "ic"
     || forcePreconLower == "dic"
    )
    {
        useColour = false;
        useSGS = false;
        useILU = false;
        useComposite = false;
        useChebyshev = false;
        useIC = true;
        icForced = true;
    }

    colourInitiallyRequested = useColour;

    label polySweepsCfg = dict.lookupOrDefault<label>("polyJacobiSweeps", 0);
    scalar jacOmegaCfg = dict.lookupOrDefault<scalar>("jacobiOmega", scalar(0.7));
    const Switch polyAuto = dict.lookupOrDefault<Switch>("polyJacobiAuto", Switch(true));
    bool usePolyJacobi = (!useColour && !useSGS && !useILU && !useIC && !useChebyshev && !useAmg) && ((polySweepsCfg > 0) || polyAuto);
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
    if (colourSpdMode)
    {
        colourOmegaCfg = scalar(1);
        colourBackwardOmegaCfg = scalar(1);
    }
    const bool buildColour = useColour || colourSpdMode || useChebyshev;

    if (buildColour)
    {
        int colourStage = 0;
        if (colourSpdMode)
        {
            colourAutoTuneReset(fieldName_);
        }
        else if (useColour && autoTuneEnabled(dict))
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
        else if (useColour)
        {
            colourAutoTuneReset(fieldName_);
        }
        else
        {
            colourAutoTuneReset(fieldName_);
        }

        if (colourSpdMode)
        {
            colourOmegaCfg = scalar(1);
            colourBackwardOmegaCfg = scalar(1);
        }
    }
    else if (autoTuneEnabled(dict))
    {
        colourAutoTuneReset(fieldName_);
    }
    const int autoTuneStage = colourAutoTuneStage(fieldName_);

    const label colourBlockSizeCfg =
        std::max<label>
        (
            32,
            std::min<label>(1024, dict.lookupOrDefault<label>("colourBlockSize", 128))
        );
    const bool colourVerbose =
        dict.lookupOrDefault<Switch>("colourVerbose", Switch(false));

    scalar chebyshevLambdaMinFloor = scalar(1e-6);
    scalar chebyshevLambdaInflate = scalar(1.5);
    scalar chebyshevClampRatioLimit =
        dict.lookupOrDefault<scalar>("chebyshevInvSqrtClampRatio", scalar(0.01));
    label chebyshevDegree = 0;
    if (useChebyshev)
    {
        chebyshevLambdaMinFloor =
            dict.lookupOrDefault<scalar>("chebyshevLambdaMinFloor", scalar(1e-6));
        chebyshevLambdaInflate =
            dict.lookupOrDefault<scalar>("chebyshevLambdaInflate", scalar(1.5));
        chebyshevLambdaInflateUsed = chebyshevLambdaInflate;
        const label chebyshevDegreeDefault = (autoTuneStage <= 0) ? 2 : 3;
        chebyshevDegree = chebyshevDegreeDefault;
        chebyshevDegreeUsed = chebyshevDegree;
    }

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
            os<< "# chebRestarted," << (chebyshevRestarted ? "true" : "false") << nl;
            os<< "# chebClampGate," << (chebyshevClampGate ? "true" : "false") << nl;
            os<< "# invSqrtClampRatio," << invSqrtClampRatio << nl;
            os<< "# icDotGuard," << (icDotGuardTriggered ? "true" : "false") << nl;
            if (icEffectComputed)
            {
                os<< "# icPrecondEffect0," << icPrecondEffect0 << nl;
            }
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

        if (amgApplyCountTotal > 0 && reportStats)
        {
            const scalar avgApply = amgApplyCountTotal
                ? amgApplyMsTotal/static_cast<scalar>(amgApplyCountTotal)
                : scalar(0);
            Info<< "cudaPCG(" << fieldName_ << "): AMG applyCount="
                << amgApplyCountTotal
                << " setupMs=" << amgSetupMsTotal
                << " applyMs=" << amgApplyMsTotal
                << " avgApplyMs=" << avgApply << nl;
        }

        if (diagHistogramBinsCfg > 0 && !diagHistogramQuantiles.empty() && master)
        {
            fileName resolved = diagHistogramFile;
            if (resolved == fileName::null)
            {
                resolved = fileName("postProcessing/cudaPCG/" + fieldName_ + "_diag.csv");
            }
            resolved = resolvePath(resolved);
            const fileName parent = resolved.path();
            if (!parent.empty())
            {
                mkDir(parent);
            }
            OFstream os(resolved);
            os.setf(std::ios::scientific);
            os.precision(IOstream::defaultPrecision());
            os<< "# cudaPCG diagonal histogram for " << fieldName_ << nl;
            os<< "# success," << (success ? "true" : "false") << nl;
            os<< "# chebRestarted," << (chebyshevRestarted ? "true" : "false") << nl;
            os<< "# chebClampGate," << (chebyshevClampGate ? "true" : "false") << nl;
            os<< "# invSqrtClampRatio," << invSqrtClampRatio << nl;
            os<< "# icDotGuard," << (icDotGuardTriggered ? "true" : "false") << nl;
            if (icEffectComputed)
            {
                os<< "# icPrecondEffect0," << icPrecondEffect0 << nl;
            }
            os<< "# quantileFrac," << diagQuantileFracCfg << nl;
            os<< "# quantileAbs," << diagQuantileValue << nl;
            if (failureMessage.size())
            {
                os<< "# failure," << failureMessage << nl;
            }
            os<< "fraction,absDiag" << nl;
            for (label i = 0; i < static_cast<label>(diagHistogramQuantiles.size()); ++i)
            {
                const scalar fraction = diagHistogramBinsCfg
                    ? scalar(i)/scalar(diagHistogramBinsCfg)
                    : scalar(0);
                os<< fraction << ',' << diagHistogramQuantiles[i] << nl;
            }
            Info<< "cudaPCG(" << fieldName_ << "): diag histogram -> " << resolved << nl;
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
                << " spdMode=" << (colourSpdMode ? "true" : "false")
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
            os<< "nColours,minSize,maxSize,avgSize,diagFloor,omegaForward,omegaBackward,diagMin,diagMax,built,applied,activeFinal,disableCount,disableStages,colourSpd" << nl;
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
            os<< ',' << (colourSpdMode ? 1 : 0) << nl;
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
        if (iluBuf && !icBufferFromCache) cudaFree(iluBuf);
        if (iluInfo) cusparseDestroyCsrilu02Info(iluInfo);
        if (icInfo && !icInfoFromCache) cusparseDestroyCsric02Info(icInfo);
        if (iluDescr) cusparseDestroyMatDescr(iluDescr);
        if (d_iluVals && !d_iluValsFromCache) cudaFree(d_iluVals);
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

    if (precondDiag.size())
    {
        sqrtDiagHost.setSize(rows);
        invSqrtDiagHost.setSize(rows);
        double invSqrtClampEffective = icUseDiagonalScaling ? invSqrtClampUsed : 0.0;
        label invSqrtClampLocal = 0;
        if (icUseDiagonalScaling)
        {
            for (int i = 0; i < rows; ++i)
            {
                const double reg = static_cast<double>(precondDiag[i]);
                const double sqrtVal = reg > 0 ? std::sqrt(reg) : 1.0;
                sqrtDiagHost[i] = static_cast<DeviceScalar>(sqrtVal);
                double invSqrt = (sqrtVal > VSMALL) ? 1.0/sqrtVal : 0.0;
                if (invSqrt > invSqrtClampEffective)
                {
                    invSqrt = invSqrtClampEffective;
                    ++invSqrtClampLocal;
                }
                invSqrtDiagHost[i] = static_cast<DeviceScalar>(invSqrt);
            }
        }
        else
        {
            for (int i = 0; i < rows; ++i)
            {
                sqrtDiagHost[i] = DeviceScalar(1);
                invSqrtDiagHost[i] = DeviceScalar(1);
            }
        }

        if (!precondDiag.empty())
        {
            std::vector<double> diagTmp(precondDiag.size());
            for (label i = 0; i < precondDiag.size(); ++i)
            {
                diagTmp[i] = static_cast<double>(precondDiag[i]);
            }
            const std::size_t midIdx = diagTmp.size()/2;
            std::nth_element(diagTmp.begin(), diagTmp.begin() + midIdx, diagTmp.end());
            double medianScaled = diagTmp[midIdx];
            if (diagTmp.size() % 2 == 0 && midIdx > 0)
            {
                std::nth_element(diagTmp.begin(), diagTmp.begin() + midIdx - 1, diagTmp.end());
                medianScaled = 0.5*(medianScaled + diagTmp[midIdx - 1]);
            }
            icDiagMedianScaled = std::max(icDiagMedianScaled, std::abs(medianScaled));
        }

        std::size_t scaleId = hashCombine(0, hashDouble(static_cast<double>(diagFloorCfg)));
        scaleId = hashCombine(scaleId, hashDouble(invSqrtClampUsed));
        scaleId = hashCombine(scaleId, static_cast<std::size_t>(numFlooredDiag));
        scaleId = hashCombine(scaleId, static_cast<std::size_t>(invSqrtClampLocal));
        scaleId = hashCombine(scaleId, static_cast<std::size_t>(icUseDiagonalScaling ? 1 : 0));
        scaleId = hashCombine(scaleId, static_cast<std::size_t>(icUseRuizScaling ? 1 : 0));
        scaleId = hashCombine(scaleId, matvecSignD < 0.0 ? std::size_t(1) : std::size_t(0));

        PreconCacheKey cacheKey;
        cacheKey.matrixPatternId = matrixPatternId;
        cacheKey.scaleId = scaleId;
        cacheKey.permutationId = permutationId;
        cacheKey.deviceId = deviceId;
        if (useAmg)
        {
            cacheKey.kind = PreconCacheKind::AMG;
        }
        else if (useIC)
        {
            cacheKey.kind = PreconCacheKind::IC0;
        }
        else if (useILU)
        {
            cacheKey.kind = PreconCacheKind::ILU0;
        }
        else if (useColour)
        {
            cacheKey.kind = PreconCacheKind::Colour;
        }
        else if (useSGS)
        {
            cacheKey.kind = PreconCacheKind::SGS;
        }
        else if (useChebyshev)
        {
            cacheKey.kind = PreconCacheKind::Chebyshev;
        }
        else
        {
            cacheKey.kind = PreconCacheKind::Diagonal;
        }

        preconCacheEntry = &getPreconCacheEntry(cacheKey);
        if (std::fabs(preconCacheEntry->cachedMatvecSign - matvecSignD) > 0.5)
        {
            preconCacheEntry->resetIcReuseState();
        }
        preconCacheEntry->cachedMatvecSign = matvecSignD;
        if (!preconCacheEntry->ensureCapacity(rows))
        {
            return fail("preconCacheAlloc");
        }
        preconCacheEntry->icUsedDiagonalScaling = icUseDiagonalScaling;
        preconCacheEntry->icUsedRuizScaling = icUseRuizScaling;

        preconCacheEntry->ruizReady = false;
        preconCacheEntry->ruizClampRatio = 0.0;
        preconCacheEntry->ruizScaleMin = 1.0;
        preconCacheEntry->ruizScaleMax = 1.0;
        if (icUseRuizScaling)
        {
            Foam::List<double> ruizValues(nnz);
            for (int idx = 0; idx < nnz; ++idx)
            {
                ruizValues[idx] = static_cast<double>(values[idx]);
            }
            std::vector<double> ruizScale;
            computeRuizScale(rowPtr, colInd, ruizValues, ruizScale);
            if (!ruizScale.empty() && preconCacheEntry->d_ruizScale && preconCacheEntry->d_ruizInvScale)
            {
                const double ruizClampFloor = 1e-6;
                const double ruizClampCeil = 1e6;
                Foam::label clampCount = 0;
                List<DeviceScalar> ruizScaleHost(rows);
                List<DeviceScalar> ruizInvScaleHost(rows);
                for (int i = 0; i < rows; ++i)
                {
                    double s = ruizScale[i];
                    if (s < ruizClampFloor)
                    {
                        s = ruizClampFloor;
                        ++clampCount;
                    }
                    else if (s > ruizClampCeil)
                    {
                        s = ruizClampCeil;
                        ++clampCount;
                    }
                    preconCacheEntry->ruizScaleMin = std::min(preconCacheEntry->ruizScaleMin, s);
                    preconCacheEntry->ruizScaleMax = std::max(preconCacheEntry->ruizScaleMax, s);
                    ruizScaleHost[i] = static_cast<DeviceScalar>(s);
                    const double invS = s > ruizClampFloor ? 1.0/s : 1.0;
                    ruizInvScaleHost[i] = static_cast<DeviceScalar>(invS);
                    const double invDiag = static_cast<double>(invSqrtDiagHost[i]);
                    invSqrtDiagHost[i] = static_cast<DeviceScalar>(invDiag * s);
                }
        preconCacheEntry->ruizClampRatio = rows > 0 ? static_cast<double>(clampCount)/static_cast<double>(rows) : 0.0;
        if
        (
            cudaMemcpy
            (
                        preconCacheEntry->d_ruizScale,
                        ruizScaleHost.begin(),
                        sizeof(DeviceScalar)*rows,
                        cudaMemcpyHostToDevice
                    ) != cudaSuccess
                 || cudaMemcpy
                    (
                        preconCacheEntry->d_ruizInvScale,
                        ruizInvScaleHost.begin(),
                        sizeof(DeviceScalar)*rows,
                        cudaMemcpyHostToDevice
                    ) != cudaSuccess
                )
                {
                    return fail("ruizScaleCopy");
                }
                preconCacheEntry->ruizReady = true;
            }
        }

        if
        (
            cudaMemcpy
            (
                preconCacheEntry->d_sqrtDiag,
                sqrtDiagHost.begin(),
                sizeof(DeviceScalar)*rows,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
         || cudaMemcpy
            (
                preconCacheEntry->d_invSqrtDiag,
                invSqrtDiagHost.begin(),
                sizeof(DeviceScalar)*rows,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        )
        {
            return fail("preconCacheCopy");
        }

        preconCacheEntry->flooredCount = numFlooredDiag;
        preconCacheEntry->diagFloorApplied = diagFloorCfg;
        preconCacheEntry->invSqrtClampCount = invSqrtClampLocal;
        preconCacheEntry->invSqrtClampLimit = invSqrtClampEffective;
        preconCacheEntry->lambdaReady = false;
        preconCacheEntry->lambdaMin = 0.0;
        preconCacheEntry->lambdaMax = 0.0;
        invSqrtClamped = invSqrtClampLocal;
        invSqrtClampUsed = invSqrtClampEffective;
        if (!sqrtDiagHost.empty())
        {
            scaledRowSumMax = 0.0;
            for (int row = 0; row < rows; ++row)
            {
                const double si = (row < sqrtDiagHost.size() && sqrtDiagHost[row] > VSMALL)
                    ? static_cast<double>(sqrtDiagHost[row])
                    : 1.0;
                double rowSum = 0.0;
                for (int k = rowPtr[row]; k < rowPtr[row+1]; ++k)
                {
                    const int col = colInd[k];
                    const double sj = (col < sqrtDiagHost.size() && sqrtDiagHost[col] > VSMALL)
                        ? static_cast<double>(sqrtDiagHost[col])
                        : 1.0;
                    const double scaledVal = std::abs(static_cast<double>(values[k])/(si*sj));
                    rowSum += scaledVal;
                }
                scaledRowSumMax = std::max(scaledRowSumMax, rowSum);
            }
            scaledRowSumReady = true;
        }

        if (diagSnapshotEnable && Pstream::master())
        {
            const Time& runTime = matrix_.mesh().thisDb().time();
            auto resolveSnapshotPath = [&](const fileName& fName) -> fileName
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

            auto quantileFromSorted = [](const auto& sortedVec, double fraction) -> double
            {
                if (sortedVec.empty())
                {
                    return 0.0;
                }
                fraction = std::max(0.0, std::min(1.0, fraction));
                const double position = fraction * static_cast<double>(sortedVec.size() - 1);
                const std::size_t lowerIdx = static_cast<std::size_t>(std::floor(position));
                const std::size_t upperIdx = static_cast<std::size_t>(std::ceil(position));
                const double weight = position - static_cast<double>(lowerIdx);
                const double lowerVal = static_cast<double>(sortedVec[lowerIdx]);
                const double upperVal = static_cast<double>(sortedVec[upperIdx]);
                return (1.0 - weight)*lowerVal + weight*upperVal;
            };

            const double q001 = quantileFromSorted(absDiagVals, 0.001);
            const double q005 = quantileFromSorted(absDiagVals, 0.005);
            const double q01 = quantileFromSorted(absDiagVals, 0.01);
            const double q02 = quantileFromSorted(absDiagVals, 0.02);
            const double q05 = quantileFromSorted(absDiagVals, 0.05);
            const double q50 = static_cast<double>(medianAbsDiag);

            std::vector<double> invSqrtVals(rows, 0.0);
            for (int i = 0; i < rows; ++i)
            {
                const double reg = std::max(static_cast<double>(precondDiag[i]), 0.0);
                const double sqrtVal = reg > 0 ? std::sqrt(reg) : 1.0;
                invSqrtVals[i] = (sqrtVal > VSMALL) ? 1.0/sqrtVal : 0.0;
            }
            std::vector<double> invSqrtSorted(invSqrtVals);
            std::sort(invSqrtSorted.begin(), invSqrtSorted.end());
            const double clampTarget = std::max(0.0, std::min(1.0, static_cast<double>(diagSnapshotClampRatioCfg)));
            const double clampQuantile = clampTarget < 1.0 ? 1.0 - clampTarget : 0.99;
            double clampCandidate = invSqrtSorted.empty()
                ? 0.0
                : quantileFromSorted(invSqrtSorted, clampQuantile);
            if (invSqrtClampMaxCfg > scalar(0))
            {
                clampCandidate = std::min(clampCandidate, static_cast<double>(invSqrtClampMaxCfg));
            }
            label clampCountCandidate = 0;
            for (double val : invSqrtVals)
            {
                if (val > clampCandidate)
                {
                    ++clampCountCandidate;
                }
            }

            std::vector<double> alphaList = {0.5, 0.7, 1.0};
            const double alphaCfg = static_cast<double>(diagQuantileAlphaCfg);
            if (alphaCfg > 0.0 && std::find(alphaList.begin(), alphaList.end(), alphaCfg) == alphaList.end())
            {
                alphaList.push_back(alphaCfg);
            }
            std::sort(alphaList.begin(), alphaList.end());

            std::vector<std::tuple<double, double, double>> floorSummary;
            for (double alpha : alphaList)
            {
                const double candidate =
                    std::max
                    (
                        {
                            static_cast<double>(diagFloorAbsCfg),
                            static_cast<double>(diagFloorRelCfg)*static_cast<double>(medianAbsDiag),
                            alpha*q01
                        }
                    );
                label flooredCountCandidate = 0;
                forAll(diag, cellI)
                {
                    if (mag(diag[cellI]) < candidate)
                    {
                        ++flooredCountCandidate;
                    }
                }
                const double flooredPctCandidate =
                    rows > 0 ? static_cast<double>(flooredCountCandidate)/static_cast<double>(rows) : 0.0;
                floorSummary.emplace_back(alpha, candidate, flooredPctCandidate);
            }

            fileName snapshotPath = diagSnapshotFileCfg;
            if (snapshotPath == fileName::null)
            {
                snapshotPath = fileName("postProcessing/cudaPCG/" + fieldName_ + "_diagSnapshot.csv");
            }
            snapshotPath = resolveSnapshotPath(snapshotPath);
            const fileName snapshotParent = snapshotPath.path();
            if (!snapshotParent.empty())
            {
                mkDir(snapshotParent);
            }
            OFstream snapshot(snapshotPath);
            snapshot.setf(std::ios::scientific);
            snapshot.precision(IOstream::defaultPrecision());
            snapshot<< "# cudaPCG diagonal snapshot for " << fieldName_ << nl;
            snapshot<< "# time," << runTime.value() << nl;
            snapshot<< "# rows," << rows << nl;
            snapshot<< "# diagFloorAbsCfg," << diagFloorAbsCfg << nl;
            snapshot<< "# diagFloorRelCfg," << diagFloorRelCfg << nl;
            snapshot<< "# diagQuantileFracCfg," << diagQuantileFracCfg << nl;
            snapshot<< "# diagQuantileAlphaCfg," << diagQuantileAlphaCfg << nl;
            snapshot<< "# invSqrtClampCfg," << invSqrtClampCfg << nl;
            snapshot<< "# invSqrtClampSafetyCfg," << invSqrtClampSafetyCfg << nl;
            snapshot<< "# invSqrtClampMaxCfg," << invSqrtClampMaxCfg << nl;
            snapshot<< "# medianAbs," << q50 << nl;
            snapshot<< "# quantile0.1pct," << q001 << nl;
            snapshot<< "# quantile0.5pct," << q005 << nl;
            snapshot<< "# quantile1pct," << q01 << nl;
            snapshot<< "# quantile2pct," << q02 << nl;
            snapshot<< "# quantile5pct," << q05 << nl;
            snapshot<< "# flooredCountCurrent," << numFlooredDiag << nl;
            const double flooredPctCurrent =
                rows > 0 ? static_cast<double>(numFlooredDiag)/static_cast<double>(rows) : 0.0;
            snapshot<< "# flooredPctCurrent," << flooredPctCurrent << nl;
            snapshot<< "# clampCountCurrent," << invSqrtClamped << nl;
            const double clampPctCurrent =
                rows > 0 ? static_cast<double>(invSqrtClamped)/static_cast<double>(rows) : 0.0;
            snapshot<< "# clampPctCurrent," << clampPctCurrent << nl;
            snapshot<< "# clampCandidate," << clampCandidate << nl;
            const double clampPctCandidate =
                rows > 0 ? static_cast<double>(clampCountCandidate)/static_cast<double>(rows) : 0.0;
            snapshot<< "# clampPctCandidate," << clampPctCandidate << nl;
            snapshot<< "# clampTarget," << clampTarget << nl;
            for (const auto& entry : floorSummary)
            {
                const double alpha = std::get<0>(entry);
                const double candidate = std::get<1>(entry);
                const double flooredPct = std::get<2>(entry);
                snapshot<< "# floorCandidate_alpha=" << alpha << ',' << candidate
                         << ",flooredPct," << flooredPct << nl;
            }
            snapshot<< "index,diag,absDiag,invSqrt" << nl;
            for (int i = 0; i < rows; ++i)
            {
                const double diagValue = static_cast<double>(diag[i]);
                snapshot<< i << ','
                         << diagValue << ','
                         << std::abs(diagValue) << ','
                         << invSqrtVals[i] << nl;
            }
            Info<< "cudaPCG(" << fieldName_ << "): diag snapshot -> " << snapshotPath << nl;
        }
    }

    invSqrtClampRatio =
        rows > 0 ? static_cast<scalar>(invSqrtClamped)/static_cast<scalar>(rows) : scalar(0);
    scalar clampRatioForGate = invSqrtClampRatio;
    if (preconCacheEntry && preconCacheEntry->ruizReady)
    {
        clampRatioForGate = max
        (
            clampRatioForGate,
            static_cast<scalar>(preconCacheEntry->ruizClampRatio)
        );
    }
    if
    (
        useChebyshev
     && chebyshevClampRatioLimit >= scalar(0)
     && clampRatioForGate > chebyshevClampRatioLimit
    )
    {
        chebyshevClampGuardTriggered = true;
        chebyshevGuardrailTriggered = true;
        chebyshevClampGate = true;
        useChebyshev = false;
        colourSpdMode = true;
        useColour = true;
        chebyshevPrevResidual = -1;
        chebyshevGrowthCounter = 0;
    }

    if (reportStats)
    {
        Info<< "cudaPCG: diagonal range [" << minDiag << ", " << maxDiag << "]"
            << " (regularised [" << minDiagReg << ", " << maxDiagReg << "])"
            << " medianAbs=" << medianAbsDiag
            << " quantileFrac=" << diagQuantileFracCfg
            << " quantileAbs=" << diagQuantileValue
            << " floorUsed=" << diagFloorCfg
            << " flooredCount=" << numFlooredDiag
            << " invSqrtClampLimit=" << invSqrtClampUsed
            << " invSqrtClampCount=" << invSqrtClamped
            << " clampRatio=" << invSqrtClampRatio;
        if (preconCacheEntry && preconCacheEntry->ruizReady)
        {
            Info<< " ruizClampRatio=" << preconCacheEntry->ruizClampRatio
                << " ruizScale[min,max]=[" << preconCacheEntry->ruizScaleMin
                << ',' << preconCacheEntry->ruizScaleMax << ']';
        }
        Info<< nl;
    }

    List<DeviceScalar> matrixValsHost(nnz);
    List<DeviceScalar> spmvValsHost(nnz);
    List<DeviceScalar> factorValsHost(nnz);
    const bool scaleIcFactors =
        (icUseDiagonalScaling || icUseRuizScaling)
     && (invSqrtDiagHost.size() == rows);
    std::vector<double> factorDiag(rows, 0.0);
    auto populateOperatorArrays =
        [&]()
    {
        for (int row = 0; row < rows; ++row)
        {
            const double sRow = scaleIcFactors ? static_cast<double>(invSqrtDiagHost[row]) : 1.0;
            for (int k = rowPtr[row]; k < rowPtr[row+1]; ++k)
            {
                const int col = colInd[k];
                const double sCol = scaleIcFactors ? static_cast<double>(invSqrtDiagHost[col]) : 1.0;
                const double baseValPhysical = static_cast<double>(values[k]);
                matrixValsHost[k] = static_cast<DeviceScalar>(baseValPhysical);
                double factVal = matvecSignD * baseValPhysical;
                if (scaleIcFactors)
                {
                    factVal *= sRow * sCol;
                }
                factorValsHost[k] = static_cast<DeviceScalar>(factVal);
                spmvValsHost[k] = static_cast<DeviceScalar>(matvecSignD * baseValPhysical);
                if (col == row)
                {
                    factorDiag[row] = factVal;
                }
            }
        }
    };
    populateOperatorArrays();
    double factorScale = 1.0;
    double factorMedianDiag = 0.0;
    double factorScaleApplied = 1.0;
    auto computeMedian = [](std::vector<double>& data) -> double
    {
        if (data.empty())
        {
            return 0.0;
        }
        const std::size_t mid = data.size()/2;
        std::nth_element(data.begin(), data.begin() + mid, data.end());
        double median = data[mid];
        if (data.size() % 2 == 0 && mid > 0)
        {
            std::nth_element(data.begin(), data.begin() + mid - 1, data.end());
            median = 0.5*(median + data[mid - 1]);
        }
        return median;
    };
    if (scaleIcFactors || permutationActive || icNormalizeFactor)
    {
        std::vector<double> positiveDiag;
        positiveDiag.reserve(rows);
        double factorTightMedian = 0.0;
        for (double d : factorDiag)
        {
            const double absVal = std::fabs(d);
            if (absVal > 0.0)
            {
                positiveDiag.push_back(absVal);
            }
        }
        double factorScaleRaw = factorScale;
        if (!positiveDiag.empty())
        {
            factorMedianDiag = computeMedian(positiveDiag);
            const double medianSafe = std::max(factorMedianDiag, 1e-18);
            const double scaleRoot = 1.0/std::sqrt(medianSafe);
            factorScale = scaleRoot;
            factorScaleRaw = factorScale;
        }
        if (!std::isfinite(factorScale) || factorScale <= 0.0)
        {
            factorScale = 1.0;
        }
        const double normClampMin =
            icNormalizeClampMinCfg > scalar(0)
              ? std::max(1e-12, static_cast<double>(icNormalizeClampMinCfg))
              : 0.0;
        const double normClampMax =
            icNormalizeClampMaxCfg > scalar(0)
              ? static_cast<double>(icNormalizeClampMaxCfg)
              : std::numeric_limits<double>::infinity();
        if (normClampMax > 0.0 && factorScale > normClampMax)
        {
            factorScale = normClampMax;
        }
        if (normClampMin > 0.0 && factorScale < normClampMin)
        {
            factorScale = normClampMin;
        }
        factorScaleApplied = factorScale;
        const bool scaleApplied = std::fabs(factorScale - 1.0) > 1e-12;
        if (scaleApplied)
        {
            for (int k = 0; k < nnz; ++k)
            {
                const double scaled = static_cast<double>(factorValsHost[k]) * factorScale;
                factorValsHost[k] = static_cast<DeviceScalar>(scaled);
            }
            for (double& d : factorDiag)
            {
                d *= factorScale;
            }
        }
        else if (positiveDiag.empty() && reportStats && (!Pstream::parRun() || Pstream::master()))
        {
            Info<< "cudaPCG(" << fieldName_ << "): factor diag contains no positive entries; "
                << "skipping scaling" << nl;
        }
        if (reportStats && (!Pstream::parRun() || Pstream::master()))
        {
            Info<< "cudaPCG(" << fieldName_ << "): IC normalization median=" << factorMedianDiag
                << " alphaRaw=" << factorScaleRaw
                << " alphaApplied=" << factorScaleApplied
                << " clamp=[";
            if (normClampMin > 0.0)
            {
                Info<< normClampMin;
            }
            else
            {
                Info<< "off";
            }
            Info<< ',';
            if (std::isfinite(normClampMax))
            {
                Info<< normClampMax;
            }
            else
            {
                Info<< "off";
            }
            Info<< ']' << nl;
        }
    }
    if (icDebugDump && reportStats && (!Pstream::parRun() || Pstream::master()))
    {
        double diagMin = std::numeric_limits<double>::max();
        double diagMax = -std::numeric_limits<double>::max();
        for (double d : factorDiag)
        {
            if (d == 0.0)
            {
                continue;
            }
            diagMin = std::min(diagMin, d);
            diagMax = std::max(diagMax, d);
        }
        if (diagMin <= diagMax)
        {
            Info<< "cudaPCG(" << fieldName_ << "): IC factor diag median=" << factorMedianDiag
                << " scale=" << factorScaleApplied
                << " diag[min,max]=[" << diagMin << ',' << diagMax << ']' << nl;
        }
    }

    List<int> factorRowPtrPerm;
    List<int> factorColIndPerm;
    List<DeviceScalar> factorValsPerm;
    if (permutationActive)
    {
        factorRowPtrPerm.setSize(rows + 1);
        factorColIndPerm.setSize(nnz);
        factorValsPerm.setSize(nnz);
        std::vector<double> factorDiagPerm(rows, 0.0);
        label cursor = 0;
        factorRowPtrPerm[0] = 0;
        std::vector<std::pair<int, double>> rowEntries;
        for (Foam::label newRow = 0; newRow < rows; ++newRow)
        {
            const Foam::label origRow = permutation[newRow];
            rowEntries.clear();
            const int start = rowPtr[origRow];
            const int end = rowPtr[origRow + 1];
            rowEntries.reserve(end - start);
            for (int kk = start; kk < end; ++kk)
            {
                const int origCol = colInd[kk];
                const int newCol = permutationInverse[origCol];
                rowEntries.emplace_back(newCol, static_cast<double>(factorValsHost[kk]));
            }
            std::sort
            (
                rowEntries.begin(),
                rowEntries.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b)
                {
                    return a.first < b.first;
                }
            );
            for (const auto& entry : rowEntries)
            {
                factorColIndPerm[cursor] = entry.first;
                factorValsPerm[cursor] = static_cast<DeviceScalar>(entry.second);
                if (entry.first == newRow)
                {
                    factorDiagPerm[newRow] = entry.second;
                }
                ++cursor;
            }
            factorRowPtrPerm[newRow + 1] = cursor;
        }
        factorDiag = factorDiagPerm;

        bool permStructureOk = true;
        double permMaxDiff = 0.0;
        if (preconCacheEntry)
        {
            std::unordered_set<Foam::label> sampledRows;
            std::mt19937 rng(static_cast<unsigned>
                (
                    permutationId
                  ? permutationId
                  : hashCombine(matrixPatternId, static_cast<std::size_t>(rows))
                ));
            const int samples = std::min(rows, 5);
            std::uniform_int_distribution<Foam::label> rowDist(0, rows - 1);
            for (int sample = 0; sample < samples && permStructureOk; ++sample)
            {
                Foam::label newRow = sample;
                if (rows > samples)
                {
                    do
                    {
                        newRow = rowDist(rng);
                    }
                    while (!sampledRows.insert(newRow).second);
                }
                else
                {
                    sampledRows.insert(newRow);
                }

                const Foam::label origRow = permutation[newRow];
                const int origStart = rowPtr[origRow];
                const int origEnd = rowPtr[origRow + 1];
                const int permStart = factorRowPtrPerm[newRow];
                const int permEnd = factorRowPtrPerm[newRow + 1];
                if ((origEnd - origStart) != (permEnd - permStart))
                {
                    permStructureOk = false;
                    break;
                }
                for (int kk = origStart; kk < origEnd; ++kk)
                {
                    const int expectedCol = permutationInverse[colInd[kk]];
                    const double expectedVal = static_cast<double>(factorValsHost[kk]);
                    bool found = false;
                    for (int jj = permStart; jj < permEnd; ++jj)
                    {
                        if (factorColIndPerm[jj] == expectedCol)
                        {
                            const double actualVal = static_cast<double>(factorValsPerm[jj]);
                            const double diff = std::fabs(actualVal - expectedVal);
                            permMaxDiff = std::max(permMaxDiff, diff);
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        permStructureOk = false;
                        break;
                    }
                }
            }
        }

        if (!permStructureOk)
        {
            return fail("icPermutationAssemble");
        }
        if (preconCacheEntry)
        {
            preconCacheEntry->permutationStructureValidated = true;
        }
        if
        (
            preconCacheEntry
         && reportStats
         && (!Pstream::parRun() || Pstream::master())
        )
        {
            Info<< "cudaPCG(" << fieldName_ << "): permutation CSR verification max|Δ|="
                << permMaxDiff << nl;
        }
    }

    const List<int>* factorRowPtrPtr =
        permutationActive ? &factorRowPtrPerm : &rowPtr;
    const List<int>* factorColIndPtr =
        permutationActive ? &factorColIndPerm : &colInd;
    const List<DeviceScalar>* factorValsNumericPtr =
        permutationActive ? &factorValsPerm : &factorValsHost;

    const List<int>& factorRowPtrHost = *factorRowPtrPtr;
    const List<int>& factorColIndHost = *factorColIndPtr;
    const List<DeviceScalar>& factorValsNumeric = *factorValsNumericPtr;

    if (permutationActive)
    {
        if (!preconCacheEntry->setPermutationData(permutation))
        {
            return fail("icPermutationAlloc");
        }

        Foam::labelList rowPtrLabels(factorRowPtrHost.size());
        forAll(rowPtrLabels, i)
        {
            rowPtrLabels[i] = factorRowPtrHost[i];
        }
        Foam::labelList colIndLabels(factorColIndHost.size());
        forAll(colIndLabels, i)
        {
            colIndLabels[i] = factorColIndHost[i];
        }
        if (!preconCacheEntry->setIcPatternData(rowPtrLabels, colIndLabels))
        {
            return fail("icPermutationPattern");
        }
        preconCacheEntry->permutationStructureValidated = true;
    }
    else
    {
        preconCacheEntry->permutationReady = false;
        preconCacheEntry->icPatternReady = false;
        preconCacheEntry->permutationValidated = false;
        preconCacheEntry->permutationStructureValidated = false;
    }

    auto scaledSpMV =
        [&](double* yScaled, const double* xScaled) -> bool
    {
        if (!preconCacheEntry || !preconCacheEntry->d_invSqrtDiag)
        {
            message = word("scaledSpMVNoCache");
            return false;
        }
        double* tmpVec = preconCacheEntry->d_scaledTmp;
        if (!tmpVec)
        {
            message = word("scaledSpMVNoTmp");
            return false;
        }
        if (cublasDcopy(cublasHandle, rows, xScaled, 1, tmpVec, 1) != CUBLAS_STATUS_SUCCESS)
        {
            message = word("scaledSpMVCopy");
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
                tmpVec,
                rows,
                preconCacheEntry->d_invSqrtDiag,
                1,
                tmpVec,
                rows
            ) != CUBLAS_STATUS_SUCCESS
        )
        {
            message = word("scaledSpMVScaleIn");
            return false;
        }

        cusparseDnVecDescr_t localVecX = nullptr;
        cusparseDnVecDescr_t localVecY = nullptr;
        if (cusparseCreateDnVec(&localVecX, rows, tmpVec, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        {
            message = word("scaledSpMVCreateX");
            return false;
        }
        if (cusparseCreateDnVec(&localVecY, rows, yScaled, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        {
            cusparseDestroyDnVec(localVecX);
            message = word("scaledSpMVCreateY");
            return false;
        }

        const double oneD = 1.0;
        const double zeroD = 0.0;
        cusparseStatus_t spmvStatus =
            cusparseSpMV
            (
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &oneD,
                matA,
                localVecX,
                &zeroD,
                localVecY,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                spmvBuffer
            );
        cusparseDestroyDnVec(localVecX);
        cusparseDestroyDnVec(localVecY);
        if (spmvStatus != CUSPARSE_STATUS_SUCCESS)
        {
            message = word("scaledSpMVSpmv");
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
                yScaled,
                rows,
                preconCacheEntry->d_invSqrtDiag,
                1,
                yScaled,
                rows
            ) != CUBLAS_STATUS_SUCCESS
        )
        {
            message = word("scaledSpMVScaleOut");
            return false;
        }
        return true;
    };

    auto ensureChebyshevSpectrum = [&]() -> bool
    {
        if (!useChebyshev)
        {
            return true;
        }
        if
        (
            !preconCacheEntry
         || !preconCacheEntry->d_scaledResidual
         || !preconCacheEntry->d_scaledAp
         || !preconCacheEntry->d_invSqrtDiag
         || !preconCacheEntry->d_scaledTmp
        )
        {
            message = word("ChebCacheIncomplete");
            return false;
        }
        if (preconCacheEntry->lambdaReady && preconCacheEntry->lambdaMax > VSMALL)
        {
            chebyshevLambdaMinUsed = static_cast<scalar>(preconCacheEntry->lambdaMin);
            chebyshevLambdaMaxUsed = static_cast<scalar>(preconCacheEntry->lambdaMax);
            return true;
        }

        List<DeviceScalar> onesHost(rows, DeviceScalar(1));
        if
        (
            cudaMemcpy
            (
                preconCacheEntry->d_scaledResidual,
                onesHost.begin(),
                sizeof(DeviceScalar)*rows,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        )
        {
            message = word("ChebSpectrumInit");
            return false;
        }

        double lambdaEstimate = 0.0;
        bool powerOk = true;
        const int maxPowerIter = 12;
        for (int iter = 0; iter < maxPowerIter && powerOk; ++iter)
        {
            if (!scaledSpMV(preconCacheEntry->d_scaledAp, preconCacheEntry->d_scaledResidual))
            {
                powerOk = false;
                break;
            }

            double vTw = 0.0;
            double vTv = 0.0;
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    preconCacheEntry->d_scaledResidual,
                    1,
                    preconCacheEntry->d_scaledAp,
                    1,
                    &vTw
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                powerOk = false;
                break;
            }
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    preconCacheEntry->d_scaledResidual,
                    1,
                    preconCacheEntry->d_scaledResidual,
                    1,
                    &vTv
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                powerOk = false;
                break;
            }
            if (vTv > VSMALL)
            {
                lambdaEstimate = vTw / vTv;
            }

            double normW = 0.0;
            if
            (
                cublasDnrm2
                (
                    cublasHandle,
                    rows,
                    preconCacheEntry->d_scaledAp,
                    1,
                    &normW
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                powerOk = false;
                break;
            }
            if (normW > VSMALL)
            {
                const double invNorm = 1.0/normW;
                if
                (
                    cublasDcopy
                    (
                        cublasHandle,
                        rows,
                        preconCacheEntry->d_scaledAp,
                        1,
                        preconCacheEntry->d_scaledResidual,
                        1
                    ) != CUBLAS_STATUS_SUCCESS
                )
                {
                    powerOk = false;
                    break;
                }
                if
                (
                    cublasDscal
                    (
                        cublasHandle,
                        rows,
                        &invNorm,
                        preconCacheEntry->d_scaledResidual,
                        1
                    ) != CUBLAS_STATUS_SUCCESS
                )
                {
                    powerOk = false;
                    break;
                }
            }
        }

        double lambdaMaxCandidate = lambdaEstimate * 1.2;
        if (!powerOk || !std::isfinite(lambdaEstimate) || lambdaEstimate <= VSMALL)
        {
            if (scaledRowSumReady && std::isfinite(scaledRowSumMax) && scaledRowSumMax > VSMALL)
            {
                lambdaMaxCandidate = 1.5 * scaledRowSumMax;
            }
            else
            {
                lambdaMaxCandidate = 1.0;
            }
        }
        if (!std::isfinite(lambdaMaxCandidate) || lambdaMaxCandidate <= VSMALL)
        {
            lambdaMaxCandidate = 1.0;
        }

        preconCacheEntry->lambdaMax = lambdaMaxCandidate;
        preconCacheEntry->lambdaMin = std::max
        (
            std::max(0.05*lambdaMaxCandidate, static_cast<double>(chebyshevLambdaMinFloor)),
            static_cast<double>(chebyshevLambdaMinFloor)
        );
        if (preconCacheEntry->lambdaMin >= preconCacheEntry->lambdaMax)
        {
            preconCacheEntry->lambdaMin = std::max
            (
                static_cast<double>(chebyshevLambdaMinFloor),
                0.5*preconCacheEntry->lambdaMax
            );
        }
        preconCacheEntry->lambdaReady = true;

        chebyshevLambdaMinUsed = static_cast<scalar>(preconCacheEntry->lambdaMin);
        chebyshevLambdaMaxUsed = static_cast<scalar>(preconCacheEntry->lambdaMax);

        if (reportStats)
        {
            Info<< "cudaPCG(" << fieldName_ << "): Chebyshev spectral window ["
                << preconCacheEntry->lambdaMin << ", " << preconCacheEntry->lambdaMax << ']' << nl;
        }
        return true;
    };

    auto applyChebyshevSpd =
        [&](const double* rhs, double* out) -> bool
    {
        if
        (
            !preconCacheEntry
         || !preconCacheEntry->d_scaledResidual
         || !preconCacheEntry->d_scaledDirection
         || !preconCacheEntry->d_scaledAp
         || !preconCacheEntry->d_invSqrtDiag
        )
        {
            message = word("ChebCacheMissing");
            return false;
        }
        if (!ensureChebyshevSpectrum())
        {
            return false;
        }

        const double lambdaMax = preconCacheEntry->lambdaMax;
        const double lambdaMin = preconCacheEntry->lambdaMin;
        const double theta = 0.5*(lambdaMax + lambdaMin);
        const double delta = 0.5*(lambdaMax - lambdaMin);

        double alpha = 1.0/theta;
        double alphaPrev = alpha;
        double beta = 0.0;
        const double oneD = 1.0;

        double* rScaled = preconCacheEntry->d_scaledResidual;
        double* pScaled = preconCacheEntry->d_scaledDirection;
        double* ApScaled = preconCacheEntry->d_scaledAp;

        if (cublasDcopy(cublasHandle, rows, rhs, 1, rScaled, 1) != CUBLAS_STATUS_SUCCESS)
        {
            message = word("ChebCopyResidual");
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
                rScaled,
                rows,
                preconCacheEntry->d_invSqrtDiag,
                1,
                rScaled,
                rows
            ) != CUBLAS_STATUS_SUCCESS
        )
        {
            message = word("ChebScaleResidual");
            return false;
        }

        if (cudaMemset(pScaled, 0, sizeof(double)*rows) != cudaSuccess)
        {
            message = word("ChebZeroDirection");
            return false;
        }
        if (cudaMemset(out, 0, sizeof(DeviceScalar)*rows) != cudaSuccess)
        {
            message = word("ChebZeroOut");
            return false;
        }

        for (label iter = 0; iter < chebyshevDegree; ++iter)
        {
            if (iter == 0)
            {
                beta = 0.0;
                alpha = 1.0/theta;
            }
            else
            {
                beta = std::pow(delta * alphaPrev * 0.5, 2);
                double denom = theta - beta/alphaPrev;
                if (std::fabs(denom) <= VSMALL)
                {
                    denom = theta;
                }
                alpha = 1.0/denom;
            }

            if (beta != 0.0)
            {
                if (cublasDscal(cublasHandle, rows, &beta, pScaled, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("ChebScaleP");
                    return false;
                }
                if (cublasDaxpy(cublasHandle, rows, &oneD, rScaled, 1, pScaled, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("ChebUpdateP");
                    return false;
                }
            }
            else
            {
                if (cublasDcopy(cublasHandle, rows, rScaled, 1, pScaled, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("ChebCopyP");
                    return false;
                }
            }

            if (!scaledSpMV(ApScaled, pScaled))
            {
                return false;
            }

            if (cublasDaxpy(cublasHandle, rows, &alpha, pScaled, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("ChebUpdateZ");
                return false;
            }
            const double negAlpha = -alpha;
            if (cublasDaxpy(cublasHandle, rows, &negAlpha, ApScaled, 1, rScaled, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("ChebUpdateResidual");
                return false;
            }

            alphaPrev = alpha;
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
                preconCacheEntry->d_invSqrtDiag,
                1,
                out,
                rows
            ) != CUBLAS_STATUS_SUCCESS
        )
        {
            message = word("ChebUnscale");
            return false;
        }

        double zNorm = 0.0;
        if (cublasDnrm2(cublasHandle, rows, out, 1, &zNorm) != CUBLAS_STATUS_SUCCESS)
        {
            message = word("ChebNormFail");
            return false;
        }
        double rzCheck = 0.0;
        if (cublasDdot(cublasHandle, rows, rhs, 1, out, 1, &rzCheck) != CUBLAS_STATUS_SUCCESS)
        {
            message = word("ChebDotFail");
            return false;
        }
        if (!std::isfinite(zNorm) || !std::isfinite(rzCheck) || zNorm <= 0 || rzCheck <= 0)
        {
            chebyshevGuardrailTriggered = true;
            cudaMemset(out, 0, sizeof(DeviceScalar)*rows);
            return false;
        }

        chebyshevSpdApplied = true;
        return true;
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

    if (equilibrateMatrix)
    {
        populateOperatorArrays();
    }

    // Copy values
    if (cudaMemcpy(d_vals, spmvValsHost.begin(), sizeof(DeviceScalar)*nnz, cudaMemcpyHostToDevice) != cudaSuccess)
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
            const double psiVal = static_cast<double>(psi[i]) * static_cast<double>(invScaleHost[i]);
            const double rhsVal = static_cast<double>(source[i]) * static_cast<double>(scaleHost[i]);
            xScaled[i] = static_cast<DeviceScalar>(psiVal);
            bScaled[i] = static_cast<DeviceScalar>(rhsVal);
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
        if (matvecSignD == 1.0)
        {
            if (cudaMemcpy(d_b, source.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("cudaMemcpy(b)");
        }
        else
        {
            List<DeviceScalar> bHost(rows);
            for (int i = 0; i < rows; ++i)
            {
                bHost[i] = static_cast<DeviceScalar>(matvecSignD * static_cast<double>(source[i]));
            }
            if (cudaMemcpy(d_b, bHost.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("cudaMemcpy(bScaledSign)");
        }
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

    if (cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS)
        return fail("cublasSetPointerMode");

    cudaStream_t solverStream = nullptr;
    if (usePipelinedCG || useCudaGraph)
    {
        if (cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking) != cudaSuccess)
        {
            return fail("cudaStreamCreate");
        }
        solverStream = computeStream;
    }

    if (cublasSetStream(cublasHandle, solverStream) != CUBLAS_STATUS_SUCCESS)
    {
        return fail("cublasSetStream");
    }
    if (cusparseSetStream(cusparseHandle, solverStream) != CUSPARSE_STATUS_SUCCESS)
    {
        return fail("cusparseSetStream");
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

    if (reportStats && (!Pstream::parRun() || Pstream::master()) && rows > 0)
    {
        DeviceScalar* d_nullIn = nullptr;
        DeviceScalar* d_nullOut = nullptr;
        if
        (
            cudaMalloc(reinterpret_cast<void**>(&d_nullIn), sizeof(DeviceScalar)*rows) == cudaSuccess
         && cudaMalloc(reinterpret_cast<void**>(&d_nullOut), sizeof(DeviceScalar)*rows) == cudaSuccess
        )
        {
            std::vector<double> onesHost(static_cast<std::size_t>(rows), 1.0);
            if
            (
                cudaMemcpy
                (
                    d_nullIn,
                    onesHost.data(),
                    sizeof(DeviceScalar)*rows,
                    cudaMemcpyHostToDevice
                ) == cudaSuccess
             && cudaMemset(d_nullOut, 0, sizeof(DeviceScalar)*rows) == cudaSuccess
            )
            {
                cusparseDnVecDescr_t nullInDesc = nullptr;
                cusparseDnVecDescr_t nullOutDesc = nullptr;
                if
                (
                    cusparseCreateDnVec(&nullInDesc, rows, d_nullIn, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                 && cusparseCreateDnVec(&nullOutDesc, rows, d_nullOut, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                )
                {
                    const double alphaNull = 1.0;
                    const double betaNull = 0.0;
                    size_t nullBufferSize = 0;
                    void* nullBuffer = nullptr;
                    cusparseStatus_t bufStatus =
                        cusparseSpMV_bufferSize
                        (
                            cusparseHandle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alphaNull,
                            matA,
                            nullInDesc,
                            &betaNull,
                            nullOutDesc,
                            CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT,
                            &nullBufferSize
                        );
                    if
                    (
                        bufStatus == CUSPARSE_STATUS_SUCCESS
                     && nullBufferSize > 0
                     && cudaMalloc(&nullBuffer, nullBufferSize) != cudaSuccess
                    )
                    {
                        nullBuffer = nullptr;
                    }
                    cusparseStatus_t nsStatus =
                        cusparseSpMV
                        (
                            cusparseHandle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alphaNull,
                            matA,
                            nullInDesc,
                            &betaNull,
                            nullOutDesc,
                            CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT,
                            nullBuffer ? nullBuffer : nullptr
                        );
                    if (nullBuffer)
                    {
                        cudaFree(nullBuffer);
                    }
                    if (nsStatus == CUSPARSE_STATUS_SUCCESS)
                    {
                        std::vector<double> nullOutHost(static_cast<std::size_t>(rows), 0.0);
                        if
                        (
                            cudaMemcpy
                            (
                                nullOutHost.data(),
                                d_nullOut,
                                sizeof(DeviceScalar)*rows,
                                cudaMemcpyDeviceToHost
                            ) == cudaSuccess
                        )
                        {
                            double nullNorm = 0.0;
                            double maxAbsNull = 0.0;
                            for (double v : nullOutHost)
                            {
                                nullNorm += v*v;
                                maxAbsNull = std::max(maxAbsNull, std::fabs(v));
                            }
                            nullNorm = std::sqrt(std::max(0.0, nullNorm));
                            Info<< "cudaPCG(" << fieldName_ << "): nullspace check |A*1|_2="
                                << nullNorm << " max|A*1|=" << maxAbsNull << nl;
                        }
                    }
                }
                if (nullInDesc) cusparseDestroyDnVec(nullInDesc);
                if (nullOutDesc) cusparseDestroyDnVec(nullOutDesc);
            }
        }
        if (d_nullIn) cudaFree(d_nullIn);
        if (d_nullOut) cudaFree(d_nullOut);
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
                    DeviceScalar v = static_cast<DeviceScalar>(matvecSignD * static_cast<double>(values[k]));
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
    bool icCacheWasAllowed = false;
    bool icRefactorThisSolve = false;
    label icShiftAppliedSolve = 0;
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
        int* factorRowPtrDev =
            (permutationActive && preconCacheEntry && preconCacheEntry->icPatternReady)
          ? preconCacheEntry->d_icRowPtr
          : d_rowPtr;
        int* factorColIndDev =
            (permutationActive && preconCacheEntry && preconCacheEntry->icPatternReady)
          ? preconCacheEntry->d_icColInd
          : d_colInd;

        // Allocate separate values array for in-place factorization
        // Start from host values and apply diagonal regularisation specifically for ILU/IC
        List<DeviceScalar> iluValsHost(nnz);
        List<int> diagPositions(rows, -1);
        const scalar iluDiagAbsFloorCfg = dict.lookupOrDefault<scalar>("iluDiagAbsFloor", diagFloorCfg);
        const scalar iluDiagRelFloorCfg = dict.lookupOrDefault<scalar>("iluDiagRelFloor", scalar(0));
        const double iluDiagFloorEff = std::max(static_cast<double>(iluDiagAbsFloorCfg), static_cast<double>(iluDiagRelFloorCfg)*static_cast<double>(maxAbsDiag));
        const bool allowIcCache = useIC && !useILU && preconCacheEntry;
        const Foam::scalar icRefactorMultiplier = scalar(1.5);
        bool reuseIcNumeric = false;
        bool requireNumericFactor = true;
        if (useIC && allowIcCache && preconCacheEntry->icFactorValid && preconCacheEntry->icAnalysisReady)
        {
            const Foam::scalar medianIters = preconCacheEntry->icRollingMedian;
            const Foam::label lastIters = preconCacheEntry->icLastIterations;
            const bool refactorDueToMedian =
                (medianIters > VSMALL)
             && (static_cast<Foam::scalar>(lastIters) > icRefactorMultiplier*medianIters);
            const bool refactorDueToPivot = preconCacheEntry->icHadZeroPivot;
            if (!refactorDueToMedian && !refactorDueToPivot)
            {
                reuseIcNumeric = true;
                requireNumericFactor = false;
            }
        }
        if (reuseIcNumeric)
        {
            const double cachedScale = preconCacheEntry->icFactorScale;
            const double scaleTolerance =
                1e-8*std::max(std::max(std::fabs(cachedScale), std::fabs(factorScale)), 1.0);
            if (std::fabs(cachedScale - factorScale) > scaleTolerance)
            {
                reuseIcNumeric = false;
                requireNumericFactor = true;
            }
        }
        if (reuseIcNumeric)
        {
            if
            (
                !preconCacheEntry->d_icVals
             || preconCacheEntry->icCapacityNnz != nnz
             || !preconCacheEntry->icInfo
            )
            {
                reuseIcNumeric = false;
                requireNumericFactor = true;
            }
            else
            {
                ++preconCacheEntry->icFactorReuseHits;
                factorScale = preconCacheEntry->icFactorScale;
            }
        }
        label icShiftApplied = 0;
        icCacheWasAllowed = allowIcCache;
        const bool buildNumericFactors = useILU || requireNumericFactor;
        std::vector<std::pair<double, Foam::label>> shiftTraceSolve;

        if (buildNumericFactors)
        {
            for (int i = 0; i < nnz; ++i)
            {
                iluValsHost[i] = factorValsNumeric[i];
            }

            for (int i = 0; i < rows; ++i)
            {
                int diagK = -1;
                for (int k = factorRowPtrHost[i]; k < factorRowPtrHost[i+1]; ++k)
                {
                    if (factorColIndHost[k] == i)
                    {
                        diagK = k;
                        break;
                    }
                }
                if (diagK >= 0)
                {
                    double dv = static_cast<double>(iluValsHost[diagK]);
                    if (useIC)
                    {
                        // IC0 expects positive diagonals
                        const Foam::label origRow = permutationActive ? permutation[i] : i;
                        double s = 1.0;
                        if (scaleIcFactors && origRow < invSqrtDiagHost.size())
                        {
                            s = static_cast<double>(invSqrtDiagHost[origRow]);
                        }
                        double scaleSq = s*s;
                        if (!std::isfinite(scaleSq) || scaleSq <= 0.0)
                        {
                            scaleSq = 1.0;
                        }
                        const double diagFloorScaled = iluDiagFloorEff * scaleSq;
                        if (dv <= 0.0 || std::abs(dv) < diagFloorScaled)
                        {
                            dv = diagFloorScaled;
                        }
                    }
                    else
                    {
                        // ILU0: avoid near-zero diagonals, preserve sign
                        if (std::abs(dv) < iluDiagFloorEff)
                        {
                            dv = (dv >= 0.0 ? iluDiagFloorEff : -iluDiagFloorEff);
                        }
                    }
                    iluValsHost[diagK] = static_cast<DeviceScalar>(dv);
                    diagPositions[i] = diagK;
                }
            }
            if (icDebugDump && reportStats && Pstream::master())
            {
                double diagMinTest = std::numeric_limits<double>::max();
                double diagMaxTest = -std::numeric_limits<double>::max();
                for (int i = 0; i < rows; ++i)
                {
                    const int diagK = diagPositions[i];
                    if (diagK >= 0)
                    {
                        const double dv = static_cast<double>(iluValsHost[diagK]);
                        diagMinTest = std::min(diagMinTest, dv);
                        diagMaxTest = std::max(diagMaxTest, dv);
                    }
                }
                Info<< "cudaPCG(" << fieldName_ << "): IC host diag pre-factor range ["
                    << diagMinTest << ", " << diagMaxTest << ']' << nl;
            }
        }

        if (allowIcCache && iluDisableReason.empty())
        {
            if (requireNumericFactor)
            {
                if (!preconCacheEntry->ensureIcCapacity(nnz))
                {
                    iluDisableReason = std::string("allocVals");
                }
                else
                {
                    d_iluVals = reinterpret_cast<DeviceScalar*>(preconCacheEntry->d_icVals);
                    d_iluValsFromCache = true;
                }
            }
            else
            {
                d_iluVals = reinterpret_cast<DeviceScalar*>(preconCacheEntry->d_icVals);
                d_iluValsFromCache = true;
            }
        }

        if (!d_iluVals && iluDisableReason.empty())
        {
            if (cudaMalloc(reinterpret_cast<void**>(&d_iluVals), sizeof(DeviceScalar)*nnz) != cudaSuccess)
            {
                iluDisableReason = std::string("allocVals");
            }
        }

        if (requireNumericFactor && iluDisableReason.empty())
        {
            if
            (
                cudaMemcpy
                (
                    d_iluVals,
                    iluValsHost.begin(),
                    sizeof(DeviceScalar)*nnz,
                    cudaMemcpyHostToDevice
                ) != cudaSuccess
            )
            {
                iluDisableReason = std::string("copyVals");
            }
        }

        if (iluDisableReason.empty() && cusparseCreateMatDescr(&iluDescr) != CUSPARSE_STATUS_SUCCESS)
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
                            factorRowPtrDev, factorColIndDev,
                            iluInfo, &bs
                        ) == CUSPARSE_STATUS_SUCCESS
                     && (bs == 0 || cudaMalloc(&iluBuf, bs) == cudaSuccess)
                     && cusparseDcsrilu02_analysis
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            factorRowPtrDev, factorColIndDev,
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
                            factorRowPtrDev, factorColIndDev,
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
                            cusparseCreateCsr(&matL, rows, rows, nnz, factorRowPtrDev, factorColIndDev, d_iluVals,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                         && cusparseCreateCsr(&matU, rows, rows, nnz, factorRowPtrDev, factorColIndDev, d_iluVals,
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
                csric02Info_t icInfoLocal = nullptr;
                bool analysisRequired = true;
                bool factorSuccess = false;
                if (!requireNumericFactor)
                {
                    factorSuccess = true;
                }

                if (allowIcCache)
                {
                    if (!preconCacheEntry->icInfo)
                    {
                        if (cusparseCreateCsric02Info(&preconCacheEntry->icInfo) != CUSPARSE_STATUS_SUCCESS)
                        {
                            iluDisableReason = std::string("createInfo");
                        }
                    }
                    if (iluDisableReason.empty())
                    {
                        icInfoLocal = preconCacheEntry->icInfo;
                        icInfoFromCache = true;
                        analysisRequired = !preconCacheEntry->icAnalysisReady;
                    }
                }

                if (!icInfoLocal && iluDisableReason.empty())
                {
                    if (cusparseCreateCsric02Info(&icInfoLocal) != CUSPARSE_STATUS_SUCCESS)
                    {
                        iluDisableReason = std::string("createInfo");
                    }
                }

                if (!requireNumericFactor)
                {
                    analysisRequired = false;
                }

                if (iluDisableReason.empty())
                {
                    icInfo = icInfoLocal;
                    int bs = 0;
                    if
                    (
                        cusparseDcsric02_bufferSize
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            factorRowPtrDev, factorColIndDev,
                            icInfo, &bs
                        ) != CUSPARSE_STATUS_SUCCESS
                    )
                    {
                        iluDisableReason = std::string("csric02Buffer");
                    }
                    else
                    {
                        iluBufSize = static_cast<size_t>(bs);
                        if (allowIcCache)
                        {
                            if (!preconCacheEntry->ensureIcBuffer(iluBufSize))
                            {
                                iluDisableReason = std::string("allocBuf");
                            }
                            else
                            {
                                iluBuf = preconCacheEntry->icBuffer;
                                icBufferFromCache = true;
                            }
                        }
                        if (!icBufferFromCache)
                        {
                            if (iluBufSize && cudaMalloc(&iluBuf, iluBufSize) != cudaSuccess)
                            {
                                iluDisableReason = std::string("allocBuf");
                            }
                        }
                    }
                }

                if (iluDisableReason.empty() && analysisRequired)
                {
                    if
                    (
                        cusparseDcsric02_analysis
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            factorRowPtrDev, factorColIndDev,
                            icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                            iluBuf
                        ) != CUSPARSE_STATUS_SUCCESS
                    )
                    {
                        iluDisableReason = std::string("csric02Analysis");
                    }
                    else if (allowIcCache)
                    {
                        preconCacheEntry->icAnalysisReady = true;
                    }
                }
                else if (iluDisableReason.empty() && allowIcCache)
                {
                    ++preconCacheEntry->icAnalysisReuseHits;
                }

                if (requireNumericFactor && iluDisableReason.empty())
                {
                    const double diagMedianScaled = icDiagMedianScaled;
                    List<DeviceScalar> iluValsBase(iluValsHost);

                    cusparseStatus_t status =
                        cusparseDcsric02
                        (
                            cusparseHandle, rows, nnz,
                            iluDescr,
                            reinterpret_cast<double*>(d_iluVals),
                            factorRowPtrDev, factorColIndDev,
                            icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                            iluBuf
                        );
                    if (status == CUSPARSE_STATUS_SUCCESS)
                    {
                        factorSuccess = true;
                        shiftTraceSolve.emplace_back(0.0, -1);
                    }
#if defined(CUSPARSE_STATUS_NUMERICAL_ERROR)
                    else if
                    (
                        status == CUSPARSE_STATUS_ZERO_PIVOT
                     || status == CUSPARSE_STATUS_NUMERICAL_ERROR
                    )
#else
                    else if (status == CUSPARSE_STATUS_ZERO_PIVOT)
#endif
                    {
                        int pivot = -1;
                        cusparseXcsric02_zeroPivot(cusparseHandle, icInfo, &pivot);
                        shiftTraceSolve.emplace_back(0.0, static_cast<Foam::label>(pivot));
                        const std::array<double, 10> shiftTau = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0};
                        for (double tau : shiftTau)
                        {
                            double shiftAmount =
                                tau * (diagMedianScaled > 0.0 ? diagMedianScaled : static_cast<double>(diagFloorCfg));
                            if (shiftAmount <= 0.0)
                            {
                                shiftAmount = tau * std::max(static_cast<double>(diagFloorCfg), 1e-12);
                            }
                            iluValsHost = iluValsBase;
                            for (int r = 0; r < rows; ++r)
                            {
                                const int diagPos = diagPositions[r];
                                if (diagPos >= 0)
                                {
                                    double updated = static_cast<double>(iluValsHost[diagPos]) + shiftAmount;
                                    if (updated <= 0.0) updated = shiftAmount;
                                    iluValsHost[diagPos] = static_cast<DeviceScalar>(updated);
                                }
                            }
                            if
                            (
                                cudaMemcpy
                                (
                                    d_iluVals,
                                    iluValsHost.begin(),
                                    sizeof(DeviceScalar)*nnz,
                                    cudaMemcpyHostToDevice
                                ) != cudaSuccess
                            )
                            {
                                iluDisableReason = std::string("copyValsShift");
                                break;
                            }
                            ++icShiftApplied;
                            status =
                                cusparseDcsric02
                                (
                                    cusparseHandle, rows, nnz,
                                    iluDescr,
                                    reinterpret_cast<double*>(d_iluVals),
                                    factorRowPtrDev, factorColIndDev,
                                    icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                    iluBuf
                                );
                            int pivotIdx = -1;
#if defined(CUSPARSE_STATUS_NUMERICAL_ERROR)
                            if
                            (
                                status == CUSPARSE_STATUS_ZERO_PIVOT
                             || status == CUSPARSE_STATUS_NUMERICAL_ERROR
                            )
#else
                            if (status == CUSPARSE_STATUS_ZERO_PIVOT)
#endif
                            {
                                cusparseXcsric02_zeroPivot(cusparseHandle, icInfo, &pivotIdx);
                                shiftTraceSolve.emplace_back(tau, static_cast<Foam::label>(pivotIdx));
                                continue;
                            }
                            else if (status == CUSPARSE_STATUS_SUCCESS)
                            {
                                factorSuccess = true;
                                shiftTraceSolve.emplace_back(tau, -1);
                                break;
                            }
                            else
                            {
                                iluDisableReason = std::string("csric02");
                                break;
                            }
                        }
                    }
                    else
                    {
                        iluDisableReason = std::string("csric02");
                    }

                    if (!factorSuccess && iluDisableReason.empty())
                    {
                        iluDisableReason = std::string("csric02");
                    }
                }
                else if (!requireNumericFactor && iluDisableReason.empty())
                {
                    factorSuccess = true;
                }

                if (iluDisableReason.empty())
                {
                    if (icDebugDump && useIC && factorSuccess)
                    {
                        Info<< "cudaPCG(" << fieldName_ << "): IC debug dump triggered (nnz=" << nnz << ")" << nl;
                        Foam::List<double> icValsHostDebug(nnz, 0.0);
                        if
                        (
                            cudaMemcpy
                            (
                                icValsHostDebug.begin(),
                                reinterpret_cast<const double*>(d_iluVals),
                                sizeof(double)*nnz,
                                cudaMemcpyDeviceToHost
                            ) == cudaSuccess
                        )
                        {
                            double minDiagIC = std::numeric_limits<double>::max();
                            double maxDiagIC = -std::numeric_limits<double>::max();
                            double minAbsDiagIC = std::numeric_limits<double>::max();
                            double maxAbsDiagIC = 0.0;
                            Foam::label diagCountIC = 0;
                            Foam::label nonPosDiagIC = 0;
                            for (int row = 0; row < rows; ++row)
                            {
                                const int diagPos = diagPositions[row];
                                if (diagPos >= 0)
                                {
                                    const double diagVal = icValsHostDebug[diagPos];
                                    minDiagIC = std::min(minDiagIC, diagVal);
                                    maxDiagIC = std::max(maxDiagIC, diagVal);
                                    const double absDiagVal = std::fabs(diagVal);
                                    minAbsDiagIC = std::min(minAbsDiagIC, absDiagVal);
                                    maxAbsDiagIC = std::max(maxAbsDiagIC, absDiagVal);
                                    if (!std::isfinite(diagVal) || diagVal <= 0.0)
                                    {
                                        ++nonPosDiagIC;
                                    }
                                    ++diagCountIC;
                                }
                            }

                            if (diagCountIC == 0)
                            {
                                minDiagIC = 0.0;
                                maxDiagIC = 0.0;
                                minAbsDiagIC = 0.0;
                                maxAbsDiagIC = 0.0;
                            }

                            double minScale = std::numeric_limits<double>::max();
                            double maxScale = 0.0;
                            for (int i = 0; i < invSqrtDiagHost.size(); ++i)
                            {
                                const double scaleVal = static_cast<double>(invSqrtDiagHost[i]);
                                const double absScale = std::fabs(scaleVal);
                                if (absScale > 0.0)
                                {
                                    minScale = std::min(minScale, absScale);
                                    maxScale = std::max(maxScale, absScale);
                                }
                            }

                            double minRuiz = std::numeric_limits<double>::max();
                            double maxRuiz = 0.0;
                            if (icUseRuizScaling && preconCacheEntry && preconCacheEntry->ruizReady)
                            {
                                Foam::List<double> ruizHost(rows, 1.0);
                                if
                                (
                                    cudaMemcpy
                                    (
                                        ruizHost.begin(),
                                        preconCacheEntry->d_ruizScale,
                                        sizeof(double)*rows,
                                        cudaMemcpyDeviceToHost
                                    ) == cudaSuccess
                                )
                                {
                                    for (int i = 0; i < rows; ++i)
                                    {
                                        const double val = std::fabs(ruizHost[i]);
                                        minRuiz = std::min(minRuiz, val);
                                        maxRuiz = std::max(maxRuiz, val);
                                    }
                                }
                            }

                            Info<< "cudaPCG(" << fieldName_ << "): IC diag stats"
                                << " min=" << minDiagIC
                                << " max=" << maxDiagIC
                                << " min|diag|=" << minAbsDiagIC
                                << " max|diag|=" << maxAbsDiagIC
                                << " nonPos=" << nonPosDiagIC
                                << " scale|min,max|=" << minScale << ',' << maxScale;
                            if (icUseRuizScaling && preconCacheEntry && preconCacheEntry->ruizReady)
                            {
                                Info<< " ruiz|min,max|=" << minRuiz << ',' << maxRuiz;
                            }
                            Info<< nl;
                        }
                        else
                        {
                            Info<< "cudaPCG(" << fieldName_ << "): IC diag stats unavailable (cudaMemcpy failure)" << nl;
                        }
                    }

                    if
                    (
                        cusparseCreateCsr(&matLchol, rows, rows, nnz, factorRowPtrDev, factorColIndDev, d_iluVals,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS
                     && cusparseSpSV_createDescr(&spsvLowerChol) == CUSPARSE_STATUS_SUCCESS
                     && cusparseSpSV_createDescr(&spsvUpperTChol) == CUSPARSE_STATUS_SUCCESS
                    )
                    {
                        cusparseFillMode_t fillL = CUSPARSE_FILL_MODE_LOWER;
                        cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
                        cusparseSpMatSetAttribute(matLchol, CUSPARSE_SPMAT_FILL_MODE, &fillL, sizeof(fillL));
                        cusparseSpMatSetAttribute(matLchol, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

                        const double oneD = 1.0;
                        size_t bL = 0, bUT = 0;
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
                            size_t requiredBuf = std::max(iluBufSize, std::max(bL, bUT));
                            if (allowIcCache)
                            {
                                if (!preconCacheEntry->ensureIcBuffer(requiredBuf))
                                {
                                    iluDisableReason = std::string("allocBuf");
                                }
                                else
                                {
                                    iluBuf = preconCacheEntry->icBuffer;
                                    icBufferFromCache = true;
                                    iluBufSize = preconCacheEntry->icBufferSize;
                                }
                            }
                            if (!icBufferFromCache)
                            {
                                if (requiredBuf && (iluBufSize < requiredBuf))
                                {
                                    if (iluBuf) cudaFree(iluBuf);
                                    if (cudaMalloc(&iluBuf, requiredBuf) != cudaSuccess)
                                    {
                                        iluDisableReason = std::string("allocBuf");
                                    }
                                    else
                                    {
                                        iluBufSize = requiredBuf;
                                    }
                                }
                            }
                            if (iluDisableReason.empty())
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
                    else
                    {
                        iluDisableReason = std::string("csric02Mat");
                    }
                }

                if (allowIcCache)
                {
                    preconCacheEntry->icShiftCounter += icShiftApplied;
                    icRefactorThisSolve = requireNumericFactor && factorSuccess && icReady && iluDisableReason.empty();
                    if (requireNumericFactor)
                    {
                        preconCacheEntry->icShiftTrace = shiftTraceSolve;
                        if (factorSuccess && icReady && iluDisableReason.empty())
                        {
                            preconCacheEntry->icFactorValid = true;
                            preconCacheEntry->icHadZeroPivot = icShiftApplied > 0;
                            ++preconCacheEntry->icRefactorCount;
                            preconCacheEntry->cachedMatvecSign = matvecSignD;
                            preconCacheEntry->icFactorScale = factorScale;
                            preconCacheEntry->icApplyScale = 1.0;
                        }
                        else
                        {
                            preconCacheEntry->icFactorValid = false;
                            preconCacheEntry->icHadZeroPivot = false;
                            preconCacheEntry->icFactorScale = 1.0;
                            preconCacheEntry->icApplyScale = 1.0;
                        }
                    }
                }
            }
            icShiftAppliedSolve = icShiftApplied;
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
        const double omegaForward = colourSpdMode ? 1.0 : colourOmegaEff;
        const double omegaBackward = colourSpdMode ? 1.0 : colourBackwardOmegaEff;

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
        auto scaleVector = [&](double* vec, const double* diag, const char* tag) -> bool
        {
            if (!diag)
            {
                return true;
            }
            if
            (
                cublasDdgmm
                (
                    cublasHandle,
                    CUBLAS_SIDE_LEFT,
                    rows,
                    1,
                    vec,
                    rows,
                    diag,
                    1,
                    vec,
                    rows
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                message = word(tag);
                return false;
            }
            return true;
        };

        const bool haveScaleVector =
            (icUseDiagonalScaling || (icUseRuizScaling && preconCacheEntry && preconCacheEntry->ruizReady))
         && preconCacheEntry && preconCacheEntry->d_invSqrtDiag;

        auto solveIcInPlace = [&](double* vec) -> bool
        {
            if (!preconCacheEntry || !d_preTmp || !preIn || !preOut)
            {
                message = word("ICScaleMissing");
                return false;
            }

            const bool usePermutation =
                preconCacheEntry
             && preconCacheEntry->permutationReady
             && preconCacheEntry->d_perm;

            if
            (
                preconCacheEntry
             && std::fabs(preconCacheEntry->cachedMatvecSign - matvecSignD) > 0.5
            )
            {
                message = word("ICCacheSignMismatch");
                return false;
            }

            if (preconCacheEntry->icFactorValid)
            {
                const double cacheScale = preconCacheEntry->icFactorScale;
                const double scaleTol =
                    1e-8*std::max(std::max(std::fabs(cacheScale), std::fabs(factorScaleApplied)), 1.0);
                if (std::fabs(cacheScale - factorScaleApplied) > scaleTol)
                {
                    message = word("ICCacheScaleMismatch");
                    return false;
                }
            }

            if (usePermutation)
            {
                const std::size_t cachePermId =
                    preconCacheEntry ? preconCacheEntry->cachedPermutationId : std::size_t(0);
                if (cachePermId != permutationId)
                {
                    message = word("ICCachePermutationMismatch");
                    return false;
                }
                if (preconCacheEntry && !preconCacheEntry->permutationStructureValidated)
                {
                    message = word("ICPermutationStructureInvalid");
                    return false;
                }
            }
            else if
            (
                preconCacheEntry
             && preconCacheEntry->cachedPermutationId != 0
             && preconCacheEntry->permutationReady
            )
            {
                message = word("ICPermutationUnexpected");
                return false;
            }

            const bool logProbeEnabled =
                (icDebugIterLog > 0)
             && reportStats
             && (!Pstream::parRun() || Pstream::master());

            auto logProbeVec = [&](const char* tag, const double* devicePtr) -> bool
            {
                if (!logProbeEnabled)
                {
                    return true;
                }
                if (!devicePtr)
                {
                    if (reportStats && (!Pstream::parRun() || Pstream::master()))
                    {
                        WarningInFunction
                            << "cudaPCG(" << fieldName_ << "): IC probe " << tag
                            << " skipped (null device pointer)" << Foam::endl;
                    }
                    return true;
                }
                if (!syncStream())
                {
                    return false;
                }
                std::vector<double> hostVec(rows, 0.0);
                const std::size_t bytes = static_cast<std::size_t>(rows)*sizeof(double);
                if
                (
                    cudaMemcpy
                    (
                        hostVec.data(),
                        devicePtr,
                        bytes,
                        cudaMemcpyDeviceToHost
                    ) != cudaSuccess
                )
                {
                    message = word("ICProbeCopyVec");
                    return false;
                }
                double nrm2 = 0.0;
                double maxAbs = 0.0;
                for (double val : hostVec)
                {
                    nrm2 += val*val;
                    maxAbs = std::max(maxAbs, std::fabs(val));
                }
                nrm2 = std::sqrt(std::max(nrm2, 0.0));
                Info<< "cudaPCG(" << fieldName_ << "): IC probe " << tag
                    << " nrm2=" << nrm2
                    << " maxAbs=" << maxAbs << nl;
                return true;
            };

            CUfunction permGatherKernel = nullptr;
            CUfunction permScatterKernel = nullptr;

            auto gatherPermutationVec =
                [&](const double* inDev, double* outDev) -> bool
            {
                if (!usePermutation)
                {
                    if (inDev != outDev)
                    {
                        if
                        (
                            cudaMemcpy
                            (
                                outDev,
                                inDev,
                                sizeof(double)*rows,
                                cudaMemcpyDeviceToDevice
                            ) != cudaSuccess
                        )
                        {
                            message = word("ICPermMemcpy");
                            return false;
                        }
                    }
                    return true;
                }
                try
                {
                    if (!permGatherKernel)
                    {
                        if (!getPermutationKernels(deviceId, permGatherKernel, permScatterKernel, reportStats))
                        {
                            message = word("ICPermKernelCompile");
                            return false;
                        }
                    }
                    CUstream launchStream = computeStream ? reinterpret_cast<CUstream>(computeStream) : nullptr;
                    if
                    (
                        !launchPermutationKernel
                        (
                            permGatherKernel,
                            rows,
                            reinterpret_cast<CUdeviceptr>(preconCacheEntry->d_perm),
                            reinterpret_cast<CUdeviceptr>(const_cast<double*>(inDev)),
                            reinterpret_cast<CUdeviceptr>(outDev),
                            launchStream
                        )
                    )
                    {
                        message = word("ICPermGatherRun");
                        return false;
                    }
                }
                catch (...)
                {
                    message = word("ICPermGatherFail");
                    return false;
                }
                return true;
            };

            auto scatterPermutationVec =
                [&](const double* inDev, double* outDev) -> bool
            {
                if (!usePermutation)
                {
                    if (inDev != outDev)
                    {
                        if
                        (
                            cudaMemcpy
                            (
                                outDev,
                                inDev,
                                sizeof(double)*rows,
                                cudaMemcpyDeviceToDevice
                            ) != cudaSuccess
                        )
                        {
                            message = word("ICPermMemcpy");
                            return false;
                        }
                    }
                    return true;
                }
                try
                {
                    if (!permScatterKernel)
                    {
                        if (!getPermutationKernels(deviceId, permGatherKernel, permScatterKernel, reportStats))
                        {
                            message = word("ICPermKernelCompile");
                            return false;
                        }
                    }
                    CUstream launchStream = computeStream ? reinterpret_cast<CUstream>(computeStream) : nullptr;
                    if
                    (
                        !launchPermutationKernel
                        (
                            permScatterKernel,
                            rows,
                            reinterpret_cast<CUdeviceptr>(preconCacheEntry->d_perm),
                            reinterpret_cast<CUdeviceptr>(const_cast<double*>(inDev)),
                            reinterpret_cast<CUdeviceptr>(outDev),
                            launchStream
                        )
                    )
                    {
                        message = word("ICPermScatterRun");
                        return false;
                    }
                }
                catch (...)
                {
                    message = word("ICPermScatterFail");
                    return false;
                }
                return true;
            };

            auto verifyPermutationRoundTrip = [&]() -> bool
            {
                if (!usePermutation || !preconCacheEntry)
                {
                    return true;
                }
                if (rows <= 0)
                {
                    preconCacheEntry->permutationValidated = true;
                    return true;
                }

                std::vector<double> hostVec(rows, 0.0);
                std::mt19937 rng(static_cast<unsigned>
                    (
                        permutationId
                      ? permutationId
                      : hashCombine(matrixPatternId, static_cast<std::size_t>(rows))
                    ));
                std::uniform_real_distribution<double> dist(-1.0, 1.0);
                for (int i = 0; i < rows; ++i)
                {
                    hostVec[i] = dist(rng);
                }

                double* d_vec = nullptr;
                double* d_tmp1 = nullptr;
                double* d_tmp2 = nullptr;
                auto cleanup = [&]()
                {
                    if (d_vec) { cudaFree(d_vec); d_vec = nullptr; }
                    if (d_tmp1) { cudaFree(d_tmp1); d_tmp1 = nullptr; }
                    if (d_tmp2) { cudaFree(d_tmp2); d_tmp2 = nullptr; }
                };

                const std::size_t bytes = static_cast<std::size_t>(rows)*sizeof(double);
                if
                (
                    cudaMalloc(reinterpret_cast<void**>(&d_vec), bytes) != cudaSuccess
                 || cudaMalloc(reinterpret_cast<void**>(&d_tmp1), bytes) != cudaSuccess
                 || cudaMalloc(reinterpret_cast<void**>(&d_tmp2), bytes) != cudaSuccess
                )
                {
                    cleanup();
                    message = word("ICPermRoundTripAlloc");
                    return false;
                }

                if
                (
                    cudaMemcpy
                    (
                        d_vec,
                        hostVec.data(),
                        bytes,
                        cudaMemcpyHostToDevice
                    ) != cudaSuccess
                )
                {
                    cleanup();
                    message = word("ICPermRoundTripCopyIn");
                    return false;
                }

                if (!gatherPermutationVec(d_vec, d_tmp1))
                {
                    cleanup();
                    return false;
                }

                std::vector<double> hostGather(rows, 0.0);
                if
                (
                    cudaMemcpy
                    (
                        hostGather.data(),
                        d_tmp1,
                        bytes,
                        cudaMemcpyDeviceToHost
                    ) != cudaSuccess
                )
                {
                    cleanup();
                    message = word("ICPermRoundTripGatherCopy");
                    return false;
                }

                std::vector<double> hostScatter(rows, 0.0);
                if (!scatterPermutationVec(d_tmp1, d_tmp2))
                {
                    cleanup();
                    return false;
                }
                if
                (
                    cudaMemcpy
                    (
                        hostScatter.data(),
                        d_tmp2,
                        bytes,
                        cudaMemcpyDeviceToHost
                    ) != cudaSuccess
                )
                {
                    cleanup();
                    message = word("ICPermRoundTripScatterCopy");
                    return false;
                }

                if (!scatterPermutationVec(d_vec, d_tmp2))
                {
                    cleanup();
                    return false;
                }
                if (!gatherPermutationVec(d_tmp2, d_tmp1))
                {
                    cleanup();
                    return false;
                }
                std::vector<double> hostGatherBack(rows, 0.0);
                if
                (
                    cudaMemcpy
                    (
                        hostGatherBack.data(),
                        d_tmp1,
                        bytes,
                        cudaMemcpyDeviceToHost
                    ) != cudaSuccess
                )
                {
                    cleanup();
                    message = word("ICPermRoundTripGatherBackCopy");
                    return false;
                }

                cleanup();

                double roundTripDiff = 0.0;
                double scatterGatherDiff = 0.0;
                double gatherNormSq = 0.0;
                double originalNormSq = 0.0;
                for (int i = 0; i < rows; ++i)
                {
                    roundTripDiff = std::max
                    (
                        roundTripDiff,
                        std::fabs(hostScatter[i] - hostVec[i])
                    );
                    scatterGatherDiff = std::max
                    (
                        scatterGatherDiff,
                        std::fabs(hostGatherBack[i] - hostVec[i])
                    );
                    gatherNormSq += hostGather[i]*hostGather[i];
                    originalNormSq += hostVec[i]*hostVec[i];
                }

                const double gatherNorm = std::sqrt(std::max(gatherNormSq, 0.0));
                const double originalNorm = std::sqrt(std::max(originalNormSq, 0.0));
                const double normDiff = std::fabs(gatherNorm - originalNorm);

                if
                (
                    roundTripDiff > 1e-12
                 || scatterGatherDiff > 1e-12
                 || normDiff > 1e-12
                )
                {
                    if (reportStats && (!Pstream::parRun() || Pstream::master()))
                    {
                        WarningInFunction
                            << "cudaPCG(" << fieldName_
                            << "): permutation round-trip check failed "
                            << "roundTripDiff=" << roundTripDiff
                            << " scatterGatherDiff=" << scatterGatherDiff
                            << " normDiff=" << normDiff << Foam::endl;
                    }
                    message = word("ICPermRoundTrip");
                    return false;
                }

                preconCacheEntry->permutationValidated = true;
                if (reportStats && (!Pstream::parRun() || Pstream::master()))
                {
                    Info<< "cudaPCG(" << fieldName_ << "): permutation round-trip max|Δ|="
                        << roundTripDiff << " scatter∘gather max|Δ|=" << scatterGatherDiff
                        << " normDiff=" << normDiff << nl;
                }
                return true;
            };

            if
            (
                usePermutation
             && preconCacheEntry
             && !preconCacheEntry->permutationValidated
            )
            {
                if (!verifyPermutationRoundTrip())
                {
                    return false;
                }
            }

            if (logProbeEnabled && !d_preTmp2)
            {
                const std::size_t probeBytes = static_cast<std::size_t>(rows)*sizeof(DeviceScalar);
                if (cudaMalloc(reinterpret_cast<void**>(&d_preTmp2), probeBytes) != cudaSuccess)
                {
                    message = word("ICProbeTmpAlloc");
                    return false;
                }
            }

            if (logProbeEnabled)
            {
                if
                (
                    cublasDcopy
                    (
                        cublasHandle,
                        rows,
                        rhs,
                        1,
                        reinterpret_cast<double*>(d_preTmp),
                        1
                    ) != CUBLAS_STATUS_SUCCESS
                )
                {
                    message = word("ICProbeCopy");
                    return false;
                }
                if (!logProbeVec("r", reinterpret_cast<const double*>(d_preTmp)))
                {
                    return false;
                }
                if (haveScaleVector)
                {
                    if (!scaleVector(reinterpret_cast<double*>(d_preTmp), preconCacheEntry->d_invSqrtDiag, "ICProbeScaleIn"))
                    {
                        return false;
                    }
                }
                if (!logProbeVec("S*r", reinterpret_cast<const double*>(d_preTmp)))
                {
                    return false;
                }
                cusparseDnVecSetValues(preIn, d_preTmp);
                cusparseDnVecSetValues(preOut, d_preTmp2);
                const double oneD = 1.0;
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
                    message = word("ICProbeForwardFail");
                    return false;
                }
                if (!logProbeVec("L^-1(S*r)", reinterpret_cast<const double*>(d_preTmp2)))
                {
                    return false;
                }
                cusparseDnVecSetValues(preIn, d_preTmp2);
                cusparseDnVecSetValues(preOut, d_preTmp);
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
                    message = word("ICProbeBackwardFail");
                    return false;
                }
                if (!logProbeVec("L^-T L^-1(S*r)", reinterpret_cast<const double*>(d_preTmp)))
                {
                    return false;
                }
                if (haveScaleVector)
                {
                    if (!scaleVector(reinterpret_cast<double*>(d_preTmp), preconCacheEntry->d_invSqrtDiag, "ICProbeScaleOut"))
                    {
                        return false;
                    }
                }
                if (!logProbeVec("zProbe", reinterpret_cast<const double*>(d_preTmp)))
                {
                    return false;
                }
            }

            double* vecDouble = reinterpret_cast<double*>(vec);
            double* tmpBuffer = reinterpret_cast<double*>(d_preTmp);
            double* tmpBuffer2 = reinterpret_cast<double*>(d_preTmp2);

            if (haveScaleVector)
            {
                if (!scaleVector(vec, preconCacheEntry->d_invSqrtDiag, "ICScaleInFail"))
                {
                    return false;
                }
            }

            double* solveInput = vecDouble;
            double* solveTmp = tmpBuffer2;

            if (usePermutation)
            {
                if (!gatherPermutationVec(vecDouble, tmpBuffer))
                {
                    return false;
                }
                solveInput = tmpBuffer;
                solveTmp = tmpBuffer2;
            }
            else
            {
                solveInput = vecDouble;
                solveTmp = tmpBuffer;
            }

            cusparseDnVecSetValues(preIn, solveInput);
            cusparseDnVecSetValues(preOut, solveTmp);
            const double oneD = 1.0;
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

            cusparseDnVecSetValues(preIn, solveTmp);
            cusparseDnVecSetValues(preOut, solveInput);
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

            if (usePermutation)
            {
                if (!scatterPermutationVec(solveInput, vecDouble))
                {
                    return false;
                }
            }
            else if (solveInput != vecDouble)
            {
                if
                (
                    cudaMemcpy
                    (
                        vecDouble,
                        solveInput,
                        sizeof(double)*rows,
                        cudaMemcpyDeviceToDevice
                    ) != cudaSuccess
                )
                {
                    message = word("ICPermMemcpy");
                    return false;
                }
            }

            if (haveScaleVector)
            {
                if (!scaleVector(vec, preconCacheEntry->d_invSqrtDiag, "ICScaleOutFail"))
                {
                    return false;
                }
            }

            const double applyScale =
                preconCacheEntry ? preconCacheEntry->icApplyScale : 1.0;
            if (std::fabs(applyScale - 1.0) > 1e-12)
            {
                if (cublasDscal(cublasHandle, rows, &applyScale, vec, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("ICApplyScaleFail");
                    return false;
                }
            }

            icAppliedThisSolve = true;
            return true;
        };
        (void)stage;
        if (useChebyshev)
        {
            if (chebyshevDegree <= 0)
            {
                message = word("ChebDegreeInvalid");
                return false;
            }
            if (!applyChebyshevSpd(rhs, out))
            {
                if (chebyshevGuardrailTriggered)
                {
                    useChebyshev = false;
                    colourSpdMode = true;
                    useColour = true;
                    if (applyColourDevice(rhs, out, "chebGuard"))
                    {
                        return true;
                    }
                    return applyDiagonalDevice(rhs, out);
                }
                return false;
            }
            return true;
        }
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
        else if (useAmg)
        {
            if (!preconCacheEntry)
            {
                return fail("amgCacheMissing");
            }

            if (preconCacheEntry->amgHostR.size() != rows)
            {
                preconCacheEntry->amgHostR.setSize(rows);
            }
            if (preconCacheEntry->amgHostZ.size() != rows)
            {
                preconCacheEntry->amgHostZ.setSize(rows);
            }

            if
            (
                cudaMemcpy
                (
                    preconCacheEntry->amgHostR.begin(),
                    rhs,
                    sizeof(double)*rows,
                    cudaMemcpyDeviceToHost
                ) != cudaSuccess
            )
            {
                message = word("amgCopyToHost");
                return false;
            }
            preconCacheEntry->amgHostZ = 0.0;

            if (!preconCacheEntry->amgSolver.valid())
            {
                IStringStream amgStream
                (
                    "{\n"
                    "    solver GAMG;\n"
                    "    tolerance 1e-3;\n"
                    "    relTol 0;\n"
                    "    smoother DICGaussSeidel;\n"
                    "    nPreSweeps 1;\n"
                    "    nPostSweeps 1;\n"
                    "    cacheAgglomeration true;\n"
                    "}\n"
                );
                dictionary amgSolverDict(amgStream());

                auto setupStart = std::chrono::steady_clock::now();
                preconCacheEntry->amgSolver = lduMatrix::solver::New
                (
                    fieldName_,
                    matrix_,
                    interfaceBouCoeffs_,
                    interfaceIntCoeffs_,
                    interfaces_,
                    amgSolverDict
                );
                auto setupEnd = std::chrono::steady_clock::now();
                const double setupMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(setupEnd - setupStart).count();
                preconCacheEntry->amgSetupMs += setupMs;
                amgSetupMsTotal += setupMs;
            }

            auto applyStart = std::chrono::steady_clock::now();
            solverPerformance amgPerf =
                preconCacheEntry->amgSolver->solve
                (
                    preconCacheEntry->amgHostZ,
                    preconCacheEntry->amgHostR,
                    cmpt
                );
            (void)amgPerf;
            auto applyEnd = std::chrono::steady_clock::now();
            const double applyMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(applyEnd - applyStart).count();
            ++preconCacheEntry->amgApplyCount;
            preconCacheEntry->amgApplyMs += applyMs;
            ++amgApplyCountTotal;
            amgApplyMsTotal += applyMs;

            if
            (
                cudaMemcpy
                (
                    out,
                    preconCacheEntry->amgHostZ.begin(),
                    sizeof(double)*rows,
                    cudaMemcpyHostToDevice
                ) != cudaSuccess
            )
            {
                message = word("amgCopyToDevice");
                return false;
            }

            return true;
        }
        else if (useComposite && icReady && (colourOperational && colourReady))
        {
            // Composite: a few coloured sweeps, then IC0
            const label sweeps = std::max<label>(1, dict.lookupOrDefault<label>("compositeColourSweeps", 2));
            bool colourOk = applyColourDevice(rhs, out, "compColour1");
            if (colourOk)
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
            else
            {
                if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
                {
                    message = word("ICCopyFail");
                    return false;
                }
            }
            if (!solveIcInPlace(out))
            {
                return false;
            }
            return true;
        }
        else if (icReady)
        {
            if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
            {
                message = word("ICCopyFail");
                return false;
            }
            return solveIcInPlace(out);
        }
        else if (icForced)
        {
            return applyDiagonalDevice(rhs, out);
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
                << (icForced ? " [forced]" : "")
                << (icReady ? "" : (std::string(" (") + (iluDisableReason.size()?iluDisableReason:"buildFailure") + ")"))
                << " matvecSign=" << matvecSign
                << " cacheId=" << (preconCacheEntry ? preconCacheEntry->icCacheKeyId : 0)
                << nl;
            if (icCacheWasAllowed && preconCacheEntry)
            {
                Info<< "cudaPCG(" << fieldName_ << "): IC cache id="
                    << preconCacheEntry->icCacheKeyId
                    << " analysisReuse=" << preconCacheEntry->icAnalysisReuseHits
                    << " factorReuse=" << preconCacheEntry->icFactorReuseHits
                    << " refactors=" << preconCacheEntry->icRefactorCount
                    << " shiftCount=" << preconCacheEntry->icShiftCounter
                    << " shiftCountSolve=" << icShiftAppliedSolve
                    << " factorValid=" << (preconCacheEntry->icFactorValid ? "true" : "false")
                    << " hadZeroPivot=" << (preconCacheEntry->icHadZeroPivot ? "true" : "false")
                    << " rollingMedian=" << preconCacheEntry->icRollingMedian
                    << " lastIterations=" << preconCacheEntry->icLastIterations << nl;
                if (!preconCacheEntry->icShiftTrace.empty())
                {
                    Info<< "cudaPCG(" << fieldName_ << "): IC shift ladder ("
                        << (icRefactorThisSolve ? "success" : "fail") << ") [";
                    for (std::size_t i = 0; i < preconCacheEntry->icShiftTrace.size(); ++i)
                    {
                        if (i) Info<< ' ';
                        Info<< '(' << preconCacheEntry->icShiftTrace[i].first
                            << ',' << preconCacheEntry->icShiftTrace[i].second << ')';
                    }
                    Info<< ']' << nl;
                }
            }
            else if (icShiftAppliedSolve > 0)
            {
                Info<< "cudaPCG(" << fieldName_ << "): IC zero-pivot shifts this solve="
                    << icShiftAppliedSolve << nl;
            }
            if (icDotGuardTriggered || icEffectComputed)
            {
                Info<< "cudaPCG(" << fieldName_ << "): IC telemetry dotGuard="
                    << (icDotGuardTriggered ? "true" : "false");
                if (icEffectComputed)
                {
                    Info<< " effect0=" << icPrecondEffect0;
                }
                Info<< nl;
            }
        }
    }


    if (chebyshevRestarted && reportStats)
    {
        Info<< "cudaPCG(" << fieldName_ << "): Chebyshev spectral window inflated x"
            << chebyshevLambdaInflateUsed
            << " lambdaMin=" << chebyshevLambdaMinUsed
            << " lambdaMax=" << chebyshevLambdaMaxUsed << nl;
    }

    if (chebyshevGuardrailTriggered && reportStats)
    {
        std::string guardReason = " (residual growth)";
        if (chebyshevDotGuardTriggered)
        {
            guardReason = " (chebSpdDotGuard)";
        }
        else if (chebyshevClampGuardTriggered)
        {
            guardReason = " (invSqrtClamp)";
        }
        Info<< "cudaPCG(" << fieldName_ << "): Chebyshev guardrail engaged"
            << guardReason << nl;
    }

    if (chebyshevSpdApplied && reportStats)
    {
        Info<< "cudaPCG(" << fieldName_ << "): Chebyshev SPD applied degree="
            << chebyshevDegreeUsed
            << " lambdaMin=" << chebyshevLambdaMinUsed
            << " lambdaMax=" << chebyshevLambdaMaxUsed << nl;
    }

    const double alpha = 1.0;
    const double zero = 0.0;

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
    const double negOne = -1.0;
    if (cublasDaxpy(cublasHandle, rows, &negOne, reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas axpy r -= Ap");

    if (!applyPreconditionerVec(reinterpret_cast<const double*>(d_r), reinterpret_cast<double*>(d_z), "initial"))
        return fail(message);

    double rNormInitial = 0.0;
    double zNormInitial = 0.0;
    if (icAppliedThisSolve)
    {
        if (cublasDnrm2(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, &rNormInitial) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas nrm2(r)");
        }
        if (cublasDnrm2(cublasHandle, rows, reinterpret_cast<const double*>(d_z), 1, &zNormInitial) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas nrm2(z)");
        }
        if (std::isfinite(rNormInitial) && std::isfinite(zNormInitial) && rNormInitial > 0.0)
        {
            icPrecondEffect0 = zNormInitial / rNormInitial;
            icEffectComputed = std::isfinite(icPrecondEffect0);
        }
        else
        {
            icEffectComputed = false;
        }
    }

    if (icAppliedThisSolve && icEffectComputed && icPrecondEffect0 > 0.0)
    {
        const double targetEffect =
            std::max(1e-12, static_cast<double>(icNormalizeTargetEffectCfg));
        double postScale = targetEffect / icPrecondEffect0;
        const double postClampMin =
            icNormalizeClampMinCfg > scalar(0)
              ? static_cast<double>(icNormalizeClampMinCfg)
              : 0.0;
        const double postClampMax =
            icNormalizeClampMaxCfg > scalar(0)
              ? static_cast<double>(icNormalizeClampMaxCfg)
              : std::numeric_limits<double>::infinity();
        if (postClampMax > 0.0 && postScale > postClampMax)
        {
            postScale = postClampMax;
        }
        if (postClampMin > 0.0 && postScale < postClampMin)
        {
            postScale = postClampMin;
        }

        if (std::fabs(postScale - 1.0) > 1e-12)
        {
            if (cublasDscal(cublasHandle, rows, &postScale, reinterpret_cast<double*>(d_z), 1) != CUBLAS_STATUS_SUCCESS)
            {
                cleanup();
                return fail("cublas scale(z)");
            }
        }
        if (preconCacheEntry)
        {
            preconCacheEntry->icApplyScale = postScale;
        }
        if (reportStats && (!Pstream::parRun() || Pstream::master()))
        {
            Info<< "cudaPCG(" << fieldName_ << "): IC auto scale postFactor="
                << postScale << " effect0=" << icPrecondEffect0 << nl;
        }
    }
    else if (preconCacheEntry)
    {
        preconCacheEntry->icApplyScale = 1.0;
    }

    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas copy z->p");

    scalarField wA(nCells, Zero);
    scalarField pA(nCells, Zero);

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

    if (icAppliedThisSolve && reportStats)
    {
        Info<< "cudaPCG(" << fieldName_ << "): IC apply metrics rz=" << rz
            << " rNorm0=" << rNormInitial << " zNorm0=" << zNormInitial << nl;
    }

    if (icAppliedThisSolve && (!std::isfinite(rz) || rz <= scalar(0)))
    {
        icDotGuardTriggered = true;
        icEffectComputed = false;
        icPrecondEffect0 = 0.0;
        if (preconCacheEntry)
        {
            preconCacheEntry->icFactorValid = false;
        }
        useIC = false;
        icReady = false;
        useComposite = false;
        icForced = false;
        icAppliedThisSolve = false;
        if (reportStats)
        {
            Info<< "cudaPCG(" << fieldName_ << "): IC dot guard triggered (rz=" << rz
                << ", rNorm0=" << rNormInitial << ", zNorm0=" << zNormInitial << "), demoting to diagonal fallback" << nl;
        }
        if (!applyPreconditionerVec(reinterpret_cast<const double*>(d_r), reinterpret_cast<double*>(d_z), "icDotGuardFallback"))
        {
            return fail(message);
        }
        if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
        {
            return fail("cublas copy z->p");
        }
        double rzDeviceFallback = 0.0;
        if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, reinterpret_cast<const double*>(d_z), 1, &rzDeviceFallback) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dot(r,z) fallback");
        }
        if (!syncStream())
        {
            return false;
        }
        rz = returnReduce(static_cast<scalar>(rzDeviceFallback), sumOp<scalar>());
        if (!std::isfinite(rz) || rz <= scalar(0))
        {
            message = word("ICDotGuardNonPos");
            return fail(message);
        }
    }

    cudaMemcpy(wA.begin(), d_Ap, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(pA.begin(), d_p, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

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
        if (preconCacheEntry && icAppliedThisSolve && icCacheWasAllowed)
        {
            preconCacheEntry->recordIterations(solverPerf.nIterations());
        }
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

        cudaStream_t launchStreamBase = computeStream ? computeStream : nullptr;
        if (iter == 0)
        {
            if (cudaMemsetAsync(d_Ap, 0, sizeof(DeviceScalar)*rows, launchStreamBase) != cudaSuccess)
            {
                return fail("cudaMemset(Ap)");
            }
        }

        bool launchedFromGraph = false;
        if (useCudaGraph)
        {
            if (!spmvGraphReady && iter >= cudaGraphWarmup)
            {
                if (cudaStreamBeginCapture(launchStreamBase, cudaStreamCaptureModeGlobal) != cudaSuccess)
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
                    cudaStreamEndCapture(launchStreamBase, &spmvGraph);
                    return fail("cusparseSpMV (capture)");
                }
                if (cudaStreamEndCapture(launchStreamBase, &spmvGraph) != cudaSuccess)
                {
                    return fail("cudaStreamEndCapture");
                }
                if (cudaGraphInstantiate(&spmvGraphExec, spmvGraph, nullptr, nullptr, 0) != cudaSuccess)
                {
                    return fail("cudaGraphInstantiate");
                }
                spmvGraphReady = true;
                if (cudaGraphLaunch(spmvGraphExec, launchStreamBase) != cudaSuccess)
                {
                    return fail("cudaGraphLaunch");
                }
                launchedFromGraph = true;
            }
            else if (spmvGraphReady)
            {
                if (cudaGraphLaunch(spmvGraphExec, launchStreamBase) != cudaSuccess)
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

        const bool logCgDebug =
            (icDebugIterLog > 0)
         && reportStats
         && (iter < icDebugIterLog)
         && (!Pstream::parRun() || Pstream::master());

        scalar rTrDebug = 0;
        scalar zTrDebug = 0;
        scalar pTrDebug = 0;
        scalar apTrDebug = 0;
        if (logCgDebug)
        {
            double rTrDeviceDebug = 0.0;
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    reinterpret_cast<const double*>(d_r),
                    1,
                    reinterpret_cast<const double*>(d_r),
                    1,
                    &rTrDeviceDebug
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                cleanup();
                return fail("cublas dot(r,r) debug");
            }
            if (!syncStream())
            {
                return false;
            }
            rTrDebug = returnReduce(static_cast<scalar>(rTrDeviceDebug), sumOp<scalar>());

            double zTrDeviceDebug = 0.0;
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    reinterpret_cast<const double*>(d_z),
                    1,
                    reinterpret_cast<const double*>(d_z),
                    1,
                    &zTrDeviceDebug
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                cleanup();
                return fail("cublas dot(z,z) debug");
            }
            if (!syncStream())
            {
                return false;
            }
            zTrDebug = returnReduce(static_cast<scalar>(zTrDeviceDebug), sumOp<scalar>());
            double pTrDeviceDebug = 0.0;
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    reinterpret_cast<const double*>(d_p),
                    1,
                    reinterpret_cast<const double*>(d_p),
                    1,
                    &pTrDeviceDebug
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                cleanup();
                return fail("cublas dot(p,p) debug");
            }
            if (!syncStream())
            {
                return false;
            }
            pTrDebug = returnReduce(static_cast<scalar>(pTrDeviceDebug), sumOp<scalar>());

            double apTrDeviceDebug = 0.0;
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    reinterpret_cast<const double*>(d_Ap),
                    1,
                    reinterpret_cast<const double*>(d_Ap),
                    1,
                    &apTrDeviceDebug
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                cleanup();
                return fail("cublas dot(Ap,Ap) debug");
            }
            if (!syncStream())
            {
                return false;
            }
            apTrDebug = returnReduce(static_cast<scalar>(apTrDeviceDebug), sumOp<scalar>());
        }

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

        if (pAp == scalar(0))
        {
            return fail("pApZero");
        }

        const scalar alphaStep = rz/pAp;
        const double alphaDevice = static_cast<double>(alphaStep);

        if (logCgDebug)
        {
            const double rNorm = std::sqrt(std::max(0.0, static_cast<double>(rTrDebug)));
            const double zNorm = std::sqrt(std::max(0.0, static_cast<double>(zTrDebug)));
            const double pNorm = std::sqrt(std::max(0.0, static_cast<double>(pTrDebug)));
            const double apNorm = std::sqrt(std::max(0.0, static_cast<double>(apTrDebug)));
            std::vector<double> pHost(static_cast<std::size_t>(rows));
            std::vector<double> apHost(static_cast<std::size_t>(rows));
            if
            (
                cudaMemcpy
                (
                    pHost.data(),
                    reinterpret_cast<const double*>(d_p),
                    sizeof(DeviceScalar)*rows,
                    cudaMemcpyDeviceToHost
                ) != cudaSuccess
             || cudaMemcpy
                (
                    apHost.data(),
                    reinterpret_cast<const double*>(d_Ap),
                    sizeof(DeviceScalar)*rows,
                    cudaMemcpyDeviceToHost
                ) != cudaSuccess
            )
            {
                cleanup();
                return fail("cudaMemcpy(host debug)");
            }
            double hostDot = 0.0;
            double hostApNorm = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                hostDot += pHost[i]*apHost[i];
                hostApNorm += apHost[i]*apHost[i];
            }
            hostApNorm = std::sqrt(std::max(0.0, hostApNorm));
            const int sampleCount = std::min(rows, 8);
            std::ostringstream apSampleStream;
            apSampleStream.setf(std::ios::scientific);
            apSampleStream.precision(3);
            apSampleStream << '[';
            for (int i = 0; i < sampleCount; ++i)
            {
                if (i) apSampleStream << ' ';
                apSampleStream << apHost[i];
            }
            apSampleStream << ']';
            Info<< "cudaPCG(" << fieldName_ << "): iter=" << iter
                << " rTr=" << rTrDebug
                << " rz=" << rz
                << " |r|=" << rNorm
                << " |z|=" << zNorm
                << " |p|=" << pNorm
                << " |Ap|=" << apNorm
                << " pAp=" << pAp
                << " host_pAp=" << hostDot
                << " host|Ap|=" << hostApNorm
                << " ApSample=" << apSampleStream.str()
                << " alpha=" << alphaStep
                << nl;
        }

        if (cublasDaxpy(cublasHandle, rows, &alphaDevice, reinterpret_cast<const double*>(d_p), 1, reinterpret_cast<double*>(d_x), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas axpy x += alpha*p");
        const double negAlphaScaled = -alphaDevice;
        if (cublasDaxpy(cublasHandle, rows, &negAlphaScaled, reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
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

        if (useChebyshev)
        {
            const scalar currentResidual = solverPerf.finalResidual();
            if (chebyshevPrevResidual > VSMALL)
            {
                if (currentResidual > scalar(1.5)*chebyshevPrevResidual)
                {
                    ++chebyshevGrowthCounter;
                }
                else
                {
                    chebyshevGrowthCounter = 0;
                }

                if (chebyshevGrowthCounter >= 3)
                {
                    if (!chebyshevRestartAttempted && preconCacheEntry)
                    {
                        chebyshevRestartAttempted = true;
                        chebyshevRestarted = true;
                        chebyshevGrowthCounter = 0;
                        chebyshevPrevResidual = currentResidual;

                        const double inflate = std::max(1.0, static_cast<double>(chebyshevLambdaInflate));
                        chebyshevLambdaInflateUsed = static_cast<scalar>(inflate);
                        if (preconCacheEntry->lambdaReady && inflate > 1.0)
                        {
                            preconCacheEntry->lambdaMax *= inflate;
                            preconCacheEntry->lambdaMin /= inflate;
                            if
                            (
                                !std::isfinite(preconCacheEntry->lambdaMax)
                             || preconCacheEntry->lambdaMax <= VSMALL
                            )
                            {
                                preconCacheEntry->lambdaMax = inflate;
                            }
                            preconCacheEntry->lambdaMin = std::max
                            (
                                preconCacheEntry->lambdaMin,
                                static_cast<double>(chebyshevLambdaMinFloor)
                            );
                            if (preconCacheEntry->lambdaMin >= preconCacheEntry->lambdaMax)
                            {
                                preconCacheEntry->lambdaMin = 0.5*preconCacheEntry->lambdaMax;
                            }
                        }
                        else if (preconCacheEntry)
                        {
                            preconCacheEntry->lambdaReady = false;
                        }

                        if (preconCacheEntry)
                        {
                            chebyshevLambdaMinUsed = static_cast<scalar>(preconCacheEntry->lambdaMin);
                            chebyshevLambdaMaxUsed = static_cast<scalar>(preconCacheEntry->lambdaMax);
                        }
                    }
                    else
                    {
                        chebyshevGuardrailTriggered = true;
                        useChebyshev = false;
                        colourSpdMode = true;
                        useColour = true;
                        chebyshevPrevResidual = -1;
                        chebyshevGrowthCounter = 0;
                    }
                }
            }
            else
            {
                chebyshevGrowthCounter = 0;
            }
            chebyshevPrevResidual = currentResidual;
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

        scalar zTrAfterDebug = 0;
        if (logCgDebug)
        {
            double zAfterDeviceDebug = 0.0;
            if
            (
                cublasDdot
                (
                    cublasHandle,
                    rows,
                    reinterpret_cast<const double*>(d_z),
                    1,
                    reinterpret_cast<const double*>(d_z),
                    1,
                    &zAfterDeviceDebug
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                cleanup();
                return fail("cublas dot(z,z) debugIter");
            }
            if (!syncStream())
            {
                return false;
            }
            zTrAfterDebug = returnReduce(static_cast<scalar>(zAfterDeviceDebug), sumOp<scalar>());
        }

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

        scalar rzCandidate = returnReduce(static_cast<scalar>(rzNewDevice), sumOp<scalar>());
        if (useChebyshev && (!std::isfinite(rzCandidate) || rzCandidate <= scalar(0)))
        {
            chebyshevGuardrailTriggered = true;
            chebyshevDotGuardTriggered = true;
            useChebyshev = false;
            colourSpdMode = true;
            useColour = true;
            chebyshevPrevResidual = -1;
            chebyshevGrowthCounter = 0;
            if (!applyPreconditionerVec(reinterpret_cast<const double*>(d_r), reinterpret_cast<double*>(d_z), "chebDotGuard"))
            {
                return fail(message);
            }
            rzNewDevice = 0.0;
            if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, reinterpret_cast<const double*>(d_z), 1, &rzNewDevice) != CUBLAS_STATUS_SUCCESS)
            {
                cleanup();
                return fail("cublas dot(r,z) chebGuard");
            }
            if (!syncStream())
            {
                return false;
            }
            rzCandidate = returnReduce(static_cast<scalar>(rzNewDevice), sumOp<scalar>());
            if (!std::isfinite(rzCandidate) || rzCandidate <= scalar(0))
            {
                message = word("ChebDotGuardNonPos");
                return fail(message);
            }
        }

        const scalar rzOld = rz;
        rz = rzCandidate;

        const scalar betaStep = rzOld != 0 ? rz/rzOld : 0;
        const double betaDevice = static_cast<double>(betaStep);

        if (logCgDebug)
        {
            const double zNormNew = std::sqrt(std::max(0.0, static_cast<double>(zTrAfterDebug)));
            Info<< "cudaPCG(" << fieldName_ << "): iter=" << iter
                << " newRz=" << rz
                << " beta=" << betaStep
                << " |z_new|=" << zNormNew
                << nl;
        }

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
    {
        scalarField rPhysical(nCells, Zero);
        matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        for (label cell = 0; cell < nCells; ++cell)
        {
            rPhysical[cell] = source[cell] - wA[cell];
        }
        scalar trueSumAbs = gSumMag(rPhysical, matrix_.mesh().comm());
        if (!std::isfinite(static_cast<double>(trueSumAbs)))
        {
            cleanup();
            return fail("physicalResidualNaN");
        }
        scalar trueResidual =
            normFactor > VSMALL
          ? trueSumAbs/normFactor
          : scalar(0);
        if (!std::isfinite(static_cast<double>(trueResidual)))
        {
            cleanup();
            return fail("physicalResidualNaN");
        }
        solverPerf.finalResidual() = trueResidual;
        bestResidualSoFar = min(bestResidualSoFar, solverPerf.finalResidual());
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
    if (preconCacheEntry && icAppliedThisSolve && icCacheWasAllowed)
    {
        preconCacheEntry->recordIterations(solverPerf.nIterations());
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
    (void)chebyshevRequested;
    (void)chebyshevForced;

    if (useColour && autoTuneEnabled(dict))
    {
        colourAutoTuneRecordSuccess(fieldName_);
    }
    return true;
}

#endif
