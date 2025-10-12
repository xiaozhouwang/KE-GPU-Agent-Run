#include "cudaPCG.H"
#include "PCG.H"
#include "CsrExport.H"
#include "Switch.H"
#include "List.H"
#include "PstreamReduceOps.H"

#include <algorithm>
#include <limits>

#ifdef FOAM_USE_CUDA
#include <cuda_runtime.h>
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

#ifdef FOAM_USE_CUDA
    if (requestedGpu)
    {
        solverPerformance gpuPerf
        (
            lduMatrix::preconditioner::getName(controlDict_) + typeName,
            fieldName_
        );

        word message;
        if (gpuSolve(psi, source, cmpt, gpuPerf, deviceId, message))
        {
            return gpuPerf;
        }

        WarningInFunction
            << "cudaPCG GPU path disabled: " << message << nl
            << "Falling back to CPU PCG." << endl;
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
    word& message
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
        controlDict_.lookupOrDefault<Switch>("reportGpuStats", Switch(false));

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

    if (reportStats)
    {
        const scalarField& diag = matrix_.diag();
        scalar minDiag = std::numeric_limits<scalar>::max();
        scalar maxDiag = -std::numeric_limits<scalar>::max();
        for (const scalar d : diag)
        {
            minDiag = std::min(minDiag, d);
            maxDiag = std::max(maxDiag, d);
        }
        Info<< "cudaPCG: diagonal range [" << minDiag << ", " << maxDiag << "]"
            << nl;
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

    const scalarField& diag = matrix_.diag();
    List<DeviceScalar> diagInv(diag.size());
    forAll(diag, i)
    {
        const scalar d = diag[i];
        diagInv[i] = mag(d) > VSMALL ? static_cast<DeviceScalar>(1.0/d) : static_cast<DeviceScalar>(0);
    }

    DeviceScalar *d_vals=nullptr, *d_x=nullptr, *d_b=nullptr;
    DeviceScalar *d_r=nullptr, *d_p=nullptr, *d_Ap=nullptr;
    DeviceScalar *d_z=nullptr, *d_diagInv=nullptr;
    int *d_rowPtr=nullptr, *d_colInd=nullptr;

    // Optional SGS preconditioner (D+L) solves
    bool useSGS = false;
    {
        const word precName = controlDict_.lookupOrDefault<word>("preconditioner", word("diagonal"));
        const std::string pl = toLower(precName);
        useSGS = (pl == "sgs" || pl == "symgaussseidel" || pl == "ic" || pl == "dic");
    }
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
    };

    auto fail = [&](const word& msg)
    {
        cleanup();
        message = msg;
        return false;
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

    // Copy
    if (cudaMemcpy(d_vals, values.begin(), sizeof(DeviceScalar)*nnz, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(vals)");
    if (cudaMemcpy(d_rowPtr, rowPtr.begin(), sizeof(int)*(rows+1), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(rowPtr)");
    if (cudaMemcpy(d_colInd, colInd.begin(), sizeof(int)*nnz, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(colInd)");
    if (cudaMemcpy(d_x, psi.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(x)");
    if (cudaMemcpy(d_b, source.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(b)");
    if (cudaMemcpy(d_diagInv, diagInv.begin(), sizeof(DeviceScalar)*rows, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy(diagInv)");

    if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS)
        return fail("cusparseCreate");
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
        return fail("cublasCreate");

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

    if (cusparseCreateDnVec(&vecX, rows, d_x, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        return fail("cusparseCreateDnVec(x)");
    if (cusparseCreateDnVec(&vecY, rows, d_Ap, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        return fail("cusparseCreateDnVec(y)");

    // Build optional SGS preconditioner structures: CSR of (D+L)
    bool sgsReady = false;
    label polySweepsCfg = controlDict_.lookupOrDefault<label>("polyJacobiSweeps", 0);
    scalar jacOmegaCfg = controlDict_.lookupOrDefault<scalar>("jacobiOmega", scalar(0.7));
    const Switch polyAuto = controlDict_.lookupOrDefault<Switch>("polyJacobiAuto", Switch(true));
    bool usePolyJacobi = (polySweepsCfg > 0) || polyAuto;
    label polySweepsEff = polySweepsCfg;
    double jacOmegaEff = static_cast<double>(jacOmegaCfg);
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
                    // Optional tiny diagonal regularisation if needed
                    DeviceScalar v = values[k];
                    if (j == i && std::abs(v) < 1e-20) v = (v >= 0 ? 1 : -1)*1e-20;
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
                        }
                    }
                }
            }
        }
    }

    auto applyPreconditionerVec =
        [&](const double* rhs, double* out, const char* stage) -> bool
    {
        (void)stage;
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
        else
        {
            // Diagonal scaling (Jacobi)
            if (cublasDcopy(cublasHandle, rows, rhs, 1, out, 1) != CUBLAS_STATUS_SUCCESS)
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
                    out,
                    rows,
                    reinterpret_cast<const double*>(d_diagInv),
                    1,
                    out,
                    rows
                ) != CUBLAS_STATUS_SUCCESS
            )
            {
                message = word("DiagScaleFail");
                return false;
            }
            return true;
        }
    };

    const double alpha = 1.0;
    const double zero = 0.0;
    const double negOne = -1.0;

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

    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_b), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas copy b->r");
    if (cublasDaxpy(cublasHandle, rows, reinterpret_cast<const double*>(&negOne), reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas axpy r -= Ap");

    if (!applyPreconditionerVec(reinterpret_cast<const double*>(d_r), reinterpret_cast<double*>(d_z), "initial"))
        return fail(message);

    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas copy z->p");

    scalarField wA(nCells, Zero);
    scalarField pA(nCells, Zero);

    cudaMemcpy(wA.begin(), d_Ap, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(pA.begin(), d_p, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

    double rzDevice = 0.0;
    if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, reinterpret_cast<const double*>(d_z), 1, &rzDevice) != CUBLAS_STATUS_SUCCESS)
    {
        cleanup();
        return fail("cublas dot(r,z)");
    }
    scalar rz = returnReduce(static_cast<scalar>(rzDevice), sumOp<scalar>());

    scalar normFactor = this->normFactor(psi, source, wA, pA);

    double sumAbsDevice = 0.0;
    if (cublasDasum(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, &sumAbsDevice) != CUBLAS_STATUS_SUCCESS)
    {
        cleanup();
        return fail("cublas dasum(r)");
    }
    scalar sumAbs = returnReduce(static_cast<scalar>(sumAbsDevice), sumOp<scalar>());

    solverPerf.initialResidual() = sumAbs/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if
    (
        minIter_ <= 0
     && solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        cudaMemcpy(psi.begin(), d_x, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
        cleanup();
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
        if
        (
            cusparseDnVecSetValues(vecX, d_p) != CUSPARSE_STATUS_SUCCESS
         || cusparseDnVecSetValues(vecY, d_Ap) != CUSPARSE_STATUS_SUCCESS
         || cusparseSpMV
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

        // Restore vecX to x for completeness
        cusparseDnVecSetValues(vecX, d_x);
        cusparseDnVecSetValues(vecY, d_Ap);

        double pApDevice = 0.0;
        if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_p), 1, reinterpret_cast<const double*>(d_Ap), 1, &pApDevice) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dot(p,Ap)");
        }
        scalar pAp = returnReduce(static_cast<scalar>(pApDevice), sumOp<scalar>());

        if (solverPerf.checkSingularity(mag(pAp)/normFactor))
        {
            break;
        }

        const scalar alphaStep = rz/pAp;
        const double alphaDevice = static_cast<double>(alphaStep);
        const double negAlphaDevice = -alphaDevice;

        if (cublasDaxpy(cublasHandle, rows, &alphaDevice, reinterpret_cast<const double*>(d_p), 1, reinterpret_cast<double*>(d_x), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas axpy x += alpha*p");
        if (cublasDaxpy(cublasHandle, rows, &negAlphaDevice, reinterpret_cast<const double*>(d_Ap), 1, reinterpret_cast<double*>(d_r), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas axpy r -= alpha*Ap");

        if (cublasDasum(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, &sumAbsDevice) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dasum(r iter)");
        }
        sumAbs = returnReduce(static_cast<scalar>(sumAbsDevice), sumOp<scalar>());

        solverPerf.finalResidual() = sumAbs/normFactor;

        ++iter;
        solverPerf.nIterations() = iter;

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

        double rzNewDevice = 0.0;
        if (cublasDdot(cublasHandle, rows, reinterpret_cast<const double*>(d_r), 1, reinterpret_cast<const double*>(d_z), 1, &rzNewDevice) != CUBLAS_STATUS_SUCCESS)
        {
            cleanup();
            return fail("cublas dot(r,z) iter");
        }

        const scalar rzOld = rz;
        rz = returnReduce(static_cast<scalar>(rzNewDevice), sumOp<scalar>());

        const scalar betaStep = rzOld != 0 ? rz/rzOld : 0;
        const double betaDevice = static_cast<double>(betaStep);

        if (cublasDscal(cublasHandle, rows, &betaDevice, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas scal p*=beta");
        if (cublasDaxpy(cublasHandle, rows, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas axpy p+=z");
    }

    cudaMemcpy(psi.begin(), d_x, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cleanup();
    return true;
}

#endif
