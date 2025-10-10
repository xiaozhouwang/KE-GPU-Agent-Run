#include "cudaPCG.H"
#include "PCG.H"
#include "CsrExport.H"
#include "Switch.H"
#include "List.H"

#include <algorithm>

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

    if (cublasDdgmm(cublasHandle, CUBLAS_SIDE_LEFT, rows, 1, reinterpret_cast<const double*>(d_r), rows, reinterpret_cast<const double*>(d_diagInv), 1, reinterpret_cast<double*>(d_z), rows) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas dgmm diagInv*r");

    if (cublasDcopy(cublasHandle, rows, reinterpret_cast<const double*>(d_z), 1, reinterpret_cast<double*>(d_p), 1) != CUBLAS_STATUS_SUCCESS)
        return fail("cublas copy z->p");

    scalarField rHost(nCells, Zero);
    scalarField zHost(nCells, Zero);
    scalarField pHost(nCells, Zero);
    scalarField wA(nCells, Zero);
    scalarField pA(nCells, Zero);

    cudaMemcpy(rHost.begin(), d_r, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(zHost.begin(), d_z, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(wA.begin(), d_Ap, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(pHost.begin(), d_p, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

    scalar rz = gSumProd(rHost, zHost, matrix_.mesh().comm());

    scalar normFactor = this->normFactor(psi, source, wA, pA);

    solverPerf.initialResidual() =
        gSumMag(rHost, matrix_.mesh().comm())
       /normFactor;
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

        cudaMemcpy(wA.begin(), d_Ap, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);
        cudaMemcpy(pHost.begin(), d_p, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

        scalar pAp = gSumProd(wA, pHost, matrix_.mesh().comm());

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

        cudaMemcpy(rHost.begin(), d_r, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

        solverPerf.finalResidual() =
            gSumMag(rHost, matrix_.mesh().comm())
           /normFactor;

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

        if (cublasDdgmm(cublasHandle, CUBLAS_SIDE_LEFT, rows, 1, reinterpret_cast<const double*>(d_r), rows, reinterpret_cast<const double*>(d_diagInv), 1, reinterpret_cast<double*>(d_z), rows) != CUBLAS_STATUS_SUCCESS)
            return fail("cublas dgmm diagInv*r (iter)");

        cudaMemcpy(zHost.begin(), d_z, sizeof(DeviceScalar)*rows, cudaMemcpyDeviceToHost);

        const scalar rzOld = rz;
        rz = gSumProd(rHost, zHost, matrix_.mesh().comm());

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
