// Minimal header for a runtime-selectable GPU-backed solver (stub)

#ifndef cudaPCG_H
#define cudaPCG_H

#include "lduMatrix.H"

namespace Foam
{

class cudaPCG
:
    public lduMatrix::solver
{
    // not copyable
    cudaPCG(const cudaPCG&) = delete;
    void operator=(const cudaPCG&) = delete;

public:
    TypeName("cudaPCG");

    // Construct from components (matches lduMatrix::solver API)
    cudaPCG
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    // Solve system Ax=b for psi
    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;

#ifdef FOAM_USE_CUDA
private:
    bool gpuSolve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt,
        solverPerformance& solverPerf,
        const label deviceId,
        word& message
    ) const;
#endif
};

} // namespace Foam

#endif
