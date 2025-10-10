#include "CsrExport.H"
#include "lduAddressing.H"

namespace Foam
{
namespace cudaPCGDetail
{

CsrMatrix buildSymmetricCsr(const lduMatrix& A)
{
    const scalarField& diag = A.diag();
    const label n = diag.size();

    const lduAddressing& addr = A.lduAddr();
    const labelUList& lowerAddr = addr.lowerAddr();
    const labelUList& upperAddr = addr.upperAddr();

    const label nFaces = lowerAddr.size();

    // Deg counts per row (start with 1 for the diagonal)
    labelList deg(n, 1);
    for (label f = 0; f < nFaces; ++f)
    {
        const label i = lowerAddr[f];
        const label j = upperAddr[f];
        ++deg[i];
        ++deg[j];
    }

    CsrMatrix M;
    M.nRows = n;
    M.rowPtr.setSize(n+1);
    M.rowPtr[0] = 0;
    for (label i = 0; i < n; ++i)
    {
        M.rowPtr[i+1] = M.rowPtr[i] + deg[i];
    }
    M.nnz = M.rowPtr[n];
    M.colInd.setSize(M.nnz);
    M.values.setSize(M.nnz);

    // Cursor per row
    labelList next(M.rowPtr);

    // Put diagonal entries first
    for (label i = 0; i < n; ++i)
    {
        const label pos = next[i]++;
        M.colInd[pos] = i;
        M.values[pos] = diag[i];
    }

    const scalarField& upper = A.upper();
    const scalarField& lower = A.lower();

    // Off-diagonals: for each face add A(i,j) and A(j,i)
    for (label f = 0; f < nFaces; ++f)
    {
        const label i = lowerAddr[f];
        const label j = upperAddr[f];

        // row i, col j uses upper[f]
        {
            const label pos = next[i]++;
            M.colInd[pos] = j;
            M.values[pos] = upper[f];
        }

        // row j, col i uses lower[f]
        {
            const label pos = next[j]++;
            M.colInd[pos] = i;
            M.values[pos] = lower[f];
        }
    }

    return M;
}

} // namespace cudaPCGDetail
} // namespace Foam

