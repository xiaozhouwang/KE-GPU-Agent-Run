#include "CsrExport.H"
#include "lduAddressing.H"
#include "DynamicList.H"

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

Colouring buildGreedyColouring(const lduMatrix& A)
{
    const scalarField& diag = A.diag();
    const label n = diag.size();

    Colouring result;
    if (!n)
    {
        result.nColours = 0;
        result.colourPtr.setSize(1, 0);
        return result;
    }

    const lduAddressing& addr = A.lduAddr();
    const labelUList& lowerAddr = addr.lowerAddr();
    const labelUList& upperAddr = addr.upperAddr();
    const label nFaces = lowerAddr.size();

    List<DynamicList<label>> adjacency(n);

    for (label face = 0; face < nFaces; ++face)
    {
        const label owner = lowerAddr[face];
        const label neighbour = upperAddr[face];
        adjacency[owner].append(neighbour);
        adjacency[neighbour].append(owner);
    }

    labelList colours(n, -1);
    DynamicList<label> touched;
    List<char> usedColours(0);

    label maxColour = 0;

    for (label cell = 0; cell < n; ++cell)
    {
        DynamicList<label>& nbrs = adjacency[cell];

        forAll(nbrs, i)
        {
            const label nbr = nbrs[i];
            const label colour = colours[nbr];
            if (colour < 0)
            {
                continue;
            }

            if (colour >= usedColours.size())
            {
                const label oldSize = usedColours.size();
                usedColours.setSize(colour + 1);
                for (label j = oldSize; j < usedColours.size(); ++j)
                {
                    usedColours[j] = 0;
                }
            }

            if (!usedColours[colour])
            {
                usedColours[colour] = 1;
                touched.append(colour);
            }
        }

        label chosen = 0;
        for (; chosen < usedColours.size(); ++chosen)
        {
            if (!usedColours[chosen])
            {
                break;
            }
        }

        if (chosen == usedColours.size())
        {
            usedColours.append(0);
        }

        colours[cell] = chosen;
        maxColour = max(maxColour, chosen + 1);

        forAll(touched, ti)
        {
            usedColours[touched[ti]] = 0;
        }
        touched.clear();
    }

    if (!maxColour)
    {
        // All cells isolated – assign a single colour.
        maxColour = 1;
        colours = 0;
    }

    result.nColours = maxColour;
    result.cellToColour.transfer(colours);

    result.colourPtr.setSize(result.nColours + 1);
    result.colourPtr[0] = 0;
    labelList counts(result.nColours, 0);
    forAll(result.cellToColour, idx)
    {
        const label colour = result.cellToColour[idx];
        if (colour >= 0 && colour < result.nColours)
        {
            ++counts[colour];
        }
    }
    for (label c = 0; c < result.nColours; ++c)
    {
        result.colourPtr[c+1] = result.colourPtr[c] + counts[c];
    }

    result.colourIndices.setSize(n);
    labelList cursor(result.colourPtr);
    forAll(result.cellToColour, cell)
    {
        const label colour = result.cellToColour[cell];
        const label pos = cursor[colour]++;
        result.colourIndices[pos] = cell;
    }

    return result;
}

} // namespace cudaPCGDetail
} // namespace Foam
