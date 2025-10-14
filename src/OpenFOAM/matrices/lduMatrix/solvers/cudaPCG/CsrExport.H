/*
    Minimal CSR exporter for OpenFOAM lduMatrix (symmetric full CSR)
    Phase 1 utility – local to cudaPCG implementation.
*/

#ifndef CsrExport_H
#define CsrExport_H

#include "lduMatrix.H"
#include "List.H"
#include "labelList.H"
#include "scalarField.H"

namespace Foam
{
namespace cudaPCGDetail
{

struct CsrMatrix
{
    label nRows;
    label nnz;
    labelList rowPtr;   // size nRows+1
    labelList colInd;   // size nnz
    scalarField values; // size nnz
};

struct Colouring
{
    //- Number of colours produced by the greedy colouring
    label nColours;

    //- Colour index per cell/row (size nRows)
    labelList cellToColour;

    //- Offsets into colourIndices for each colour (size nColours+1)
    labelList colourPtr;

    //- Row indices grouped by colour
    labelList colourIndices;
};

// Build a full symmetric CSR (diagonal + both off-diagonal entries)
CsrMatrix buildSymmetricCsr(const lduMatrix& A);

// Greedy graph colouring for the cell adjacency implied by the matrix
Colouring buildGreedyColouring(const lduMatrix& A);

} // namespace cudaPCGDetail
} // namespace Foam

#endif
