#ifndef Foam_gpu_FvOperators_H
#define Foam_gpu_FvOperators_H

#include "DeviceField.H"
#include "FieldOps.H"
#include "fvMatrix.H"
#include "volFieldsFwd.H"
#include "tmp.H"

namespace Foam
{
namespace gpu
{
namespace fvm
{

//- Assemble the Euler fvm::ddt operator on the GPU for volume fields.
//  Currently supports the Euler ddt scheme on stationary meshes. Returns an
//  empty tmp<> and sets usedGpu=false when falling back to the CPU path.
template<class Type>
tmp<fvMatrix<Type>> ddt
(
    Context& ctx,
    FieldRegistry& registry,
    GeometricField<Type, fvPatchField, volMesh>& vf,
    bool& usedGpu,
    word& error,
    OperationStats* stats = nullptr
);

//- Assemble the convection operator fvm::div(faceFlux, vf) on the GPU.
//  Supports Gauss-based schemes (linearUpwind/upwind) with CPU corrections for
//  boundary terms. Returns empty tmp<> with usedGpu=false on fallback.
template<class Type>
tmp<fvMatrix<Type>> div
(
    Context& ctx,
    FieldRegistry& registry,
    const surfaceScalarField& faceFlux,
    GeometricField<Type, fvPatchField, volMesh>& vf,
    bool& usedGpu,
    word& error,
    OperationStats* stats = nullptr
);

//- Assemble the laplacian operator fvm::laplacian(gamma, vf) on the GPU.
//  Currently targets scalar diffusivity fields and Gauss schemes. Returns an
//  empty tmp<> with usedGpu=false on fallback.
template<class Type>
tmp<fvMatrix<Type>> laplacian
(
    Context& ctx,
    FieldRegistry& registry,
    const GeometricField<scalar, fvPatchField, volMesh>& gamma,
    GeometricField<Type, fvPatchField, volMesh>& vf,
    bool& usedGpu,
    word& error,
    OperationStats* stats = nullptr
);

} // namespace fvm
} // namespace gpu
} // namespace Foam

#endif
