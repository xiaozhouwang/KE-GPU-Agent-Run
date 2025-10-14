/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Class
        Foam::gpu::FieldHandle

    Description
        Lightweight placeholder that marks the intended interface for GPU
        resident fields.  The implementation is intentionally minimal right
        now – it simply wraps a host-side field and records a future device
        pointer slot.  Phase A/B of the acceleration roadmap will replace this
        with an owning GPU container and asynchronous transfer hooks.

\*---------------------------------------------------------------------------*/

#ifndef Foam_gpu_FieldHandle_H
#define Foam_gpu_FieldHandle_H

#include "scalarField.H"

namespace Foam
{
namespace gpu
{

class FieldHandle
{
    //- Host mirror (temporary stand‑in until GPU storage lands)
    scalarField* hostPtr_;

    //- Placeholder for a device pointer
    void* devicePtr_;

public:
    explicit FieldHandle(scalarField& host)
    :
        hostPtr_(&host),
        devicePtr_(nullptr)
    {}

    scalarField& host() { return *hostPtr_; }
    const scalarField& host() const { return *hostPtr_; }

    void* devicePointer() const { return devicePtr_; }
};

} // namespace gpu
} // namespace Foam

#endif

// ************************************************************************* //
