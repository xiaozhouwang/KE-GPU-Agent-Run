/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025
     \\/     M anipulation  |
-------------------------------------------------------------------------------
   Description
       GPU-development staging solver.  For now this simply re-uses the
       upstream pimpleFoam implementation so that build infrastructure is in
       place for future GPU-specific work that will live alongside (not in
       place of) the CPU code.
\*---------------------------------------------------------------------------*/

#include "pimpleFoam.C"

// ************************************************************************* //
