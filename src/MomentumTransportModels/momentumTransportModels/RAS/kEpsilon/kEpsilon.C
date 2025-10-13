/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2022 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kEpsilon.H"
#include "fvModels.H"
#include "fvConstraints.H"
#include "bound.H"
#include "Switch.H"
#include "GpuContext.H"
#include "DeviceField.H"
#include "FieldOps.H"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{


template<class BasicMomentumTransportModel>
void kEpsilon<BasicMomentumTransportModel>::refreshGpuConfig()
{
    bool requestedGpu = true;

    Switch gpuSwitch(false);
    if (gpuSwitch.readIfPresent("useGPU", this->coeffDict_))
    {
        requestedGpu = gpuSwitch;
    }

    const auto normaliseDeviceWord = [](word backend) -> std::string
    {
        std::string result(backend);
        std::transform
        (
            result.begin(),
            result.end(),
            result.begin(),
            [](unsigned char c){ return std::tolower(c); }
        );
        return result;
    };

    if (!requestedGpu)
    {
        const word backend = this->coeffDict_.template lookupOrDefault<word>
        (
            "device",
            "cpu"
        );

        const std::string backendLower = normaliseDeviceWord(backend);
        requestedGpu = backendLower == "gpu" || backendLower == "cuda";
    }

    if (!requestedGpu && this->coeffDict_.found("backend"))
    {
        const word altBackend = this->coeffDict_.template lookup<word>("backend");
        const std::string backendLower = normaliseDeviceWord(altBackend);
        requestedGpu = backendLower == "gpu" || backendLower == "cuda";
    }

    useGpu_ = requestedGpu;

#ifdef FOAM_USE_CUDA
    if (useGpu_ && !Foam::gpu::available())
    {
        WarningInFunction
            << "kEpsilon GPU backend requested but CUDA context is unavailable"
            << nl << "Continuing with CPU implementation." << endl;
        useGpu_ = false;
    }
#else
    if (useGpu_)
    {
        WarningInFunction
            << "kEpsilon GPU backend requested but this build lacks CUDA support"
            << nl << "Continuing with CPU implementation." << endl;
        useGpu_ = false;
    }
#endif
}

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
void kEpsilon<BasicMomentumTransportModel>::correctNut()
{
    bool usedGpu = false;

    if (useGpu_)
    {
        Foam::gpu::Context& ctx = Foam::gpu::globalContext();
        if (ctx.ready())
        {
            Foam::gpu::FieldRegistry& registry =
                Foam::gpu::FieldRegistry::New(this->mesh_);

            Foam::gpu::DeviceField<volScalarField>& nutDev =
                registry.getOrCreate(this->nut_);
            Foam::gpu::DeviceField<volScalarField>& kDev =
                registry.getOrCreate(k_);
            Foam::gpu::DeviceField<volScalarField>& epsilonDev =
                registry.getOrCreate(epsilon_);

            word gpuError;
            if
            (
                Foam::gpu::computeNutFromKEpsilon
                (
                    ctx,
                    nutDev,
                    kDev,
                    epsilonDev,
                    Cmu_.value(),
                    this->kMin_.value(),
                    this->epsilonMin_.value(),
                    gpuError,
                    nullptr
                )
            )
            {
                usedGpu = true;
            }
            else if (!gpuError.empty())
            {
                WarningInFunction
                    << "kEpsilon GPU nut update failed: "
                    << gpuError << nl << "Falling back to CPU path." << endl;
            }
        }
    }

    if (!usedGpu)
    {
        this->nut_ = Cmu_*sqr(k_)/epsilon_;
    }
    else
    {
        const scalar CmuVal = Cmu_.value();
        const scalar kMinVal = this->kMin_.value();
        const scalar epsMinVal = this->epsilonMin_.value();

        forAll(this->nut_.boundaryField(), patchI)
        {
            scalarField& nutPatch = this->nut_.boundaryFieldRef()[patchI];
            const scalarField& kPatch = k_.boundaryField()[patchI];
            const scalarField& epsPatch = epsilon_.boundaryField()[patchI];

            forAll(nutPatch, faceI)
            {
                const scalar kVal = Foam::max(kPatch[faceI], kMinVal);
                const scalar epsVal = Foam::max(epsPatch[faceI], epsMinVal);
                nutPatch[faceI] = CmuVal*kVal*kVal/epsVal;
            }
        }
    }

    this->nut_.correctBoundaryConditions();
    fvConstraints::New(this->mesh_).constrain(this->nut_);

    if (useGpu_)
    {
        Foam::gpu::Context& ctx = Foam::gpu::globalContext();
        if (ctx.ready())
        {
            Foam::gpu::FieldRegistry& registry =
                Foam::gpu::FieldRegistry::New(this->mesh_);

            word syncError;
            Foam::gpu::DeviceField<volScalarField>& nutDev =
                registry.getOrCreate(this->nut_);
            Foam::gpu::DeviceField<volScalarField>& kDev =
                registry.getOrCreate(k_);
            Foam::gpu::DeviceField<volScalarField>& epsDev =
                registry.getOrCreate(epsilon_);

            nutDev.markHostDirty();
            kDev.markHostDirty();
            epsDev.markHostDirty();

            if
            (
                !nutDev.syncHostToDevice(ctx, syncError)
             || !kDev.syncHostToDevice(ctx, syncError)
             || !epsDev.syncHostToDevice(ctx, syncError)
            )
            {
                WarningInFunction
                    << "Failed to synchronise turbulence fields to GPU: "
                    << syncError << endl;
            }
        }
    }
}


template<class BasicMomentumTransportModel>
tmp<fvScalarMatrix> kEpsilon<BasicMomentumTransportModel>::kSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            k_,
            dimVolume*this->rho_.dimensions()*k_.dimensions()
            /dimTime
        )
    );
}


template<class BasicMomentumTransportModel>
tmp<fvScalarMatrix> kEpsilon<BasicMomentumTransportModel>::epsilonSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            epsilon_,
            dimVolume*this->rho_.dimensions()*epsilon_.dimensions()
            /dimTime
        )
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
kEpsilon<BasicMomentumTransportModel>::kEpsilon
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const viscosity& viscosity,
    const word& type
)
:
    eddyViscosity<RASModel<BasicMomentumTransportModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        viscosity
    ),

    Cmu_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cmu",
            this->coeffDict_,
            0.09
        )
    ),
    C1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C1",
            this->coeffDict_,
            1.44
        )
    ),
    C2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C2",
            this->coeffDict_,
            1.92
        )
    ),
    C3_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C3",
            this->coeffDict_,
            0
        )
    ),
    sigmak_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmak",
            this->coeffDict_,
            1.0
        )
    ),
    sigmaEps_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaEps",
            this->coeffDict_,
            1.3
        )
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    epsilon_
    (
        IOobject
        (
            IOobject::groupName("epsilon", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    useGpu_(false)
{
    bound(k_, this->kMin_);
    bound(epsilon_, this->epsilonMin_);

    refreshGpuConfig();

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
bool kEpsilon<BasicMomentumTransportModel>::read()
{
    if (eddyViscosity<RASModel<BasicMomentumTransportModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        C1_.readIfPresent(this->coeffDict());
        C2_.readIfPresent(this->coeffDict());
        C3_.readIfPresent(this->coeffDict());
        sigmak_.readIfPresent(this->coeffDict());
        sigmaEps_.readIfPresent(this->coeffDict());

        refreshGpuConfig();

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicMomentumTransportModel>
void kEpsilon<BasicMomentumTransportModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    volScalarField& nut = this->nut_;
    const Foam::fvModels& fvModels(Foam::fvModels::New(this->mesh_));
    const Foam::fvConstraints& fvConstraints
    (
        Foam::fvConstraints::New(this->mesh_)
    );

    eddyViscosity<RASModel<BasicMomentumTransportModel>>::correct();

    volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    volScalarField::Internal G
    (
        this->GName(),
        nut()*(dev(twoSymm(tgradU().v())) && tgradU().v())
    );
    tgradU.clear();

    // Update epsilon and G at the wall
    epsilon_.boundaryFieldRef().updateCoeffs();

    // Dissipation equation
    tmp<fvScalarMatrix> epsEqn
    (
        fvm::ddt(alpha, rho, epsilon_)
      + fvm::div(alphaRhoPhi, epsilon_)
      - fvm::laplacian(alpha*rho*DepsilonEff(), epsilon_)
     ==
        C1_*alpha()*rho()*G*epsilon_()/k_()
      - fvm::SuSp(((2.0/3.0)*C1_ - C3_)*alpha()*rho()*divU, epsilon_)
      - fvm::Sp(C2_*alpha()*rho()*epsilon_()/k_(), epsilon_)
      + epsilonSource()
      + fvModels.source(alpha, rho, epsilon_)
    );

    epsEqn.ref().relax();
    fvConstraints.constrain(epsEqn.ref());
    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());
    solve(epsEqn);
    fvConstraints.constrain(epsilon_);
    bound(epsilon_, this->epsilonMin_);

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*G
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(alpha()*rho()*epsilon_()/k_(), k_)
      + kSource()
      + fvModels.source(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvConstraints.constrain(kEqn.ref());
    solve(kEqn);
    fvConstraints.constrain(k_);
    bound(k_, this->kMin_);

    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
