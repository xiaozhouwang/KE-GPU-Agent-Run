/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    pimpleFoamGPU

Description
    Transient incompressible solver mirroring pimpleFoam with additional GPU
    plumbing.  Device-resident field buffers are registered after field
    creation; subsequent GPU kernels can operate directly on those buffers
    while the CPU path remains unchanged when the GPU backend is unavailable.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "viscosityModel.H"
#include "incompressibleMomentumTransportModels.H"
#include "pimpleControl.H"
#include "pressureReference.H"
#include "CorrectPhi.H"
#include "fvModels.H"
#include "fvConstraints.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "DeviceField.H"
#include "GpuContext.H"
#include "FieldOps.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace
{

inline void registerGpuFields
(
    Foam::gpu::FieldRegistry& registry,
    Foam::gpu::Context& context,
    Foam::volVectorField& U,
    Foam::volScalarField& p,
    Foam::surfaceScalarField& phi
)
{
    word error;

    auto& uDevice = registry.getOrCreate(U);
    auto& pDevice = registry.getOrCreate(p);
    auto& phiDevice = registry.getOrCreate(phi);

    if (context.ready())
    {
        if (!uDevice.syncHostToDevice(context, error))
        {
            WarningInFunction << "U host->device sync failed: " << error << nl;
        }

        if (!pDevice.syncHostToDevice(context, error))
        {
            WarningInFunction << "p host->device sync failed: " << error << nl;
        }

        if (!phiDevice.syncHostToDevice(context, error))
        {
            WarningInFunction << "phi host->device sync failed: " << error << nl;
        }
    }
    else
    {
        WarningInFunction
            << "GPU context unavailable – continuing with CPU execution" << nl;
    }
}

} // namespace

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"

    Foam::gpu::Context& gpuContext = Foam::gpu::globalContext();
    Foam::gpu::FieldRegistry& gpuRegistry =
        Foam::gpu::FieldRegistry::New(mesh);

    registerGpuFields(gpuRegistry, gpuContext, U, p, phi);

    const Switch useGpuFieldOps
    (
        pimple.dict().lookupOrDefault<Switch>("useGpuFieldOps", false)
    );

    const Switch logGpuFieldOps
    (
        pimple.dict().lookupOrDefault<Switch>("logGpuFieldOps", false)
    );

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    Info<< "\nStarting time loop\n" << endl;

    while (pimple.run(runTime))
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        fvModels.preUpdateMesh();
        mesh.update();
        runTime++;

        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        while (pimple.loop())
        {
            if (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
            {
                mesh.move();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        #include "correctPhi.H"
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            fvModels.correct();

            #include "UEqn.H"

            while (pimple.correct())
            {
                #include "pEqnGPU.H"
            }

            if (pimple.turbCorr())
            {
                viscosity->correct();
                turbulence->correct();
            }
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}


// ************************************************************************* //
