volScalarField rAU(1.0/UEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
  + MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U, phi, Uf))
);

MRF.makeRelative(phiHbyA);

if (p.needReference())
{
    fvc::makeRelative(phiHbyA, U);
    adjustPhi(phiHbyA, U, p);
    fvc::makeAbsolute(phiHbyA, U);
}

tmp<volScalarField> rAtU(rAU);
volScalarField& rAtUField = const_cast<volScalarField&>(rAtU());

if (pimple.consistent())
{
    rAtU = 1.0/max(1.0/rAU - UEqn.H1(), 0.1/rAU);
    phiHbyA +=
        fvc::interpolate(rAtUField - rAU)*fvc::snGrad(p)*mesh.magSf();
    HbyA -= (rAU - rAtUField)*fvc::grad(p);
}

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, U, phiHbyA, rAtUField, MRF);

// Non-orthogonal pressure corrector loop
while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtUField, p) == fvc::div(phiHbyA)
    );

    pEqn.setReference
    (
        pressureReference.refCell(),
        pressureReference.refValue()
    );

    pEqn.solve();

    if (pimple.finalNonOrthogonalIter())
    {
        surfaceScalarField phiFlux(pEqn.flux());

        bool gpuUpdatedPhi = false;
        Foam::gpu::DeviceField<surfaceScalarField>& phiDev =
            gpuRegistry.getOrCreate(phi);

        if (useGpuFieldOps && gpuContext.ready())
        {
            word gpuError;
            Foam::gpu::OperationStats phiStats;

            Foam::gpu::DeviceField<surfaceScalarField>& phiHbyADev =
                gpuRegistry.getOrCreate(phiHbyA);
            Foam::gpu::DeviceField<surfaceScalarField>& fluxDev =
                gpuRegistry.getOrCreate(phiFlux);

            phiDev.markHostDirty();
            phiHbyADev.markHostDirty();
            fluxDev.markHostDirty();

            bool ok =
                phiDev.syncHostToDevice(gpuContext, gpuError)
             && phiHbyADev.syncHostToDevice(gpuContext, gpuError)
             && fluxDev.syncHostToDevice(gpuContext, gpuError)
             && subtractSurfaceScalarFields
                (
                    gpuContext,
                    phiDev,
                    phiHbyADev,
                    fluxDev,
                    gpuError,
                    &phiStats
                );

            if (ok)
            {
                phiDev.markDeviceDirty();
                ok = phiDev.syncDeviceToHost(gpuContext, gpuError);
            }

            if (ok)
            {
                gpuUpdatedPhi = true;
                if (logGpuFieldOps)
                {
                    gpuOpStats.append
                    (
                        {
                            runTime.timeName(),
                            word("phiCorrector"),
                            phiStats.gpuMilliseconds
                        }
                    );
                }
            }
            else if (!gpuError.empty())
            {
                WarningInFunction
                    << "GPU phi update failed: " << gpuError << nl << endl;
            }
        }

        if (!gpuUpdatedPhi)
        {
            phi = phiHbyA - phiFlux;
            phiDev.markHostDirty();
            if (logGpuFieldOps)
            {
                gpuOpStats.append
                (
                    {
                        runTime.timeName(),
                        word("phiCorrector_cpu"),
                        Foam::scalar(-1)
                    }
                );
            }
        }
    }
}

#include "continuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

tmp<volVectorField> gradPTmp(fvc::grad(p));
volVectorField& gradP = gradPTmp.ref();

bool gpuUpdatedU = false;
Foam::gpu::DeviceField<volVectorField>& uDev = gpuRegistry.getOrCreate(U);

if (useGpuFieldOps && gpuContext.ready())
{
    word gpuError;
    Foam::gpu::OperationStats uStats;

    Foam::gpu::DeviceField<volVectorField>& hbyADev =
        gpuRegistry.getOrCreate(HbyA);
    Foam::gpu::DeviceField<volScalarField>& rAtUDev =
        gpuRegistry.getOrCreate(rAtUField);
    Foam::gpu::DeviceField<volVectorField>& gradPDev =
        gpuRegistry.getOrCreate(gradP);

    uDev.markHostDirty();
    hbyADev.markHostDirty();
    rAtUDev.markHostDirty();
    gradPDev.markHostDirty();

    bool ok =
        uDev.syncHostToDevice(gpuContext, gpuError)
     && hbyADev.syncHostToDevice(gpuContext, gpuError)
     && rAtUDev.syncHostToDevice(gpuContext, gpuError)
     && gradPDev.syncHostToDevice(gpuContext, gpuError)
     && velocityCorrector
        (
            gpuContext,
            uDev,
            hbyADev,
            rAtUDev,
            gradPDev,
            gpuError,
            &uStats
        );

    if (ok)
    {
        uDev.markDeviceDirty();
        ok = uDev.syncDeviceToHost(gpuContext, gpuError);
    }

    if (ok)
    {
        gpuUpdatedU = true;
        if (logGpuFieldOps)
        {
            gpuOpStats.append
            (
                {
                    runTime.timeName(),
                    word("velocityCorrector"),
                    uStats.gpuMilliseconds
                }
            );
        }
    }
    else if (!gpuError.empty())
    {
        WarningInFunction
            << "GPU velocity update failed: " << gpuError << nl << endl;
    }
}

if (!gpuUpdatedU)
{
    U = HbyA - rAtUField*gradP;
    uDev.markHostDirty();
    if (logGpuFieldOps)
    {
        gpuOpStats.append
        (
            {
                runTime.timeName(),
                    word("velocityCorrector_cpu"),
                    Foam::scalar(-1)
            }
        );
    }
}

gradPTmp.clear();

U.correctBoundaryConditions();
fvConstraints.constrain(U);

// Correct Uf if the mesh is moving
fvc::correctUf(Uf, U, phi, MRF);

// Make the fluxes relative to the mesh motion
fvc::makeRelative(phi, U);
