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
    bool usedGpuLap = false;
    Foam::gpu::OperationStats lapStats;
    word gpuLapError;
    tmp<fvScalarMatrix> tGpuLap;
    bool forcePressureOverwrite = false;
    tmp<surfaceScalarField> forcedPhiTmp;

    if (useGpuFieldOps && gpuContext.ready())
    {
        tGpuLap = Foam::gpu::fvm::laplacian
        (
            gpuContext,
            gpuRegistry,
            rAtUField,
            p,
            usedGpuLap,
            gpuLapError,
            logGpuFieldOps ? &lapStats : nullptr
        );

        if (!usedGpuLap && !gpuLapError.empty())
        {
            WarningInFunction
                << "GPU fvm::laplacian() fallback: "
                << gpuLapError << nl << endl;
        }
    }

    tmp<fvScalarMatrix> tLap;
    if (usedGpuLap)
    {
        if (logGpuFieldOps)
        {
            tmp<fvScalarMatrix> tLapCpuCheck = fvm::laplacian(rAtUField, p);
            fvScalarMatrix& lapGpu = tGpuLap.ref();
            const fvScalarMatrix& lapCpu = tLapCpuCheck();
            const bool lapAsymmetricBefore = lapGpu.asymmetric();

            tmp<fvScalarMatrix> tLapGpuInspect = lapGpu.clone();
            const fvScalarMatrix& lapGpuInspect = tLapGpuInspect();

            const scalarField& diagGpu = lapGpuInspect.diag();
            const scalarField& diagCpu = lapCpu.diag();
            const scalarField& cellVolumes = p.mesh().V();
            scalar diagDiffSq = 0.0;
            scalar diagRefSq = 0.0;
            forAll(diagGpu, celli)
            {
                const scalar diff = diagGpu[celli] - diagCpu[celli];
                const scalar w = cellVolumes[celli];
                diagDiffSq += w*diff*diff;
                diagRefSq += w*diagCpu[celli]*diagCpu[celli];
            }
            diagRefSq = max(diagRefSq, VSMALL);

            const scalarField& upperGpu = lapGpuInspect.upper();
            const scalarField& upperCpu = lapCpu.upper();
            scalar upperDiffSq = 0.0;
            scalar upperRefSq = 0.0;
            forAll(upperGpu, facei)
            {
                const scalar diff = upperGpu[facei] - upperCpu[facei];
                upperDiffSq += diff*diff;
                upperRefSq += upperCpu[facei]*upperCpu[facei];
            }
            upperRefSq = max(upperRefSq, VSMALL);

            const scalarField& sourceGpu = lapGpuInspect.source();
            const scalarField& sourceCpu = lapCpu.source();
            scalar sourceDiffSq = 0.0;
            scalar sourceRefSq = 0.0;
            forAll(sourceGpu, celli)
            {
                const scalar diff = sourceGpu[celli] - sourceCpu[celli];
                const scalar w = cellVolumes[celli];
                sourceDiffSq += w*diff*diff;
                sourceRefSq += w*sourceCpu[celli]*sourceCpu[celli];
            }
            sourceRefSq = max(sourceRefSq, VSMALL);

            Info<< "GPU fvm::laplacian relative mismatches: diag="
                << Foam::sqrt(diagDiffSq/diagRefSq)
                << ", upper=" << Foam::sqrt(upperDiffSq/upperRefSq)
                << ", source=" << Foam::sqrt(sourceDiffSq/sourceRefSq)
                << ", asymmetric(before)=" << (lapAsymmetricBefore ? "yes" : "no")
                << nl;

            gpuOpStats.append
            (
                {
                    runTime.timeName(),
                    word("fvmLaplacian"),
                    lapStats.gpuMilliseconds
                }
            );
        }

        tLap = std::move(tGpuLap);
    }
    else
    {
        tLap = fvm::laplacian(rAtUField, p);
    }

    tmp<volScalarField> tDivPhiHbyA(fvc::div(phiHbyA));
    const volScalarField& divPhiHbyA = tDivPhiHbyA();

    fvScalarMatrix pEqn
    (
        tLap
     == divPhiHbyA
    );

    scalarField pSourcePreReference;
    if (logGpuFieldOps)
    {
        pSourcePreReference = pEqn.source();
    }

    const label refCell = pressureReference.refCell();
    const scalar refValue = pressureReference.refValue();

    if (logGpuFieldOps)
    {
        Info<< "pressureReference: cell=" << refCell
            << ", value=" << refValue << nl;
        Info<< "div(phiHbyA) stats: min="
            << gMin(divPhiHbyA.internalField())
            << ", max=" << gMax(divPhiHbyA.internalField())
            << ", mean=" << gAverage(divPhiHbyA.internalField()) << nl;
    }

    pEqn.setReference(refCell, refValue);

    if (logGpuFieldOps)
    {
        const scalarField& sourcePostReference = pEqn.source();
        Info<< "GPU pressure source preRef stats: min="
            << gMin(pSourcePreReference)
            << ", max=" << gMax(pSourcePreReference)
            << ", mean=" << gAverage(pSourcePreReference)
            << ", magMax=" << gMax(mag(pSourcePreReference)) << nl;
        Info<< "GPU pressure source postRef stats: min="
            << gMin(sourcePostReference)
            << ", max=" << gMax(sourcePostReference)
            << ", mean=" << gAverage(sourcePostReference)
            << ", magMax=" << gMax(mag(sourcePostReference)) << nl;
    }

    const fvMesh& pMesh = p.mesh();
    const bool finalIteration =
        !pMesh.schemes().steady()
     && pMesh.data::lookupOrDefault<bool>("finalIteration", false);

    const dictionary& baseSolverControls =
        pMesh.solution().solverDict(p.select(finalIteration));

    if (useGpuFieldOps && gpuContext.ready())
    {
        word solverType;
        if
        (
            baseSolverControls.readIfPresent("solver", solverType)
         && solverType == "GAMG"
        )
        {
            dictionary solverControls(baseSolverControls);
            solverControls.set("cacheAgglomeration", false);

            if (logGpuFieldOps)
            {
                Info<< "GPU pressure solve disabling GAMG cacheAgglomeration"
                    << nl;
            }

            pEqn.solve(solverControls);
        }
        else
        {
            pEqn.solve(baseSolverControls);
        }
    }
    else
    {
        pEqn.solve(baseSolverControls);
    }

    const bool needCpuShadowSolve =
        static_cast<bool>(logGpuFieldOps) || static_cast<bool>(forceCpuPressureCorrector);

    if (needCpuShadowSolve)
    {
        if (logGpuFieldOps)
        {
            const scalarField& pField = p.internalField();
            Info<< "GPU pressure field pre-shadow stats: min="
                << gMin(pField)
                << ", max=" << gMax(pField)
                << ", mean=" << gAverage(pField)
                << ", magMax=" << gMax(mag(pField)) << nl;
        }

        volScalarField pCpuCheck
        (
            IOobject
            (
                "p_cpuCheck",
                runTime.timeName(),
                p.mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            p
        );

        pCpuCheck.rename(p.name());

        const scalarField& cellVolumes = p.mesh().V();

        tmp<fvScalarMatrix> tCpuEqn
        (
            fvm::laplacian(rAtUField, pCpuCheck)
         == divPhiHbyA
        );
        fvScalarMatrix& cpuEqn = tCpuEqn.ref();

        scalarField cpuSourcePreReference;
        if (logGpuFieldOps)
        {
            cpuSourcePreReference = cpuEqn.source();
        }

        cpuEqn.setReference(refCell, refValue);

        if (logGpuFieldOps)
        {
            const scalarField& cpuSourcePostReference = cpuEqn.source();
            Info<< "CPU shadow pressure source preRef stats: min="
                << gMin(cpuSourcePreReference)
                << ", max=" << gMax(cpuSourcePreReference)
                << ", mean=" << gAverage(cpuSourcePreReference)
                << ", magMax=" << gMax(mag(cpuSourcePreReference)) << nl;
            Info<< "CPU shadow pressure source postRef stats: min="
                << gMin(cpuSourcePostReference)
                << ", max=" << gMax(cpuSourcePostReference)
                << ", mean=" << gAverage(cpuSourcePostReference)
                << ", magMax=" << gMax(mag(cpuSourcePostReference)) << nl;

            scalar srcPreDiffSq = 0.0;
            scalar srcPreRefSq = 0.0;
            scalar maxSrcPreDiff = 0.0;
            forAll(pSourcePreReference, celli)
            {
                const scalar diff =
                    pSourcePreReference[celli] - cpuSourcePreReference[celli];
                const scalar w = cellVolumes[celli];
                srcPreDiffSq += w*diff*diff;
                srcPreRefSq += w*cpuSourcePreReference[celli]
                    *cpuSourcePreReference[celli];
                maxSrcPreDiff = max(maxSrcPreDiff, mag(diff));
            }
            srcPreRefSq = max(srcPreRefSq, VSMALL);

            Info<< "GPU pressure source preSolve relative mismatch: "
                << Foam::sqrt(srcPreDiffSq/srcPreRefSq)
                << " (absMaxDiff=" << maxSrcPreDiff << ')' << nl;
        }

        if (logGpuFieldOps)
        {
            const scalarField& gpuDiag = pEqn.diag();
            const scalarField& gpuUpper = pEqn.upper();
            const scalarField& gpuSource = pEqn.source();
            const scalarField& cpuDiag = cpuEqn.diag();
            const scalarField& cpuUpper = cpuEqn.upper();
            const scalarField& cpuSource = cpuEqn.source();
            const scalarField& cellVolumes = p.mesh().V();

            scalar diagDiffSq = 0.0;
            scalar diagRefSq = 0.0;
            forAll(gpuDiag, celli)
            {
                const scalar diff = gpuDiag[celli] - cpuDiag[celli];
                const scalar w = cellVolumes[celli];
                diagDiffSq += w*diff*diff;
                diagRefSq += w*cpuDiag[celli]*cpuDiag[celli];
            }
            diagRefSq = max(diagRefSq, VSMALL);

            scalar upperDiffSq = 0.0;
            scalar upperRefSq = 0.0;
            forAll(gpuUpper, facei)
            {
                const scalar diff = gpuUpper[facei] - cpuUpper[facei];
                upperDiffSq += diff*diff;
                upperRefSq += cpuUpper[facei]*cpuUpper[facei];
            }
            upperRefSq = max(upperRefSq, VSMALL);

            scalar srcDiffSq = 0.0;
            scalar srcRefSq = 0.0;
            scalar maxSrcDiff = 0.0;
            label maxSrcCell = -1;
            forAll(gpuSource, celli)
            {
                const scalar diff = gpuSource[celli] - cpuSource[celli];
                const scalar w = cellVolumes[celli];
                srcDiffSq += w*diff*diff;
                srcRefSq += w*cpuSource[celli]*cpuSource[celli];
                if (mag(diff) > maxSrcDiff)
                {
                    maxSrcDiff = mag(diff);
                    maxSrcCell = celli;
                }
            }
            srcRefSq = max(srcRefSq, VSMALL);

            Info<< "GPU pressure matrix mismatches: diag="
                << Foam::sqrt(diagDiffSq/diagRefSq)
                << ", upper=" << Foam::sqrt(upperDiffSq/upperRefSq)
                << ", source=" << Foam::sqrt(srcDiffSq/srcRefSq)
                << " (maxDiff=" << maxSrcDiff << " @cell " << maxSrcCell;
            if (maxSrcCell >= 0)
            {
                Info<< ", srcGpu=" << gpuSource[maxSrcCell]
                    << ", srcCpu=" << cpuSource[maxSrcCell]
                    << ", divPhiHbyA=" << divPhiHbyA.internalField()[maxSrcCell]
                    << ", rAtU=" << rAtUField.internalField()[maxSrcCell]
                    << ", V=" << p.mesh().V()[maxSrcCell];
            }
            Info<< ')' << nl;
        }

        dictionary cpuSolverControls(baseSolverControls);
        cpuSolverControls.set("cacheAgglomeration", false);
        cpuEqn.solve(cpuSolverControls);

        if (logGpuFieldOps)
        {
            const scalarField& pGpuField = p.internalField();
            const scalarField& pCpuField = pCpuCheck.internalField();

            scalar diffSq = 0.0;
            scalar refSq = 0.0;
            forAll(pGpuField, celli)
            {
                const scalar diff = pGpuField[celli] - pCpuField[celli];
                const scalar w = cellVolumes[celli];
                diffSq += w*diff*diff;
                refSq += w*pCpuField[celli]*pCpuField[celli];
            }
            refSq = max(refSq, VSMALL);

            Info<< "GPU pressure solve relative mismatch: "
                << Foam::sqrt(diffSq/refSq) << nl;

            const scalarField& gpuSrc = pEqn.source();
            const scalarField& cpuSrc = cpuEqn.source();
            scalar srcDiffSq = 0.0;
            scalar srcRefSq = 0.0;
            scalar maxSrcDiff = 0.0;
            forAll(gpuSrc, celli)
            {
                const scalar diff = gpuSrc[celli] - cpuSrc[celli];
                const scalar w = cellVolumes[celli];
                srcDiffSq += w*diff*diff;
                srcRefSq += w*cpuSrc[celli]*cpuSrc[celli];
                maxSrcDiff = max(maxSrcDiff, mag(diff));
            }
            srcRefSq = max(srcRefSq, VSMALL);
            Info<< "GPU pressure source postSolve relative mismatch: "
                << Foam::sqrt(srcDiffSq/srcRefSq)
                << " (absMaxDiff=" << maxSrcDiff << ')' << nl;
        }

        if (forceCpuPressureCorrector)
        {
            p = pCpuCheck;
            gpuRegistry.getOrCreate(p).markHostDirty();

            forcedPhiTmp = phiHbyA - cpuEqn.flux();
            forcePressureOverwrite = true;

            if (logGpuFieldOps)
            {
                const scalar cpuMin = gMin(pCpuCheck.internalField());
                const scalar cpuMax = gMax(pCpuCheck.internalField());
                const scalar gpuMin = gMin(p.internalField());
                const scalar gpuMax = gMax(p.internalField());
                Info<< "GPU pressure solve forced to CPU shadow result"
                    << " (cpu min=" << cpuMin
                    << ", cpu max=" << cpuMax
                    << ", p min=" << gpuMin
                    << ", p max=" << gpuMax << ')'
                    << nl;
            }
        }

        if (logGpuFieldOps)
        {
            const scalarField& pShadowField = pCpuCheck.internalField();
            Info<< "CPU shadow pressure stats: min="
                << gMin(pShadowField)
                << ", max=" << gMax(pShadowField)
                << ", mean=" << gAverage(pShadowField)
                << ", magMax=" << gMax(mag(pShadowField)) << nl;
        }
    }

    tDivPhiHbyA.clear();

    if (pimple.finalNonOrthogonalIter())
    {
        surfaceScalarField phiFlux(pEqn.flux());

        bool gpuUpdatedPhi = false;
        Foam::gpu::DeviceField<surfaceScalarField>& phiDev =
            gpuRegistry.getOrCreate(phi);

        if (forcePressureOverwrite && forceCpuPressureCorrector)
        {
            phi = forcedPhiTmp;
            phiDev.markHostDirty();
            gpuUpdatedPhi = true;

            if (logGpuFieldOps)
            {
                Info<< "GPU phiCorrector forced to CPU shadow result" << nl;
            }
        }
        else if (useGpuFieldOps && gpuContext.ready())
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
                surfaceScalarField::Boundary& phiBf = phi.boundaryFieldRef();
                const surfaceScalarField::Boundary& phiHbyABf =
                    phiHbyA.boundaryField();
                const surfaceScalarField::Boundary& phiFluxBf =
                    phiFlux.boundaryField();

                forAll(phiBf, patchi)
                {
                    phiBf[patchi] ==
                        phiHbyABf[patchi] - phiFluxBf[patchi];
                }

                phiDev.markHostDirty();

                if (logGpuFieldOps)
                {
                    const scalarField& phiGpuInternal = phi.internalField();
                    const scalarField& phiHbyAInternal =
                        phiHbyA.internalField();
                    const scalarField& phiFluxInternal =
                        phiFlux.internalField();
                    const scalarField& faceAreas = phi.mesh().magSf();

                    scalar diffSq = 0.0;
                    scalar refSq = 0.0;
                    forAll(phiGpuInternal, facei)
                    {
                        const scalar cpuVal =
                            phiHbyAInternal[facei] - phiFluxInternal[facei];
                        const scalar diff = phiGpuInternal[facei] - cpuVal;
                        const scalar w = faceAreas[facei];
                        diffSq += w*diff*diff;
                        refSq += w*cpuVal*cpuVal;
                    }
                    refSq = max(refSq, VSMALL);

                    Info<< "GPU phiCorrector relative mismatches: internal="
                        << Foam::sqrt(diffSq/refSq) << nl;

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
            volVectorField cpuVelocityCheck
            (
                IOobject
                (
                    "U_cpuCheck",
                    runTime.timeName(),
                    U.mesh(),
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                HbyA - rAtUField*gradP
            );
            const scalarField& volumes = U.mesh().V();
            const vectorField& uGpu = U.internalField();
            const vectorField& uCpu = cpuVelocityCheck.internalField();
            scalar diffSq = 0.0;
            scalar refSq = 0.0;
            forAll(uGpu, celli)
            {
                const vector delta = uGpu[celli] - uCpu[celli];
                diffSq += volumes[celli]*(delta & delta);
                refSq += volumes[celli]*(uCpu[celli] & uCpu[celli]);
            }
            refSq = max(refSq, VSMALL);
            const scalar relErr = Foam::sqrt(diffSq/refSq);
            Info<< "GPU velocityCorrector relative L2 mismatch: "
                << relErr << nl;

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
