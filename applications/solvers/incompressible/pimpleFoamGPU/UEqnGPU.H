// Solve the Momentum equation (GPU-aware path)

MRF.correctBoundaryVelocity(U);

bool usedGpuDdt = false;
Foam::gpu::OperationStats ddtStats;
word gpuDdtError;
tmp<fvVectorMatrix> tGpuDdt;

if (useGpuFieldOps && useGpuDdt && gpuContext.ready())
{
    tGpuDdt = Foam::gpu::fvm::ddt
    (
        gpuContext,
        gpuRegistry,
        U,
        usedGpuDdt,
        gpuDdtError,
        logGpuFieldOps ? &ddtStats : nullptr
    );

    if (!usedGpuDdt && !gpuDdtError.empty())
    {
        WarningInFunction
            << "GPU fvm::ddt() fallback: " << gpuDdtError << nl << endl;
    }
}

tmp<fvVectorMatrix> tDdt;
if (usedGpuDdt)
{
    if (logGpuFieldOps)
    {
        tmp<fvVectorMatrix> tDdtCpuCheck = fvm::ddt(U);
        fvVectorMatrix& ddtGpu = tGpuDdt.ref();
        const fvVectorMatrix& ddtCpu = tDdtCpuCheck();

        const scalarField& diagGpu = ddtGpu.diag();
        const scalarField& diagCpu = ddtCpu.diag();
        const scalarField& cellVolumes = U.mesh().V();
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

        const vectorField& sourceGpu = ddtGpu.source();
        const vectorField& sourceCpu = ddtCpu.source();
        scalar sourceDiffSq = 0.0;
        scalar sourceRefSq = 0.0;
        forAll(sourceGpu, celli)
        {
            const vector diff = sourceGpu[celli] - sourceCpu[celli];
            const scalar w = cellVolumes[celli];
            sourceDiffSq += w*(diff & diff);
            sourceRefSq += w*(sourceCpu[celli] & sourceCpu[celli]);
        }
        sourceRefSq = max(sourceRefSq, VSMALL);

        Info<< "GPU fvm::ddt relative mismatches: diag="
            << Foam::sqrt(diagDiffSq/diagRefSq)
            << ", source=" << Foam::sqrt(sourceDiffSq/sourceRefSq)
            << nl;

        gpuOpStats.append
        (
            {
                runTime.timeName(),
                word("fvmDdt"),
                ddtStats.gpuMilliseconds
            }
        );
    }

    tDdt = std::move(tGpuDdt);
}
else
{
    tDdt = fvm::ddt(U);
}

bool usedGpuDiv = false;
Foam::gpu::OperationStats divStats;
word gpuDivError;
tmp<fvVectorMatrix> tGpuDiv;

if (useGpuFieldOps && useGpuDiv && gpuContext.ready())
{
    tGpuDiv = Foam::gpu::fvm::div
    (
        gpuContext,
        gpuRegistry,
        phi,
        U,
        usedGpuDiv,
        gpuDivError,
        logGpuFieldOps ? &divStats : nullptr
    );

    if (!usedGpuDiv && !gpuDivError.empty())
    {
        WarningInFunction
            << "GPU fvm::div() fallback: " << gpuDivError << nl << endl;
    }
}

tmp<fvVectorMatrix> tDiv;
if (usedGpuDiv)
{
    if (logGpuFieldOps)
    {
        tmp<fvVectorMatrix> tDivCpuCheck = fvm::div(phi, U);
        fvVectorMatrix& divGpu = tGpuDiv.ref();
        const fvVectorMatrix& divCpu = tDivCpuCheck();

        tmp<fvVectorMatrix> tDivGpuInspect = divGpu.clone();
        const fvVectorMatrix& divGpuInspect = tDivGpuInspect();

        const scalarField& diagGpu = divGpuInspect.diag();
        const scalarField& diagCpu = divCpu.diag();
        const scalarField& cellVolumes = U.mesh().V();
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

        const scalarField& lowerGpu = divGpuInspect.lower();
        const scalarField& lowerCpu = divCpu.lower();
        scalar lowerDiffSq = 0.0;
        scalar lowerRefSq = 0.0;
        forAll(lowerGpu, facei)
        {
            const scalar diff = lowerGpu[facei] - lowerCpu[facei];
            lowerDiffSq += diff*diff;
            lowerRefSq += lowerCpu[facei]*lowerCpu[facei];
        }
        lowerRefSq = max(lowerRefSq, VSMALL);

        const scalarField& upperGpu = divGpuInspect.upper();
        const scalarField& upperCpu = divCpu.upper();
        scalar upperDiffSq = 0.0;
        scalar upperRefSq = 0.0;
        forAll(upperGpu, facei)
        {
            const scalar diff = upperGpu[facei] - upperCpu[facei];
            upperDiffSq += diff*diff;
            upperRefSq += upperCpu[facei]*upperCpu[facei];
        }
        upperRefSq = max(upperRefSq, VSMALL);

        const vectorField& sourceGpu = divGpuInspect.source();
        const vectorField& sourceCpu = divCpu.source();
        scalar sourceDiffSq = 0.0;
        scalar sourceRefSq = 0.0;
        forAll(sourceGpu, celli)
        {
            const vector diff = sourceGpu[celli] - sourceCpu[celli];
            const scalar w = cellVolumes[celli];
            sourceDiffSq += w*(diff & diff);
            sourceRefSq += w*(sourceCpu[celli] & sourceCpu[celli]);
        }
        sourceRefSq = max(sourceRefSq, VSMALL);

        Info<< "GPU fvm::div relative mismatches: diag="
            << Foam::sqrt(diagDiffSq/diagRefSq)
            << ", lower=" << Foam::sqrt(lowerDiffSq/lowerRefSq)
            << ", upper=" << Foam::sqrt(upperDiffSq/upperRefSq)
            << ", source=" << Foam::sqrt(sourceDiffSq/sourceRefSq)
            << nl;

        gpuOpStats.append
        (
            {
                runTime.timeName(),
                word("fvmDiv"),
                divStats.gpuMilliseconds
            }
        );
    }

    tDiv = std::move(tGpuDiv);
}
else
{
    tDiv = fvm::div(phi, U);
}

tmp<fvVectorMatrix> tUEqn
(
    tDdt
  + tDiv
  + MRF.DDt(U)
  + turbulence->divDevSigma(U)
 == fvModels.source(U)
);

fvVectorMatrix& UEqn = tUEqn.ref();

if (logGpuFieldOps)
{
    tmp<fvVectorMatrix> tUEqnCpuCheck
    (
        fvm::ddt(U)
      + fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevSigma(U)
     == fvModels.source(U)
    );

    tmp<fvVectorMatrix> tUEqnGpuInspect = UEqn.clone();

    const fvVectorMatrix& UEqnGpuInspect = tUEqnGpuInspect();
    const fvVectorMatrix& UEqnCpu = tUEqnCpuCheck();

    const scalarField& cellVolumes = U.mesh().V();

    const scalarField& diagGpu = UEqnGpuInspect.diag();
    const scalarField& diagCpu = UEqnCpu.diag();
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

    const scalarField& lowerGpu = UEqnGpuInspect.lower();
    const scalarField& lowerCpu = UEqnCpu.lower();
    scalar lowerDiffSq = 0.0;
    scalar lowerRefSq = 0.0;
    forAll(lowerGpu, facei)
    {
        const scalar diff = lowerGpu[facei] - lowerCpu[facei];
        lowerDiffSq += diff*diff;
        lowerRefSq += lowerCpu[facei]*lowerCpu[facei];
    }
    lowerRefSq = max(lowerRefSq, VSMALL);

    const scalarField& upperGpu = UEqnGpuInspect.upper();
    const scalarField& upperCpu = UEqnCpu.upper();
    scalar upperDiffSq = 0.0;
    scalar upperRefSq = 0.0;
    forAll(upperGpu, facei)
    {
        const scalar diff = upperGpu[facei] - upperCpu[facei];
        upperDiffSq += diff*diff;
        upperRefSq += upperCpu[facei]*upperCpu[facei];
    }
    upperRefSq = max(upperRefSq, VSMALL);

    const vectorField& sourceGpu = UEqnGpuInspect.source();
    const vectorField& sourceCpu = UEqnCpu.source();
    scalar sourceDiffSq = 0.0;
    scalar sourceRefSq = 0.0;
    scalar maxSourceDiff = 0.0;
    label maxSourceDiffCell = -1;
    vector maxSourceDiffVec = Zero;
    forAll(sourceGpu, celli)
    {
        const vector diff = sourceGpu[celli] - sourceCpu[celli];
        const scalar w = cellVolumes[celli];
        sourceDiffSq += w*(diff & diff);
        sourceRefSq += w*(sourceCpu[celli] & sourceCpu[celli]);
        const scalar magDiff = Foam::sqrt(diff & diff);
        if (magDiff > maxSourceDiff)
        {
            maxSourceDiff = magDiff;
            maxSourceDiffCell = celli;
            maxSourceDiffVec = diff;
        }
    }
    sourceRefSq = max(sourceRefSq, VSMALL);

        Info<< "GPU UEqn matrix mismatches: diag="
            << Foam::sqrt(diagDiffSq/diagRefSq)
            << ", lower=" << Foam::sqrt(lowerDiffSq/lowerRefSq)
            << ", upper=" << Foam::sqrt(upperDiffSq/upperRefSq)
            << ", source=" << Foam::sqrt(sourceDiffSq/sourceRefSq)
            << " (absMaxSourceDiff=" << maxSourceDiff;
    if (maxSourceDiffCell >= 0)
    {
        const vector expectedSource =
            UEqnGpuInspect.diag()[maxSourceDiffCell]
           *U.oldTime().internalField()[maxSourceDiffCell];
        Info<< " @cell " << maxSourceDiffCell
            << ", diff=" << maxSourceDiffVec
            << ", gpu=" << sourceGpu[maxSourceDiffCell]
            << ", cpu=" << sourceCpu[maxSourceDiffCell]
            << ", U_old=" << U.oldTime().internalField()[maxSourceDiffCell]
            << ", U_new=" << U.internalField()[maxSourceDiffCell]
            << ", V=" << U.mesh().V()[maxSourceDiffCell]
            << ", diag=" << UEqnGpuInspect.diag()[maxSourceDiffCell]
            << ", expected=" << expectedSource;
    }
    Info<< ')' << nl;

    tUEqnGpuInspect.clear();
    tUEqnCpuCheck.clear();
}

UEqn.relax();

fvConstraints.constrain(UEqn);

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvConstraints.constrain(U);
}
