// Solve the Momentum equation (GPU-aware path)

MRF.correctBoundaryVelocity(U);

bool usedGpuDdt = false;
Foam::gpu::OperationStats ddtStats;
word gpuDdtError;
tmp<fvVectorMatrix> tGpuDdt;

if (useGpuFieldOps && gpuContext.ready())
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

if (useGpuFieldOps && gpuContext.ready())
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

UEqn.relax();

fvConstraints.constrain(UEqn);

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvConstraints.constrain(U);
}
