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
#include "FvOperators.H"
#include "KernelCache.H"
#include "DynamicList.H"
#include "OFstream.H"
#include "OSspecific.H"
#include <map>
#include <fstream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

struct GpuOpRecord
{
    Foam::word timeName;
    Foam::word operation;
    Foam::scalar milliseconds;
};

struct GpuOpAggregate
{
    Foam::label samples{0};
    Foam::scalar totalMilliseconds{0};
};

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

    const Switch useGpuDdt
    (
        pimple.dict().lookupOrDefault<Switch>("useGpuDdt", true)
    );

    const Switch useGpuDiv
    (
        pimple.dict().lookupOrDefault<Switch>("useGpuDiv", true)
    );

    const Switch logGpuFieldOps
    (
        pimple.dict().lookupOrDefault<Switch>("logGpuFieldOps", false)
    );

    const Switch forceCpuPressureCorrector
    (
        pimple.dict().lookupOrDefault<Switch>
        (
            "forceCpuPressureCorrector",
            false
        )
    );

    const Switch useGpuCudaGraphs
    (
        pimple.dict().lookupOrDefault<Switch>("useGpuCudaGraphs", false)
    );

    const bool graphsEnvOptIn = Foam::gpu::graphsEnabled();
    const bool solverGraphsOptIn = static_cast<bool>(useGpuCudaGraphs);
    const bool graphsSupported = Foam::gpu::graphsSupported();
    const bool graphsRequested = graphsEnvOptIn || solverGraphsOptIn;

    if (graphsRequested && !graphsSupported)
    {
        WarningInFunction
            << "PIMPLE.useGpuCudaGraphs requested but CUDA Graph support is"
            << " disabled in this build" << nl;
    }

    Foam::gpu::setCudaGraphsRequested(graphsSupported && graphsRequested);
    const bool graphsActive = Foam::gpu::graphsEnabled();

    Info<< "GPU field ops: use=" << (useGpuFieldOps ? "on" : "off")
        << ", log=" << (logGpuFieldOps ? "on" : "off")
        << ", graphs=" << (graphsActive ? "on" : "off")
        << ", useGpuDdt=" << (useGpuDdt ? "on" : "off")
        << ", useGpuDiv=" << (useGpuDiv ? "on" : "off")
        << ", forceCpuPressure=" << (forceCpuPressureCorrector ? "on" : "off")
        << nl;

    DynamicList<GpuOpRecord> gpuOpStats;

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    Info<< "\nStarting time loop\n" << endl;

    while (pimple.run(runTime))
    {
        gpuOpStats.clear();
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

            #include "UEqnGPU.H"

            while (pimple.correct())
            {
                #include "pEqnGPU.H"
            }

            if (pimple.turbCorr())
            {
                viscosity->correct();
                turbulence->correct();

                if (logGpuFieldOps)
                {
                    tmp<volScalarField> tK = turbulence->k();
                    tmp<volScalarField> tEps = turbulence->epsilon();
                    tmp<volScalarField> tNut = turbulence->nut();

                    const volScalarField& kField = tK();
                    const volScalarField& epsField = tEps();
                    const volScalarField& nutField = tNut();

                    Info<< "turbulence k stats: min=" << gMin(kField.internalField())
                        << ", max=" << gMax(kField.internalField())
                        << ", mean=" << gAverage(kField.internalField()) << nl;
                    Info<< "turbulence epsilon stats: min="
                        << gMin(epsField.internalField())
                        << ", max=" << gMax(epsField.internalField())
                        << ", mean=" << gAverage(epsField.internalField()) << nl;
                    Info<< "turbulence nut stats: min=" << gMin(nutField.internalField())
                        << ", max=" << gMax(nutField.internalField())
                        << ", mean=" << gAverage(nutField.internalField()) << nl;

                    tK.clear();
                    tEps.clear();
                    tNut.clear();

                    const vectorField& uInternal = U.internalField();
                    const scalarField& cellVolumes = U.mesh().V();
                    scalar volWeightedMag = 0.0;
                    scalar totalVolume = 0.0;
                    scalar maxMagU = 0.0;
                    forAll(uInternal, celli)
                    {
                        const scalar magU = mag(uInternal[celli]);
                        const scalar w = cellVolumes[celli];
                        volWeightedMag += w*magU;
                        totalVolume += w;
                        maxMagU = max(maxMagU, magU);
                    }
                    totalVolume = max(totalVolume, VSMALL);
                    Info<< "velocity |U| stats: max=" << maxMagU
                        << ", volMean=" << volWeightedMag/totalVolume << nl;
                }
            }
        }

        if (logGpuFieldOps && gpuOpStats.size())
        {
            std::map<word, GpuOpAggregate> aggregates;
            for (const GpuOpRecord& rec : gpuOpStats)
            {
                GpuOpAggregate& agg = aggregates[rec.operation];
                agg.samples++;
                agg.totalMilliseconds += rec.milliseconds;
            }

            label cpuFallbackSamples = 0;
            for (const auto& kv : aggregates)
            {
                const word& op = kv.first;
                if (op.size() >= 4 && op.substr(op.size() - 4) == "_cpu")
                {
                    cpuFallbackSamples += kv.second.samples;
                }
            }

            const fileName baseDir = runTime.path()/"postProcessing"/"gpuFieldOps";
            mkDir(baseDir);
            const fileName timeDir = baseDir/runTime.timeName();
            mkDir(timeDir);

            {
                OFstream raw(timeDir/"stats_raw.csv");
                raw<< "operation,milliseconds" << '\n';
                for (const GpuOpRecord& rec : gpuOpStats)
                {
                    raw<< rec.operation << ',' << rec.milliseconds << '\n';
                }
            }

            {
                OFstream detail(timeDir/"stats.csv");
                detail<< "operation,samples,totalMilliseconds,averageMilliseconds" << '\n';
                for (const auto& kv : aggregates)
                {
                    const GpuOpAggregate& agg = kv.second;
                    const scalar average = agg.samples
                        ? agg.totalMilliseconds/agg.samples
                        : 0.0;
                    detail<< kv.first << ',' << agg.samples << ','
                          << agg.totalMilliseconds << ',' << average << '\n';
                }
            }

            const fileName summaryPath = baseDir/"summary.csv";
            const bool summaryExists = isFile(summaryPath);
            std::ofstream summary(summaryPath.c_str(), std::ios::out | std::ios::app);
            if (!summary)
            {
                WarningInFunction
                    << "Unable to open " << summaryPath << " for append" << nl;
            }
            else
            {
                if (!summaryExists)
                {
                    summary<< "time,operation,samples,totalMilliseconds,averageMilliseconds" << '\n';
                }

                for (const auto& kv : aggregates)
                {
                    const GpuOpAggregate& agg = kv.second;
                    const scalar average = agg.samples
                        ? agg.totalMilliseconds/agg.samples
                        : 0.0;
                    summary<< runTime.timeName() << ',' << kv.first << ','
                           << agg.samples << ',' << agg.totalMilliseconds << ','
                           << average << '\n';
                }
            }

            if (cpuFallbackSamples)
            {
                Info<< "GPU fallbacks this step: " << cpuFallbackSamples
                    << " (see " << timeDir << ")" << nl;
            }

            gpuOpStats.clear();
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
