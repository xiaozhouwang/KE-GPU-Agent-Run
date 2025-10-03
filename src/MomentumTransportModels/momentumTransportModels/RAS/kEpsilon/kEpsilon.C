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

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

#ifdef FOAM_USE_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <mutex>
#include <vector>
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

#ifdef FOAM_USE_CUDA
namespace
{

inline std::string cudaErrorMessage(const char* prefix, const CUresult code)
{
    const char* msg = nullptr;
    cuGetErrorString(code, &msg);

    std::ostringstream os;
    os << prefix;
    if (msg)
    {
        os << msg;
    }
    else
    {
        os << "unknown CUDA error";
    }
    os << " (" << static_cast<int>(code) << ')';
    return os.str();
}


inline std::string nvrtcErrorMessage
(
    const char* prefix,
    const nvrtcResult code
)
{
    std::ostringstream os;
    os << prefix << nvrtcGetErrorString(code) << " (" << static_cast<int>(code)
       << ')';
    return os.str();
}


class kEpsilonCudaBackend
{
    std::mutex mutex_;
    bool contextInitialised_;
    bool kernelReady_;
    int deviceId_;
    CUcontext context_;
    CUmodule module_;
    CUfunction correctNutKernel_;
    bool usingPrimaryContext_;

    void cleanupLocked()
    {
        if (module_)
        {
            cuModuleUnload(module_);
            module_ = nullptr;
        }

        if (context_)
        {
            if (usingPrimaryContext_)
            {
                cuDevicePrimaryCtxRelease(deviceId_);
            }
            else
            {
                cuCtxDestroy(context_);
            }
            context_ = nullptr;
        }

        contextInitialised_ = false;
        kernelReady_ = false;
        deviceId_ = -1;
        usingPrimaryContext_ = false;
    }

    bool compileKernel(const int deviceId, std::string& errorMessage)
    {
        CUdevice device;
        CUresult status = cuDeviceGet(&device, deviceId);

        if (status != CUDA_SUCCESS)
        {
            errorMessage = cudaErrorMessage("cuDeviceGet: ", status);
            return false;
        }

        int major = 0;
        int minor = 0;
        status = cuDeviceComputeCapability(&major, &minor, device);

        if (status != CUDA_SUCCESS)
        {
            errorMessage = cudaErrorMessage("cuDeviceComputeCapability: ", status);
            return false;
        }

        std::ostringstream arch;
        arch << "--gpu-architecture=compute_" << major << minor;

        const std::string scalarName =
            sizeof(scalar) == sizeof(float) ? "float" : "double";

        std::ostringstream source;
        source
            << "extern \"C\" __global__ void correctNutKernel("
            << scalarName << "* __restrict__ nut,"
            << " const " << scalarName << "* __restrict__ k,"
            << " const " << scalarName << "* __restrict__ epsilon,"
            << ' ' << scalarName << " Cmu,"
            << ' ' << scalarName << " kMin,"
            << ' ' << scalarName << " epsilonMin,"
            << " int n)\n"
            << "{\n"
            << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            << "    if (idx < n)\n    {\n"
            << "        " << scalarName << " kval = k[idx];\n"
            << "        if (kval < kMin) kval = kMin;\n"
            << "        " << scalarName << " eps = epsilon[idx];\n"
            << "        if (eps < epsilonMin) eps = epsilonMin;\n"
            << "        " << scalarName << " value = Cmu * kval * kval / eps;\n"
            << "        nut[idx] = value;\n"
            << "    }\n"
            << "}\n";

        nvrtcProgram prog;
        nvrtcResult nvStatus = nvrtcCreateProgram
        (
            &prog,
            source.str().c_str(),
            "kEpsilonCorrectNut.cu",
            0,
            nullptr,
            nullptr
        );

        if (nvStatus != NVRTC_SUCCESS)
        {
            errorMessage = nvrtcErrorMessage("nvrtcCreateProgram: ", nvStatus);
            return false;
        }

        std::vector<std::string> optionStorage;
        optionStorage.emplace_back("--std=c++14");
        optionStorage.emplace_back("--fmad=false");
        optionStorage.emplace_back(arch.str());

        std::vector<const char*> options;
        options.reserve(optionStorage.size());
        for (const std::string& opt : optionStorage)
        {
            options.push_back(opt.c_str());
        }

        nvStatus = nvrtcCompileProgram(prog, options.size(), options.data());

        if (nvStatus != NVRTC_SUCCESS)
        {
            size_t logSize = 0;
            nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            if (logSize > 1)
            {
                char* logPtr = log.empty() ? nullptr : &log[0];
                if (logPtr)
                {
                    nvrtcGetProgramLog(prog, logPtr);
                }
            }
            nvrtcDestroyProgram(&prog);

            std::ostringstream os;
            os << "nvrtcCompileProgram: " << nvrtcGetErrorString(nvStatus);
            if (!log.empty())
            {
                os << '\n' << log;
            }
            errorMessage = os.str();
            return false;
        }

        size_t ptxSize = 0;
        nvrtcGetPTXSize(prog, &ptxSize);
        std::string ptx(ptxSize, '\0');
        if (ptxSize)
        {
            nvrtcGetPTX(prog, &ptx[0]);
        }
        nvrtcDestroyProgram(&prog);

        status = cuModuleLoadDataEx(&module_, ptx.c_str(), 0, nullptr, nullptr);
        if (status != CUDA_SUCCESS)
        {
            errorMessage = cudaErrorMessage("cuModuleLoadDataEx: ", status);
            return false;
        }

        status = cuModuleGetFunction(&correctNutKernel_, module_, "correctNutKernel");
        if (status != CUDA_SUCCESS)
        {
            errorMessage = cudaErrorMessage("cuModuleGetFunction: ", status);
            return false;
        }

        kernelReady_ = true;
        return true;
    }

public:

    kEpsilonCudaBackend()
    :
        mutex_(),
        contextInitialised_(false),
        kernelReady_(false),
        deviceId_(-1),
        context_(nullptr),
        module_(nullptr),
        correctNutKernel_(nullptr),
        usingPrimaryContext_(false)
    {}

    ~kEpsilonCudaBackend()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        cleanupLocked();
    }

    bool prepare(const int requestedDeviceId, std::string& message)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        message.clear();

        if (kernelReady_ && contextInitialised_ && requestedDeviceId == deviceId_)
        {
            cuCtxSetCurrent(context_);
            return true;
        }

        cleanupLocked();

        CUresult status = cuInit(0);
        if (status != CUDA_SUCCESS)
        {
            message = cudaErrorMessage("cuInit: ", status);
            return false;
        }

        int deviceCount = 0;
        status = cuDeviceGetCount(&deviceCount);
        if (status != CUDA_SUCCESS)
        {
            message = cudaErrorMessage("cuDeviceGetCount: ", status);
            return false;
        }

        if (deviceCount <= 0)
        {
            message = "No CUDA devices detected";
            return false;
        }

        int selectedDevice = requestedDeviceId;
        if (selectedDevice < 0 || selectedDevice >= deviceCount)
        {
            selectedDevice = ((selectedDevice % deviceCount) + deviceCount) % deviceCount;
        }

        CUdevice device;
        status = cuDeviceGet(&device, selectedDevice);
        if (status != CUDA_SUCCESS)
        {
            message = cudaErrorMessage("cuDeviceGet: ", status);
            return false;
        }

        status = cuDevicePrimaryCtxRetain(&context_, device);
        if (status != CUDA_SUCCESS)
        {
            message = cudaErrorMessage("cuDevicePrimaryCtxRetain: ", status);
            return false;
        }

        usingPrimaryContext_ = true;

        contextInitialised_ = true;
        deviceId_ = selectedDevice;

        cuCtxSetCurrent(context_);

        if (!compileKernel(selectedDevice, message))
        {
            cleanupLocked();
            return false;
        }

        return kernelReady_;
    }

    bool launchCorrectNut
    (
        scalar* nut,
        const scalar* k,
        const scalar* epsilon,
        const label nCells,
        const scalar Cmu,
        const scalar kMin,
        const scalar epsilonMin
    )
    {
        std::lock_guard<std::mutex> guard(mutex_);

        if (!kernelReady_ || !contextInitialised_)
        {
            return false;
        }

        cuCtxSetCurrent(context_);

        const size_t bytes = nCells*sizeof(scalar);

        CUdeviceptr dNut = 0;
        CUdeviceptr dK = 0;
        CUdeviceptr dEps = 0;

        CUresult status = cuMemAlloc(&dNut, bytes);
        if (status != CUDA_SUCCESS)
        {
            return false;
        }

        status = cuMemAlloc(&dK, bytes);
        if (status != CUDA_SUCCESS)
        {
            cuMemFree(dNut);
            return false;
        }

        status = cuMemAlloc(&dEps, bytes);
        if (status != CUDA_SUCCESS)
        {
            cuMemFree(dNut);
            cuMemFree(dK);
            return false;
        }

        bool ok = true;

        status = cuMemcpyHtoD(dK, k, bytes);
        ok = ok && status == CUDA_SUCCESS;
        status = cuMemcpyHtoD(dEps, epsilon, bytes);
        ok = ok && status == CUDA_SUCCESS;

        if (!ok)
        {
            cuMemFree(dNut);
            cuMemFree(dK);
            cuMemFree(dEps);
            return false;
        }

        const int threadsPerBlock = 256;
        const int blocks = static_cast<int>
        (
            (nCells + threadsPerBlock - 1)/threadsPerBlock
        );

        int nCellsInt = static_cast<int>(nCells);
        scalar kernelCmu = Cmu;
        scalar kernelKmin = kMin;
        scalar kernelEpsilonMin = epsilonMin;

        void* args[] =
        {
            &dNut,
            &dK,
            &dEps,
            &kernelCmu,
            &kernelKmin,
            &kernelEpsilonMin,
            &nCellsInt
        };

        status = cuLaunchKernel
        (
            correctNutKernel_,
            blocks, 1, 1,
            threadsPerBlock, 1, 1,
            0,
            nullptr,
            args,
            nullptr
        );

        ok = ok && status == CUDA_SUCCESS;

        if (ok)
        {
            status = cuCtxSynchronize();
            ok = ok && status == CUDA_SUCCESS;
        }

        if (ok)
        {
            status = cuMemcpyDtoH(nut, dNut, bytes);
            ok = ok && status == CUDA_SUCCESS;
        }

        cuMemFree(dNut);
        cuMemFree(dK);
        cuMemFree(dEps);

        return ok;
    }
};


inline kEpsilonCudaBackend& getKepsilonCudaBackend()
{
    static kEpsilonCudaBackend backend;
    return backend;
}

} // End anonymous namespace

#endif

template<class BasicMomentumTransportModel>
void kEpsilon<BasicMomentumTransportModel>::refreshGpuConfig()
{
    bool requestedGpu = false;

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

    gpuDeviceId_ = this->coeffDict_.template lookupOrDefault<label>("gpuDevice", gpuDeviceId_);
    gpuDeviceId_ = this->coeffDict_.template lookupOrDefault<label>("deviceId", gpuDeviceId_);

    gpuOperational_ = useGpu_;

#ifndef FOAM_USE_CUDA
    if (useGpu_)
    {
        WarningInFunction
            << "kEpsilon GPU backend requested but this build lacks CUDA support" << nl
            << "Continuing with CPU implementation." << endl;
        useGpu_ = false;
        gpuOperational_ = false;
    }
#endif
}

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
void kEpsilon<BasicMomentumTransportModel>::correctNut()
{
    bool usedGpu = false;

#ifdef FOAM_USE_CUDA
    if (useGpu_ && gpuOperational_)
    {
        std::string setupMessage;
        auto& backend = getKepsilonCudaBackend();

        if (!backend.prepare(gpuDeviceId_, setupMessage))
        {
            gpuOperational_ = false;
            if (!setupMessage.empty())
            {
                WarningInFunction
                    << "kEpsilon GPU backend initialisation failed: "
                    << setupMessage << nl << "Falling back to CPU path." << endl;
            }
        }
        else
        {
            scalarField& nutInternal = this->nut_.primitiveFieldRef();
            const scalarField& kInternal = k_.primitiveField();
            const scalarField& epsilonInternal = epsilon_.primitiveField();

            if
            (
                nutInternal.size() == kInternal.size()
             && nutInternal.size() == epsilonInternal.size()
             && nutInternal.size() == this->mesh_.nCells()
            )
            {
                usedGpu = backend.launchCorrectNut
                (
                    nutInternal.begin(),
                    kInternal.begin(),
                    epsilonInternal.begin(),
                    nutInternal.size(),
                    Cmu_.value(),
                    this->kMin_.value(),
                    this->epsilonMin_.value()
                );
            }

            if (!usedGpu)
            {
                gpuOperational_ = false;
                WarningInFunction
                    << "kEpsilon GPU kernel launch failed; reverting to CPU path"
                    << endl;
            }
        }
    }
#endif

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
    useGpu_(false),
    gpuOperational_(false),
    gpuDeviceId_(0)
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
