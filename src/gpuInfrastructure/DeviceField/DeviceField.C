#include "DeviceField.H"
#include "error.H"
#include "volFields.H"
#include "surfaceFields.H"

namespace Foam
{
namespace gpu
{

defineTypeNameAndDebug(FieldRegistry, 0);

template<class GeoField>
DeviceField<GeoField>::DeviceField(GeoField& field)
:
    field_(field),
#ifdef FOAM_USE_CUDA
    devicePtr_(nullptr),
    count_(0),
    soaPtr_(nullptr),
    soaCount_(0),
    soaValid_(false),
#endif
    hostDirty_(true),
    deviceDirty_(false)
{}


template<class GeoField>
DeviceField<GeoField>::~DeviceField()
{
#ifdef FOAM_USE_CUDA
    if (devicePtr_)
    {
        ::cudaFree(devicePtr_);
        devicePtr_ = nullptr;
        count_ = 0;
    }
    freeSoA();
#endif
}


template<class GeoField>
bool DeviceField<GeoField>::allocate(Context& ctx, word& err)
{
#ifdef FOAM_USE_CUDA
    if (!ctx.ready())
    {
        if (!ctx.initialise(0, err))
        {
            return false;
        }
    }

    const label sz = field_.internalField().size();
    if (sz <= 0)
    {
        err.clear();
        return true;
    }

    if (devicePtr_ && count_ == sz)
    {
        err.clear();
        return true;
    }

    if (devicePtr_)
    {
        ::cudaFree(devicePtr_);
        devicePtr_ = nullptr;
        count_ = 0;
    }
    freeSoA();

    cudaError_t status =
        ::cudaMalloc(reinterpret_cast<void**>(&devicePtr_), sizeof(ValueType)*sz);

    if (status != cudaSuccess)
    {
        err = "cudaMalloc(DeviceField)";
        devicePtr_ = nullptr;
        count_ = 0;
        return false;
    }

    count_ = sz;
    hostDirty_ = true;
    deviceDirty_ = false;
    soaValid_ = false;
    err.clear();
    return true;
#else
    err = "CUDA support not available";
    return false;
#endif
}


template<class GeoField>
typename DeviceField<GeoField>::ValueType* DeviceField<GeoField>::devicePointer()
{
#ifdef FOAM_USE_CUDA
    return devicePtr_;
#else
    return nullptr;
#endif
}


template<class GeoField>
const typename DeviceField<GeoField>::ValueType*
DeviceField<GeoField>::devicePointer() const
{
#ifdef FOAM_USE_CUDA
    return devicePtr_;
#else
    return nullptr;
#endif
}


template<class GeoField>
bool DeviceField<GeoField>::syncHostToDevice(Context& ctx, word& error)
{
#ifdef FOAM_USE_CUDA
    if (!allocate(ctx, error))
    {
        return false;
    }

    if (!hostDirty_)
    {
        error.clear();
        return true;
    }

    const label sz = field_.internalField().size();
    if (sz == 0)
    {
        hostDirty_ = false;
        deviceDirty_ = false;
        error.clear();
        return true;
    }

    const ValueType* hostPtr = field_.primitiveField().begin();
    cudaError_t status =
        ::cudaMemcpyAsync
        (
            devicePtr_,
            hostPtr,
            sizeof(ValueType)*sz,
            cudaMemcpyHostToDevice,
            ctx.transferStream()
        );

    if (status != cudaSuccess)
    {
        error = "cudaMemcpyAsync(host->device)";
        return false;
    }

    status = ::cudaStreamSynchronize(ctx.transferStream());
    if (status != cudaSuccess)
    {
        error = "cudaStreamSynchronize(host->device)";
        return false;
    }

    hostDirty_ = false;
    deviceDirty_ = false;
    soaValid_ = false;
    error.clear();
    return true;
#else
    (void)ctx;
    error = "CUDA support not available";
    return false;
#endif
}


template<class GeoField>
bool DeviceField<GeoField>::syncDeviceToHost(Context& ctx, word& error)
{
#ifdef FOAM_USE_CUDA
    if (!allocate(ctx, error))
    {
        return false;
    }

    if (!deviceDirty_)
    {
        error.clear();
        return true;
    }

    const label sz = field_.internalField().size();
    if (sz == 0)
    {
        deviceDirty_ = false;
        hostDirty_ = false;
        error.clear();
        return true;
    }

    ValueType* hostPtr = field_.primitiveFieldRef().begin();

    cudaStream_t transferStream = ctx.transferStream();
    cudaStream_t computeStream = ctx.computeStream();
    cudaStream_t auxStream = ctx.auxStream();

    cudaEvent_t computeDone = nullptr;
    cudaEvent_t auxDone = nullptr;

    auto destroyEvents = [&]()
    {
        if (computeDone)
        {
            ::cudaEventDestroy(computeDone);
            computeDone = nullptr;
        }
        if (auxDone)
        {
            ::cudaEventDestroy(auxDone);
            auxDone = nullptr;
        }
    };

    if (computeStream)
    {
        if (::cudaEventCreateWithFlags(&computeDone, cudaEventDisableTiming) != cudaSuccess)
        {
            error = "cudaEventCreate(computeDone)";
            destroyEvents();
            return false;
        }

        if (::cudaEventRecord(computeDone, computeStream) != cudaSuccess)
        {
            error = "cudaEventRecord(computeDone)";
            destroyEvents();
            return false;
        }

        if (::cudaStreamWaitEvent(transferStream, computeDone, 0) != cudaSuccess)
        {
            error = "cudaStreamWaitEvent(computeDone)";
            destroyEvents();
            return false;
        }
    }

    if (auxStream && auxStream != computeStream)
    {
        if (::cudaEventCreateWithFlags(&auxDone, cudaEventDisableTiming) != cudaSuccess)
        {
            error = "cudaEventCreate(auxDone)";
            destroyEvents();
            return false;
        }

        if (::cudaEventRecord(auxDone, auxStream) != cudaSuccess)
        {
            error = "cudaEventRecord(auxDone)";
            destroyEvents();
            return false;
        }

        if (::cudaStreamWaitEvent(transferStream, auxDone, 0) != cudaSuccess)
        {
            error = "cudaStreamWaitEvent(auxDone)";
            destroyEvents();
            return false;
        }
    }

    cudaError_t status =
        ::cudaMemcpyAsync
        (
            hostPtr,
            devicePtr_,
            sizeof(ValueType)*sz,
            cudaMemcpyDeviceToHost,
            transferStream
        );

    if (status != cudaSuccess)
    {
        error = "cudaMemcpyAsync(device->host)";
        destroyEvents();
        return false;
    }

    status = ::cudaStreamSynchronize(transferStream);
    if (status != cudaSuccess)
    {
        error = "cudaStreamSynchronize(device->host)";
        destroyEvents();
        return false;
    }

    destroyEvents();

    deviceDirty_ = false;
    hostDirty_ = false;
    soaValid_ = false;
    error.clear();
    return true;
#else
    (void)ctx;
    error = "CUDA support not available";
    return false;
#endif
}


template<class GeoField>
bool DeviceField<GeoField>::allocated() const
{
#ifdef FOAM_USE_CUDA
    return devicePtr_ != nullptr;
#else
    return false;
#endif
}


template<class GeoField>
label DeviceField<GeoField>::size() const
{
#ifdef FOAM_USE_CUDA
    return count_;
#else
    return 0;
#endif
}


template<class GeoField>
void DeviceField<GeoField>::markHostDirty()
{
    hostDirty_ = true;
#ifdef FOAM_USE_CUDA
    soaValid_ = false;
#endif
}


template<class GeoField>
void DeviceField<GeoField>::markDeviceDirty()
{
    deviceDirty_ = true;
#ifdef FOAM_USE_CUDA
    soaValid_ = false;
#endif
}


template<class GeoField>
void DeviceField<GeoField>::freeSoA()
{
#ifdef FOAM_USE_CUDA
    if (soaPtr_)
    {
        ::cudaFree(soaPtr_);
        soaPtr_ = nullptr;
    }
    soaCount_ = 0;
    soaValid_ = false;
#endif
}


template<class GeoField>
bool DeviceField<GeoField>::supportsSoA() const
{
    return pTraits<ValueType>::nComponents > 1;
}


template<class GeoField>
bool DeviceField<GeoField>::ensureSoALayout(Context& ctx, word& error)
{
#ifdef FOAM_USE_CUDA
    if (!supportsSoA())
    {
        soaPtr_ = reinterpret_cast<typename pTraits<ValueType>::cmptType*>(devicePtr_);
        soaCount_ = count_;
        soaValid_ = true;
        error.clear();
        return true;
    }

    if (!syncHostToDevice(ctx, error))
    {
        return false;
    }

    const label sz = size();
    if (!sz)
    {
        soaValid_ = true;
        error.clear();
        return true;
    }

    const label components = pTraits<ValueType>::nComponents;
    const std::size_t componentSize = sizeof(typename pTraits<ValueType>::cmptType);
    const std::size_t dstPitch = componentSize;
    const std::size_t srcPitch = componentSize*components;

    if (!soaPtr_ || soaCount_ != sz)
    {
        if (soaPtr_)
        {
            ::cudaFree(soaPtr_);
        }

        cudaError_t status = ::cudaMalloc
        (
            reinterpret_cast<void**>(&soaPtr_),
            componentSize*components*sz
        );

        if (status != cudaSuccess)
        {
            soaPtr_ = nullptr;
            soaCount_ = 0;
            error = "cudaMalloc(DeviceField::SoA)";
            return false;
        }

        soaCount_ = sz;
        soaValid_ = false;
    }

    if (soaValid_)
    {
        error.clear();
        return true;
    }

    for (label cmpt = 0; cmpt < components; ++cmpt)
    {
        char* dst = reinterpret_cast<char*>(soaPtr_)
          + cmpt*componentSize*soaCount_;

        const char* src = reinterpret_cast<const char*>(devicePtr_)
          + cmpt*componentSize;

        cudaError_t status = ::cudaMemcpy2DAsync
        (
            dst,
            dstPitch,
            src,
            srcPitch,
            componentSize,
            static_cast<size_t>(soaCount_),
            cudaMemcpyDeviceToDevice,
            ctx.auxStream()
        );

        if (status != cudaSuccess)
        {
            error = "cudaMemcpy2DAsync(AoS->SoA)";
            return false;
        }
    }

    cudaStream_t auxStream = ctx.auxStream();
    if (auxStream)
    {
        cudaEvent_t copyDone = nullptr;
        if (::cudaEventCreateWithFlags(&copyDone, cudaEventDisableTiming) != cudaSuccess)
        {
            error = "cudaEventCreate(soaCopy)";
            return false;
        }

        if (::cudaEventRecord(copyDone, auxStream) != cudaSuccess)
        {
            ::cudaEventDestroy(copyDone);
            error = "cudaEventRecord(soaCopy)";
            return false;
        }

        cudaStream_t computeStream = ctx.computeStream();
        if (::cudaStreamWaitEvent(computeStream, copyDone, 0) != cudaSuccess)
        {
            ::cudaEventDestroy(copyDone);
            error = "cudaStreamWaitEvent(soaCopy)";
            return false;
        }

        ::cudaEventDestroy(copyDone);
    }

    soaValid_ = true;
    error.clear();
    return true;
#else
    (void)ctx;
    (void)error;
    return false;
#endif
}


template<class GeoField>
typename pTraits<typename DeviceField<GeoField>::ValueType>::cmptType*
DeviceField<GeoField>::soaPointer()
{
#ifdef FOAM_USE_CUDA
    return soaPtr_;
#else
    return nullptr;
#endif
}


template<class GeoField>
const typename pTraits<typename DeviceField<GeoField>::ValueType>::cmptType*
DeviceField<GeoField>::soaPointer() const
{
#ifdef FOAM_USE_CUDA
    return soaPtr_;
#else
    return nullptr;
#endif
}


FieldRegistry::FieldRegistry(const fvMesh& mesh)
:
    MeshObject<fvMesh, GeometricMeshObject, FieldRegistry>(mesh),
    registry_()
{}


FieldRegistry& FieldRegistry::New(const fvMesh& mesh)
{
    return const_cast<FieldRegistry&>
    (
        MeshObject<fvMesh, GeometricMeshObject, FieldRegistry>::New(mesh)
    );
}


template<class GeoField>
DeviceField<GeoField>& FieldRegistry::getOrCreate(GeoField& field)
{
    const void* key = static_cast<const void*>(&field);
    auto iter = registry_.find(key);
    if (iter != registry_.end())
    {
        return static_cast<DeviceField<GeoField>&>(*iter->second);
    }

    auto deviceField = std::make_unique<DeviceField<GeoField>>(field);
    DeviceField<GeoField>& ref = *deviceField;
    registry_.emplace(key, std::move(deviceField));
    return ref;
}


// Explicit template instantiations for commonly used field types

template class DeviceField<volScalarField>;
template class DeviceField<volVectorField>;
template class DeviceField<surfaceScalarField>;
template DeviceField<volScalarField>& FieldRegistry::getOrCreate(volScalarField&);
template DeviceField<volVectorField>& FieldRegistry::getOrCreate(volVectorField&);
template DeviceField<surfaceScalarField>& FieldRegistry::getOrCreate(surfaceScalarField&);

} // namespace gpu
} // namespace Foam
