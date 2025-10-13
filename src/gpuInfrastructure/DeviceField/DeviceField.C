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
            ctx.stream()
        );

    if (status != cudaSuccess)
    {
        error = "cudaMemcpyAsync(host->device)";
        return false;
    }

    status = ::cudaStreamSynchronize(ctx.stream());
    if (status != cudaSuccess)
    {
        error = "cudaStreamSynchronize(host->device)";
        return false;
    }

    hostDirty_ = false;
    deviceDirty_ = false;
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
    cudaError_t status =
        ::cudaMemcpyAsync
        (
            hostPtr,
            devicePtr_,
            sizeof(ValueType)*sz,
            cudaMemcpyDeviceToHost,
            ctx.stream()
        );

    if (status != cudaSuccess)
    {
        error = "cudaMemcpyAsync(device->host)";
        return false;
    }

    status = ::cudaStreamSynchronize(ctx.stream());
    if (status != cudaSuccess)
    {
        error = "cudaStreamSynchronize(device->host)";
        return false;
    }

    deviceDirty_ = false;
    hostDirty_ = false;
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
}


template<class GeoField>
void DeviceField<GeoField>::markDeviceDirty()
{
    deviceDirty_ = true;
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
