#ifndef Foam_gpu_DeviceField_H
#define Foam_gpu_DeviceField_H

#include "GpuContext.H"
#include "fvMesh.H"
#include "MeshObject.H"
#include "word.H"

#include <memory>
#include <unordered_map>

#ifdef FOAM_USE_CUDA
    #include <cuda_runtime.h>
#endif

namespace Foam
{
namespace gpu
{

class DeviceFieldBase
{
public:
    virtual ~DeviceFieldBase() = default;
    virtual bool syncHostToDevice(Context& ctx, word& error) = 0;
    virtual bool syncDeviceToHost(Context& ctx, word& error) = 0;
    virtual bool allocated() const = 0;
    virtual label size() const = 0;
};


template<class GeoField>
class DeviceField
:
    public DeviceFieldBase
{
    typedef typename GeoField::value_type ValueType;

    GeoField& field_;

#ifdef FOAM_USE_CUDA
    ValueType* devicePtr_;
    label count_;
#endif

    bool hostDirty_;
    bool deviceDirty_;

    bool allocate(Context& ctx, word& err);

public:

    explicit DeviceField(GeoField& field);
    ~DeviceField() override;

    DeviceField(const DeviceField&) = delete;
    DeviceField& operator=(const DeviceField&) = delete;

    ValueType* devicePointer();
    const ValueType* devicePointer() const;

    bool syncHostToDevice(Context& ctx, word& error) override;
    bool syncDeviceToHost(Context& ctx, word& error) override;
    bool allocated() const override;
    label size() const override;

    void markHostDirty();
    void markDeviceDirty();
};


class FieldRegistry
:
    public MeshObject<fvMesh, GeometricMeshObject, FieldRegistry>
{
    std::unordered_map<const void*, std::unique_ptr<DeviceFieldBase>> registry_;

public:

    TypeName("gpuFieldRegistry");

    explicit FieldRegistry(const fvMesh& mesh);

    static FieldRegistry& New(const fvMesh& mesh);

    template<class GeoField>
    DeviceField<GeoField>& getOrCreate(GeoField& field);
};

} // namespace gpu
} // namespace Foam

#endif
