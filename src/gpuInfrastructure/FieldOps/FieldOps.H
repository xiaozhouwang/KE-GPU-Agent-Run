#ifndef Foam_gpu_FieldOps_H
#define Foam_gpu_FieldOps_H

#include "DeviceField.H"
#include "surfaceFields.H"
#include "volFields.H"

namespace Foam
{
namespace gpu
{

struct OperationStats
{
    scalar gpuMilliseconds{0};
};

//- Performs dst = a - b on the device (surface scalar fields).
//  Returns true on success; on failure leaves error message and dst untouched.
bool subtractSurfaceScalarFields
(
    Context& ctx,
    DeviceField<surfaceScalarField>& dst,
    const DeviceField<surfaceScalarField>& a,
    const DeviceField<surfaceScalarField>& b,
    word& error,
    OperationStats* stats = nullptr
);

//- Updates velocity via U = HbyA - rAtU * gradP using device data.
bool velocityCorrector
(
    Context& ctx,
    DeviceField<volVectorField>& U,
    const DeviceField<volVectorField>& HbyA,
    const DeviceField<volScalarField>& rAtU,
    const DeviceField<volVectorField>& gradP,
    word& error,
    OperationStats* stats = nullptr
);

//- Compute nut = Cmu * max(k,kMin)^2 / max(epsilon, epsilonMin) on device.
bool computeNutFromKEpsilon
(
    Context& ctx,
    DeviceField<volScalarField>& nut,
    const DeviceField<volScalarField>& k,
    const DeviceField<volScalarField>& epsilon,
    const scalar Cmu,
    const scalar kMin,
    const scalar epsilonMin,
    word& error,
    OperationStats* stats = nullptr
);

} // namespace gpu
} // namespace Foam

#endif
