
#pragma once
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void ComputeUnitPlanes(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    Image<Vector4fda>& pl
    );

}
