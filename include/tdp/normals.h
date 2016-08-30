#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/image.h>

namespace tdp {

void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<tdp::Vector3fda>& n,
    float f, float uc, float vc);

}
