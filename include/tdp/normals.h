#pragma once

#include <Eigen/Dense>
#include <tdp/image.h>

namespace tdp {

void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<Eigen::Vector3f>& n,
    float f, float uc, float vc);

}
