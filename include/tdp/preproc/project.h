#pragma once
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void ProjectPc(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& dirs,
    const Image<uint16_t>& z,
    uint16_t K,
    Image<float>& proj
    );

void ProjectPc(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& dirs,
    const Image<uint16_t>& z,
    uint16_t K,
    Image<Vector3fda>& proj
    );

void ProjectPc(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& dirs,
    Image<float>& proj
    );


}
