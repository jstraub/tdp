#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>

namespace tdp {

void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<tdp::Vector3fda>& n,
    float f, float uc, float vc);

void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN);

template<int LEVELS>
void Depth2Normals(
    const Pyramid<float,LEVELS>& cuD,
    const Camera<float>& cam,
    Pyramid<Vector3fda,LEVELS> cuN);

}
