#pragma once
#include <tdp/image.h>
#include <tdp/camera.h>
#include <tdp/pyramid.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void Depth2PC(
    const Image<float>& d,
    const Camera<float>& cam,
    Image<Vector3fda>& pc
    );

void PyramidDepth2PCs(
    const Pyramid<float>& dPyr,
    const Camera<float>& cam,
    Pyramid<Vector3fda>& pc
    );

}
