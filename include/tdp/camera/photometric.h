#pragma once 
#include <tdp/data/image.h>
#include <tdp/camera/camera_base.h>

namespace tdp {

template <int D, class Derived>
void OverlapGpu(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA,
    const Image<Vector3fda>& pcB,
    const SE3f& T_ab, 
    const CameraBase<float,D,Derived>& camA, float& overlap, float& rmse, 
    Image<float>* errB = nullptr);

}
