#pragma once

#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/camera/camera_base.h>

namespace tdp {

/// This directly computes the gradient image in a custom kernel -
/// should be faster so use this.
/// also does not care about size of image
void GradientShar(const Image<float>& I,
    Image<Vector2fda>& gradI);

void Gradient(const Image<float>& I, 
    Image<float>& Iu, Image<float>& Iv);

void Gradient2Vector(const Image<float>& Iu, const Image<float>& Iv,
    Image<Vector2fda>& gradI);

void Gradient2AngleNorm(const Image<float>& Iu, const Image<float>& Iv,
    Image<float>& Itheta, Image<float>& Inorm);

void Gradient2AngleNorm(const Image<tdp::Vector2fda>& gradI, 
    Image<float>& Itheta, Image<float>& Inorm);

void Gradient(const Image<float>& I, 
    Image<float>& Iu, Image<float>& Iv,
    Image<Vector2fda>& gradI);


void Grad2Image(const Image<Vector2fda>& grad,
    Image<Vector3bda>& grad2d);

template<int D, typename Derived>
void Gradient3D(const Image<float>& Iu, const Image<float>& Iv,
    const Image<float>& cuD,
    const Image<Vector3fda>& cuN,
    const CameraBase<float,D,Derived>& cam,
    float gradNormThr,
    Image<Vector3fda>& cuGrad3D);

template<int D, typename Derived>
void Gradient3D(const Image<float>& cuI, 
    const Image<float>& cuD,
    const Image<Vector3fda>& cuN,
    const CameraBase<float,D,Derived>& cam,
    float gradNormThr,
    Image<float>& cuIu, Image<float>& cuIv,
    Image<Vector3fda>& cuGrad3D) {
  Gradient(cuI, cuIu, cuIv);
  Gradient3D<D,Derived>(cuIu, cuIv, cuD, cuN, cam, gradNormThr, cuGrad3D); 
}

}
