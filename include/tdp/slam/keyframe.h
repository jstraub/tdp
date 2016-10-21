#pragma once
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>
namespace tdp {

struct KeyFrame {
  KeyFrame() {};
  KeyFrame(
      const Image<Vector3fda>& pc, 
      const Image<Vector3fda>& n,
      const Image<Vector3bda>& rgb,
      const SE3f& T_wk) :
    pc_(pc.w_, pc.h_), n_(n.w_, n.h_), rgb_(rgb.w_, rgb.h_), 
    T_wk_(T_wk)
  {
    pc_.CopyFrom(pc, cudaMemcpyHostToHost);
    n_.CopyFrom(n, cudaMemcpyHostToHost);
    rgb_.CopyFrom(rgb, cudaMemcpyHostToHost);
  }

  ManagedHostImage<Vector3fda> pc_;
  ManagedHostImage<Vector3fda> n_;
  ManagedHostImage<Vector3bda> rgb_;
  ManagedHostImage<float> d_;

  ManagedHostPyramid<Vector3fda,3> pyrPc_;
  ManagedHostPyramid<Vector3fda,3> pyrN_;
  ManagedHostPyramid<float,3> pyrGrey_;

  SE3f T_wk_; // Transformation from keyframe to world
};

/// Compute overlap fraction between two KFs in a given pyramid level
/// lvl
void Overlap(const KeyFrame& kfA, const KeyFrame& kfB,
    const CameraBase<D,Derived>& cam, int lvl, float& overlap, float& rmse, 
    const SE3f* T_ab = nullptr) {
  tdp::SE3f T_ab_ = kfA.T_wk_.Inverse() * kfB.T_wk_;
  if (T_ab)
    T_ab_ = *T_ab;

  Image<float> greyA = kfA.pyrGrey_.GetImage(lvl);
  Image<float> greyB = kfB.pyrGrey_.GetImage(lvl);
  Image<Vector3fda> pcB = kfB.pyrPc_.GetImage(lvl);
  Overlap(greyA, grayB, pcB, T_ab_, CameraBase<D,Derived>::Scale(cam,lvl), 
      overlap, rmse);
}

void Overlap(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcB, const SE3f& T_ab,
    const CameraBase<D,Derived>& camA, float& overlap, float& rmse) {

  float N = 0.f;
  overlap = 0.f;
  rmse = 0.f;
  for (size_t i=0; i<pcB.Area(); ++i) {
    if (IsValidData(pcB[i])) {
      Vector2f x = camA.Project(T_ab*pcB[i]);
      if (grayA.Inside(x)) {
        ++overlap;
        float y = grayA.GetBilinear(x)-grayB[i];
        rmse += y*y;
      }
      ++N;
    }
  }
  overlap /= N;
  rmse = sqrt( rmse/N );
}

}
