#pragma once
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>
#include <tdp/camera/camera_base.h>
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

  KeyFrame(
      const Pyramid<Vector3fda,3>& pc, 
      const Pyramid<Vector3fda,3>& n,
      const Pyramid<float,3>& grey,
      const Image<Vector3bda>& rgb,
      const SE3f& T_wk) :
    pc_(pc.w_, pc.h_), n_(n.w_, n.h_), rgb_(rgb.w_, rgb.h_), 
    d_(pc.w_, pc.h_), pyrPc_(pc.w_, pc.h_), pyrN_(n.w_, n.h_),
    pyrGrey_(grey.w_, grey.h_),
    T_wk_(T_wk)
  {
    pc_.CopyFrom(pc.GetConstImage(0),  cudaMemcpyDeviceToHost);
    n_.CopyFrom(n.GetConstImage(0),    cudaMemcpyDeviceToHost);
    rgb_.CopyFrom(rgb,            cudaMemcpyHostToHost);

    pyrPc_.CopyFrom(pc,     cudaMemcpyDeviceToHost);
    pyrN_.CopyFrom(n,       cudaMemcpyDeviceToHost);
    pyrGrey_.CopyFrom(grey, cudaMemcpyHostToHost);
  }

//  KeyFrame(const KeyFrame& ) = delete;
//  KeyFrame& operator=(const KeyFrame& ) = delete;

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
template <int D, class Derived>
void Overlap(const KeyFrame& kfA, const KeyFrame& kfB,
    const CameraBase<float, D,Derived>& cam, int lvl, float& overlap,
    float& rmse, const SE3f* T_ab = nullptr) {
  tdp::SE3f T_ab_ = kfA.T_wk_.Inverse() * kfB.T_wk_;
  if (T_ab)
    T_ab_ = *T_ab;

  const Image<float> greyA = kfA.pyrGrey_.GetConstImage(lvl);
  const Image<float> greyB = kfB.pyrGrey_.GetConstImage(lvl);
  const Image<Vector3fda> pcB = kfB.pyrPc_.GetConstImage(lvl);
  Overlap(greyA, greyB, pcB, T_ab_, 
      ScaleCamera<float,D,Derived>(cam,pow(0.5,lvl)), 
      overlap, rmse);
}

template <int D, class Derived>
void Overlap(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcB, const SE3f& T_ab,
    const CameraBase<float,D,Derived>& camA, float& overlap, float& rmse) {

  float N = 0.f;
  overlap = 0.f;
  rmse = 0.f;
  for (size_t i=0; i<pcB.Area(); ++i) {
    if (IsValidData(pcB[i])) {
      Eigen::Vector2f x = camA.Project(T_ab*pcB[i]);
      if (greyA.Inside(x) && i < greyB.Area()) {
        ++overlap;
        float y = greyA.GetBilinear(x)-greyB[i];
        rmse += y*y;
      }
      ++N;
    }
  }
  overlap /= N;
  rmse = sqrt( rmse/N );
}

}
