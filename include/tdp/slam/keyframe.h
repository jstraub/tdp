#pragma once
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/rig.h>
#include <tdp/camera/photometric.h>
namespace tdp {

/// KeyFrame
///
/// Use emplace_back in std::vector to not run into memory issues.
/// Make sure that ManagedImage and ManagedPyramdid (and all additional
/// containers) have a propper move constructor.
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
      const Pyramid<Vector2fda,3>& gradGrey,
      const Image<Vector3bda>& rgb,
      const Image<float>& d,
      const SE3f& T_wk) :
    pc_(pc.w_, pc.h_), n_(n.w_, n.h_), rgb_(rgb.w_, rgb.h_), 
    d_(d.w_, d.h_), pyrPc_(pc.w_, pc.h_), pyrN_(n.w_, n.h_),
    pyrGrey_(grey.w_, grey.h_), pyrGradGrey_(gradGrey.w_, gradGrey.h_),
    T_wk_(T_wk)
  {
    pc_.CopyFrom(pc.GetConstImage(0),  cudaMemcpyDeviceToHost);
    n_.CopyFrom(n.GetConstImage(0),    cudaMemcpyDeviceToHost);
    d_.CopyFrom(d,                cudaMemcpyDeviceToHost);
    rgb_.CopyFrom(rgb,            cudaMemcpyHostToHost);

    pyrPc_.CopyFrom(pc,     cudaMemcpyDeviceToHost);
    pyrN_.CopyFrom(n,       cudaMemcpyDeviceToHost);
    pyrGrey_.CopyFrom(grey, cudaMemcpyHostToHost);
    pyrGradGrey_.CopyFrom(gradGrey, cudaMemcpyDeviceToHost);
  }

  ManagedHostImage<Vector3fda> pc_;
  ManagedHostImage<Vector3fda> n_;
  ManagedHostImage<Vector3bda> rgb_;
  ManagedHostImage<float> d_;

  ManagedHostPyramid<Vector3fda,3> pyrPc_;
  ManagedHostPyramid<Vector3fda,3> pyrN_;
  ManagedHostPyramid<float,3> pyrGrey_;
  ManagedHostPyramid<Vector2fda,3> pyrGradGrey_;

  SE3f T_wk_; // Transformation from keyframe to world

};

/// Compute overlap fraction between two KFs in a given pyramid level
/// lvl
template <int D, class Derived>
void Overlap(const KeyFrame& kfA, const KeyFrame& kfB,
    const CameraBase<float, D,Derived>& cam, int lvl, float& overlap,
    float& rmse, const SE3f* T_ab = nullptr, Image<float>* errB=nullptr) {
  tdp::SE3f T_ab_ = kfA.T_wk_.Inverse() * kfB.T_wk_;
  if (T_ab)
    T_ab_ = *T_ab;

  const Image<float> greyA = kfA.pyrGrey_.GetConstImage(lvl);
  const Image<float> greyB = kfB.pyrGrey_.GetConstImage(lvl);
  const Image<Vector3fda> pcA = kfA.pyrPc_.GetConstImage(lvl);
  const Image<Vector3fda> pcB = kfB.pyrPc_.GetConstImage(lvl);
  Overlap(greyA, greyB, pcA, pcB, T_ab_, 
      ScaleCamera<float,D,Derived>(cam,pow(0.5,lvl)), 
      overlap, rmse, errB);
}

template <typename CamT>
void Overlap(const KeyFrame& kfA, const KeyFrame& kfB,
    const Rig<CamT>& rig, int lvl, float& overlap,
    float& rmse, const SE3f* T_ab = nullptr, Image<float>* errB=nullptr) {
  tdp::SE3f T_ab_ = kfA.T_wk_.Inverse() * kfB.T_wk_;
  if (T_ab)
    T_ab_ = *T_ab;

  overlap = 0.f;
  rmse = 0.f;
  const Image<float> greyA = kfA.pyrGrey_.GetConstImage(lvl);
  const Image<float> greyB = kfB.pyrGrey_.GetConstImage(lvl);
  const Image<Vector3fda> pcA = kfA.pyrPc_.GetConstImage(lvl);
  const Image<Vector3fda> pcB = kfB.pyrPc_.GetConstImage(lvl);

  for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
    int32_t cId;
//    if (useRgbCamParasForDepth) {
      cId = rig.rgbStream2cam_[sId]; 
//    } else {
//      cId = rig.dStream2cam_[sId]; 
//    }
    CamT cam = rig.cams_[cId].Scale(pow(0.5,lvl));
    tdp::SE3f T_rc = rig.T_rcs_[cId];

    const Image<float> greyAi = rig.GetStreamRoiOrigSize(greyA, sId);
    const Image<float> greyBi = rig.GetStreamRoiOrigSize(greyB, sId);
    const Image<Vector3fda> pcAi = rig.GetStreamRoiOrigSize(pcA, sId);
    const Image<Vector3fda> pcBi = rig.GetStreamRoiOrigSize(pcB, sId);

    //TODO
    ManagedDeviceImage<float> cuGreyAi(greyAi.w_, greyAi.h_);
    cuGreyAi.CopyFrom(greyAi, cudaMemcpyHostToDevice);
    ManagedDeviceImage<float> cuGreyBi(greyBi.w_, greyBi.h_);
    cuGreyBi.CopyFrom(greyBi, cudaMemcpyHostToDevice);
    ManagedDeviceImage<Vector3fda> cuPcAi(pcAi.w_, pcAi.h_);
    cuPcAi.CopyFrom(pcAi, cudaMemcpyHostToDevice);
    ManagedDeviceImage<Vector3fda> cuPcBi(pcBi.w_, pcBi.h_);
    cuPcBi.CopyFrom(pcBi, cudaMemcpyHostToDevice);

    Image<float>* errBi = nullptr;
    if (errB) {
      errBi = new Image<float>();
      *errBi = rig.GetStreamRoiOrigSize(*errB, sId);
//      std::cout << errBi->Description() << std::endl;
    }

//    std::cout << cuGreyBi.Description() << std::endl
//      << greyBi.Description() << std::endl;
//    std::cout << cuGreyAi.Description() << std::endl
//      << greyAi.Description() << std::endl;
//    std::cout << cuPcBi.Description() << std::endl
//      << pcBi.Description() << std::endl;

    float overlapi = 0.;
    float rmsei = 0.;
//    Overlap(greyAi, greyBi, pcBi, T_rc.Inverse()*T_ab_, 
//        cam, overlapi, rmsei, errBi);
    OverlapGpu(cuGreyAi, cuGreyBi, cuPcAi, cuPcBi, T_rc.Inverse()*T_ab_, 
        cam, overlapi, rmsei, errBi);
    rmse += rmsei;
    overlap += overlapi;

    if (errB) {
      delete errBi;
    }
  }
  rmse /= rig.dStream2cam_.size();
  overlap /= rig.dStream2cam_.size();
}

template <int D, class Derived>
void Overlap(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA, 
    const Image<Vector3fda>& pcB, 
    const SE3f& T_ab, 
    const CameraBase<float,D,Derived>& camA, float& overlap, float& rmse, 
    Image<float>* errB=nullptr) {

  float N = 0.f;
  overlap = 0.f;
  rmse = 0.f;
//  if (errB) 
//    std::cout << errB->Description() << std::endl;
  for (size_t i=0; i<pcB.Area(); ++i) {
    if (IsValidData(pcB[i])) {
      Eigen::Vector2f x = camA.Project(T_ab*pcB[i]);
      if (greyA.Inside(x) && i < greyB.Area()) {
        ++overlap;
        float y = greyA.GetBilinear(x)-greyB[i];
        rmse += y*y;
        if (errB) (*errB)[i] = sqrt(y*y);
      }
      ++N;
    }
  }
  overlap /= N;
  rmse = sqrt( rmse/N );
}



}
