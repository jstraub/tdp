#pragma once 
#include <tdp/data/image.h>
#include <tdp/camera/camera_base.h>

namespace tdp {

template <int D, class Derived>
void OverlapGpu(
    const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA,
    const Image<Vector3fda>& pcB,
    const SE3f& T_ab, 
    const CameraBase<float,D,Derived>& camA, float& overlap, float& rmse, 
    Image<float>* errB = nullptr);

template <typename CamT>
void OverlapGpu(
    const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA,
    const Image<Vector3fda>& pcB,
    const SE3f& T_ab,
    const Rig<CamT>& rig,  float& overlap,
    float& rmse, Image<float>* errB=nullptr) {

  overlap = 0.f;
  rmse = 0.f;
  for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
    int32_t cId;
//    if (useRgbCamParasForDepth) {
      cId = rig.rgbStream2cam_[sId]; 
//    } else {
//      cId = rig.dStream2cam_[sId]; 
//    }
    CamT cam = rig.cams_[cId];
    tdp::SE3f T_rc = rig.T_rcs_[cId];

    const Image<float> greyAi = rig.GetStreamRoiOrigSize(greyA, sId);
    const Image<float> greyBi = rig.GetStreamRoiOrigSize(greyB, sId);
    const Image<Vector3fda> pcAi = rig.GetStreamRoiOrigSize(pcA, sId);
    const Image<Vector3fda> pcBi = rig.GetStreamRoiOrigSize(pcB, sId);

    Image<float>* errBi = nullptr;
    if (errB) {
      errBi = new Image<float>();
      *errBi = rig.GetStreamRoiOrigSize(*errB, sId);
    }

    float overlapi = 0.;
    float rmsei = 0.;
    OverlapGpu(greyAi, greyBi, pcAi, pcBi, T_rc.Inverse()*T_ab, 
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
}
