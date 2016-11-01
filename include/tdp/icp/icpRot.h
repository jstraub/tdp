#pragma once

#include <vector>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

template<int D, typename Derived>
void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    const SE3f& T_mo, 
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count
    );

/// Compute relative rotation between two surface normal "images";
/// uese projective data association
template<int D, typename Derived>
  void ComputeProjectiveRotation(
      Pyramid<Vector3fda,3>& ns_m,
      Pyramid<Vector3fda,3>& ns_o,
      Pyramid<Vector3fda,3>& pcs_o,
      SE3f& T_mo,
      const SE3f& T_cm,
      const CameraBase<float,D,Derived>& cam,
      const std::vector<size_t>& maxIt, float angleThr_deg);

}
