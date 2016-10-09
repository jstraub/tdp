#pragma once

#include <vector>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

#ifdef CUDA_FOUND
template<int D, typename Derived>
void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    const SE3f& T_mo, 
    const SE3f& T_mc, 
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

template<int D, typename Derived>
void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    const SE3f& T_mo,
    const CameraBase<float,D,Derived>& cam,
    float angleThr,
    float distThr,
    Image<float>& assoc_m,
    Image<float>& assoc_o
    );

template<int D, typename Derived>
void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    const SE3f& T_mo, 
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count
    );
#endif

class ICP {
 public:

  /// Compute realtive pose between the given depth and normals and the
  /// model; uses pyramids, projective data association and
  /// point-to-plane distance
  template<int D, typename Derived>
  static void ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    );

  /// Same as above but for multi-camera rigs
  template<typename CameraT>
  static void ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    const Rig<CameraT>& rig,
    const std::vector<size_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    SE3f& T_mr,
    std::vector<float>& errPerLvl,
    std::vector<float>& countPerLvl
    );

  /// Compute relative rotation between two surface normal "images";
  /// uese projective data association
  template<int D, typename Derived>
  static void ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);

 private:
};



}
