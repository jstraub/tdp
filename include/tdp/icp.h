#pragma once

#include <vector>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

#ifdef CUDA_FOUND
void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_c,
    Image<Vector3fda> n_c,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    const Camera<float>& cam,
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_c,
    Image<Vector3fda> n_c,
    SE3f& T_mc,
    const Camera<float>& cam,
    float angleThr,
    float distThr,
    Image<float>& assoc_m,
    Image<float>& assoc_c
    );

void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_c,
    Image<Vector3fda> pc_c,
    Matrix3fda R_mc, 
    Vector3fda t_mc, 
    const Camera<float>& cam,
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
  static void ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_c,
    Pyramid<Vector3fda,3>& ns_c,
    SE3f& T_mc,
    const Camera<float>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    );

  /// Compute relative rotation between two surface normal "images";
  /// uese projective data association
  static void ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_c,
    Pyramid<Vector3fda,3>& pcs_c,
    SE3f& T_mc,
    const Camera<float>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);

 private:
};



}
