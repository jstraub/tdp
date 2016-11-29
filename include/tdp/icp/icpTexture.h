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
void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector2fda> gradGrey_m,
    Image<float> grey_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    Image<Vector2fda> gradGrey_o,
    Image<float> grey_o,
    const SE3f& T_mo, 
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    float distThr,
    float lambda,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

class IcpTexture {
 public:

  template<typename CameraT>
  static void ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector2fda,3>& gradGrey_m,
    Pyramid<float,3>& grey_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector2fda,3>& gradGrey_o,
    Pyramid<float,3>& grey_o,
    const Rig<CameraT>& rig,
    const std::vector<int32_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    float lambda,
    bool verbose,
    SE3f& T_mr,
    Eigen::Matrix<float,6,6>& Sigma_mr,
    std::vector<float>& errPerLvl,
    std::vector<float>& countPerLvl
    );

};

}
