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
    Image<Vector3fda> g_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    Image<Vector3fda> g_o,
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

class IcpGrad3d {
 public:

  /// gs are the 3D gradients.
  template<int D, typename Derived>
  static void ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& gs_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& gs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr,
    bool verbose
    );

};

}
