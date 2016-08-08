#pragma once
#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>
#include <tdp/camera.h>

namespace tdp {

TDP_HOST_DEVICE
inline 
Eigen::Vector2f TransformHomography(const Eigen::Vector2f& u_d, 
    const SE3<float>& T_rd, 
    const Camera<float>& camR, 
    const Camera<float>& camD, 
    const Eigen::Vector3f& nd_r) {
  Eigen::Matrix3f H = (T_rd.rotation().matrix()-T_rd.translation()*nd_r.transpose());
  return camR.Project(H*camD.Unproject(u_d(0), u_d(1), 1.));
};

}
