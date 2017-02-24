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

namespace tdp {

template<int D, typename Derived>
void ICPStep (
    Image<float> grey_p,
    Image<float> grey_c,
    Image<Vector2fda> gradGrey_c,
    Image<Vector3fda> rays,
    SO3f R_cp, 
    const CameraBase<float,D,Derived> cam,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,3,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

/// Photometric alignment of only rotation
class PhotometricSO3 {
 public:

  template<typename CameraT>
  static void ComputeProjective(
    Pyramid<float,3>& grey_p,
    Pyramid<float,3>& grey_c,
    Pyramid<Vector2fda,3>& gradGrey_c,
    Pyramid<Vector3fda,3>& rays,
    const std::vector<size_t>& maxIt, 
    bool verbose,
    SO3f& R_cp
    );

};

}
