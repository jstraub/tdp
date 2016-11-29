/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/cuda/cuda.h>
#include <tdp/data/image.h>
#include <tdp/camera/camera.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

template<int D, class Derived>
__device__ 
inline int AssociateModelIntoCurrent(
    int x, int y, 
    const Image<Vector3fda>& pc_m,
    const SE3f& T_mo,
    const SE3f& T_co,
    const CameraBase<float,D,Derived>& cam,
    int& u, int& v
    ) {
  // project model point into camera frame to get association
  if (x < pc_m.w_ && y < pc_m.h_ ) {
    Vector3fda pc_mi = pc_m(x,y);
    if (IsValidData(pc_mi)) {
      Vector3fda pc_m_in_o = T_mo.Inverse() * pc_mi;
      // project into current camera
      Vector2fda x_m_in_o = cam.Project(T_co*pc_m_in_o);
      u = floor(x_m_in_o(0)+0.5f);
      v = floor(x_m_in_o(1)+0.5f);
      if (0 <= u && u < pc_m.w_ && 0 <= v && v < pc_m.h_
          && pc_m_in_o(2) > 0.
          && IsValidData(pc_m_in_o)) {
        return 0;
      } else {
        return 1;
      }
    } else {
      return 2;
    }
  } else {
    return 3;
  }
}

}
