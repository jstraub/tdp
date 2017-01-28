/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <tdp/data/image.h>
#include <tdp/data/circular_buffer.h>
#include <tdp/manifold/SE3.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/plane.h>
#include <tdp/camera/camera_base.h>
#include <tdp/features/brief.h>

namespace tdp {

template <int D, typename Derived>
bool ExtractClosestBrief(
    const Image<Vector3fda>& pc, 
    const Image<uint8_t>& grey,
    const Image<Vector2ida>& pts,
    const Image<float>& orientation,
    const Vector3fda& pci,
    const Vector3fda& ni,
    const SE3f& T_wc, 
    const CameraBase<float,D,Derived>& cam,
    size_t W,
    size_t u, size_t v,
    Brief& feat) {

  feat.pt_(0) = u;
  feat.pt_(1) = v;
  feat.desc_.fill(0);
  // try finding a closeby feature point and get the feature there
  for (size_t j=0; j<pts.Area(); ++j) {
    if ((pts[j].cast<float>() - feat.pt_.cast<float>()).norm() < W) {
      feat.orientation_ = orientation[j];
      if (!tdp::ExtractBrief(grey, feat)) 
        feat.desc_.fill(0);
      else {
        Vector3fda p = pc(pts[j](0), pts[j](1));
        if (!IsValidData(p)) {
          tdp::Rayfda ray(Vector3fda::Zero(), 
              cam.Unproject(pts[j](0), pts[j](1), 1.));
          p = ray.IntersectPlane(pci, ni);
        }
//        std::cout << "FAST feat at " << pts[j].transpose() 
//          << " for " << feat.pt_.transpose() 
//          << " pc " << pc(pts[j](0), pts[j](1)).transpose()
//          << " pIntersect " << p.transpose()
//          << std::endl;
        // TODO: not going to be updated if pl.p_ is !
        feat.p_c_ = T_wc*p; 
      }
      return true;
    }
  }
  return false;
}

}
