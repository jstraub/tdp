#include <assert.h>
#include <tdp/preproc/depth.h>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void BoundingBox(
    const Image<Vector3fda>& pc,
    Eigen::Vector3f& minPc,
    Eigen::Vector3f& maxPc, bool resetMinMax) {
  if (resetMinMax) {
    minPc(0) = std::numeric_limits<float>::max();
    minPc(1) = std::numeric_limits<float>::max();
    minPc(2) = std::numeric_limits<float>::max();
    maxPc(0) = std::numeric_limits<float>::lowest();
    maxPc(1) = std::numeric_limits<float>::lowest();
    maxPc(2) = std::numeric_limits<float>::lowest();
  }

  for (size_t i=0; i<pc.Area(); ++i) {
    for (size_t j=0; j<3; ++j) {
      if (pc[i](j) < minPc(j)) minPc(j) = pc[i](j);
      if (pc[i](j) > maxPc(j)) maxPc(j) = pc[i](j);
    }
  }
}

//void Depth2PC(
//    const Image<float>& d,
//    const Camera<float>& cam,
//    Image<Vector3fda>& pc
//    ) {
//  for (size_t v=0; v<pc.h_; ++v)
//    for (size_t u=0; u<pc.w_; ++u) 
//      if (u<d.w_ && v<d.h_) {
//        pc(u,v) = cam.Unproject(u,v,d(u,v));
//      }
//}

}
