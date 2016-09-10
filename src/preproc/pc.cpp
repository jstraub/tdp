#include <assert.h>
#include <tdp/preproc/depth.h>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void Depth2PC(
    const Image<float>& d,
    const Camera<float>& cam,
    Image<Vector3fda>& pc
    ) {
  for (size_t v=0; v<pc.h_; ++v)
    for (size_t u=0; u<pc.w_; ++u) 
      if (u<d.w_ && v<d.h_) {
        pc(u,v) = cam.Unproject(u,v,d(u,v));
      }
}

}
