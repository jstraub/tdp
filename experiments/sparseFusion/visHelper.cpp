#include "visHelper.h"
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <tdp/gl/gl_draw.h>

namespace tdp {

void ShowCurrentNormals(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    const std::vector<std::pair<size_t, size_t>>& assoc,
    const SE3f& T_wc,
    float scale){
  glColor4f(1,0,0.,0.5);
  pangolin::glSetFrameOfReference(T_wc.matrix());
  for (const auto& ass : assoc) {
    int32_t u = ass.second%pc.w_;
    int32_t v = ass.second/pc.w_;
    tdp::glDrawLine(pc(u,v), pc(u,v) + scale*n(u,v));
  }
  pangolin::glUnsetFrameOfReference();
}

void ShowGlobalNormals(
  const CircularBuffer<tdp::Vector3fda>& pc_w,
  const CircularBuffer<tdp::Vector3fda>& n_w,
  float scale,
  int step){
  glColor4f(0,1,0,0.5);
  for (size_t i=0; i<n_w.SizeToRead(); i+=step) {
    tdp::glDrawLine(pc_w.GetCircular(i), 
        pc_w.GetCircular(i) + scale*n_w.GetCircular(i));
  }
}

}
