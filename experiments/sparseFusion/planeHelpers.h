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

namespace tdp {

template<int D, class Derived>
void ExtractPlanes(
    const Image<Vector3fda>& pc, 
    const Image<Vector3bda>& rgb,
    const Image<uint8_t>& grey,
    const Image<float>& greyFl,
    const Image<Vector2fda>& gradGrey,
    const Image<Vector2ida>& pts,
    const Image<float>& orientation,
    const Image<uint8_t>& mask, uint32_t W, size_t frame,
    const SE3f& T_wc, 
    const CameraBase<float,D,Derived>& cam,
    Image<Vector4fda>& dpc, 
    ManagedHostCircularBuffer<Plane>& pl_w,
    ManagedHostCircularBuffer<Vector3fda>& pc_w,
    ManagedHostCircularBuffer<Vector3fda>& pc0_w,
    ManagedHostCircularBuffer<Vector3bda>& rgb_w,
    ManagedHostCircularBuffer<Vector3fda>& n_w,
    ManagedHostCircularBuffer<float>& rs
    ) {
  Plane pl;
  tdp::Brief feat;
  Vector3fda n, p;
  float curv;
  for (size_t i=0; i<mask.Area(); ++i) {
    if (mask[i] 
        && tdp::IsValidData(pc[i])
        && pc[i].norm() < 5. 
        && 0.3 < pc[i].norm() )  {
//      uint32_t Wscaled = floor(W*pc[i](2));
      uint32_t Wscaled = W;
//      if (tdp::NormalViaScatter(pc, i%mask.w_, i/mask.w_, Wscaled, n)) {
      if (tdp::NormalViaVoting(pc, i%mask.w_, i/mask.w_, Wscaled, 0.29,
            dpc, n, curv, p)) {
        ExtractClosestBrief(pc, grey, pts, orientation, 
            p, n, T_wc, cam, Wscaled, i%mask.w_, i/mask.w_, feat);
        pl.p_ = T_wc*p;
        pl.n_ = T_wc.rotation()*n;
        pl.curvature_ = curv;
        pl.rgb_ = rgb[i];
        pl.gradGrey_ = gradGrey[i];
        pl.grey_ = greyFl[i];
        pl.lastFrame_ = frame;
        pl.w_ = 1.;
        pl.numObs_ = 1;
        pl.feat_ = feat;
//        pl.r_ = 2*W*pc[i](2)/cam.params_(0); // unprojected radius in m
        pl.r_ = p(2)/cam.params_(0); // unprojected radius in m

        pl_w.Insert(pl);
        pc_w.Insert(pl.p_);
        pc0_w.Insert(pl.p_);
        n_w.Insert(pl.n_);
        rgb_w.Insert(pl.rgb_);
        rs.Insert(pl.r_);
      }
    }
  }
}
}
}
