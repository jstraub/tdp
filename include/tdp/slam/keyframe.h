#pragma once
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>
namespace tdp {

struct KeyFrame {
  KeyFrame() {};
  KeyFrame(const Image<Vector3fda>& pc, 
      const Image<Vector3fda>& n,
      const Image<Vector3bda>& rgb,
      const SE3f& T_wk) :
    pc_(pc.w_, pc.h_), n_(n.w_, n.h_), rgb_(rgb.w_, rgb.h_), 
    T_wk_(T_wk)
  {
    pc_.CopyFrom(pc, cudaMemcpyHostToHost);
    n_.CopyFrom(n, cudaMemcpyHostToHost);
    rgb_.CopyFrom(rgb, cudaMemcpyHostToHost);
  }

  ManagedHostImage<Vector3fda> pc_;
  ManagedHostImage<Vector3fda> n_;
  ManagedHostImage<Vector3bda> rgb_;
  SE3f T_wk_; // Transformation from keyframe to world
};

}
