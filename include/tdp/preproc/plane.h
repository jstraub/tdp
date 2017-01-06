
#pragma once
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/features/brief.h>
#include <tdp/manifold/SE3.h>
#include <tdp/geometry/cosy.h>

namespace tdp {

void ComputeUnitPlanes(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    Image<Vector4fda>& pl
    );

struct Plane {
  Vector3fda p_; 
  Vector3fda n_; 
  Vector3bda rgb_; 
  float grey_;
  Vector2fda gradGrey_; 
  Vector3fda dir_; 

  Brief feat_;

  uint16_t z_;
  uint32_t lastFrame_;
  uint32_t numObs_;

  float w_; // weight

  void AddObs(const Vector3fda& p, const Vector3fda& n);
  void AddObs(const Vector3fda& p, const Vector3fda& n, const Vector3bda& rgb);

  tdp::SE3f LocalCosy();

  bool Close(const Plane& other, float dotThr, float distThr, float p2plThr);

};

}
