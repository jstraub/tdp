
#pragma once
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/features/brief.h>
#include <tdp/manifold/SE3.h>
#include <tdp/geometry/cosy.h>
#include <tdp/camera/ray.h>

namespace tdp {

void ComputeUnitPlanes(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    Image<Vector4fda>& pl
    );

struct Plane {
  Vector3fda p_; 
  Vector3fda n_; 
  float curvature_;
  Vector3bda rgb_; 
  float grey_;
  Vector2fda gradGrey_; 
  Vector3fda dir_; 
  Vector3fda grad_; 

  Brief feat_;

  uint16_t z_;
  uint16_t lastFrame_;
  uint16_t numObs_;

  float gradNorm_;

  float w_; // weight
  float N_; // number of Obs
  float r_; // radius

  bool valid_;

  void AddObs(const Vector3fda& p, const Vector3fda& n);
  void AddObs(const Vector3fda& p, const Vector3fda& n, const Vector3bda& rgb);

  tdp::SE3f LocalCosy();

  bool Close(const Plane& other, float dotThr, float distThr, float p2plThr);

  float p2plDist(const Vector3fda& p);

  template<int D, typename Derived>
  tdp::Vector3fda Compute3DGradient(
      const tdp::SE3f& T_wc,
      const CameraBase<float,D,Derived>& cam,
      float u, float v,
      const tdp::Vector2fda& gradGrey
      ) {
    if (gradGrey.squaredNorm() > 1e-6) {
      float uGrad = u + gradGrey(0);
      float vGrad = v + gradGrey(1);
      tdp::Rayfda ray(tdp::Vector3fda::Zero(),
          cam.Unproject(uGrad,vGrad,1.));
      ray.Transform(T_wc);
      tdp::Vector3fda grad = gradGrey.norm()*(
          ray.IntersectPlane(p_,n_)-p_).normalized();
      return grad;
    }
    return tdp::Vector3fda::Zero();
  }

};

}
