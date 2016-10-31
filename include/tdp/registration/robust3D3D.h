#pragma once

#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>
//#include <tdp/opt/newtonSE3.h>
#include <tdp/opt/gradientDescentSE3.h>

namespace tdp {

template <typename T>
class Huber3D3D : public GDSE3<T> {
 public:
  Huber3D3D(const Image<Vector3fda>& pcA, const Image<Vector3fda>& pcB,
      const Image<Vector2ida>& assoc, float delta);

  ~Huber3D3D() {};

  virtual void ComputeJacobian(const SE3<T>& theta,
      Eigen::Matrix<T,6,1>* J, T* f) {

    Eigen::Matrix<float,3,6> Jse3;
    Jse3 << Eigen::Matrix3f::Zero(), Eigen::Matrix3f::Identity();
    if (f) *f = 0.f;
    if (J) J->fill(0.f);
    for (size_t i=0; i<assoc_.Area(); ++i) {
      const Vector3fda& pA = pcA_[assoc_[i](0)];
      const Vector3fda& pB = pcB_[assoc_[i](1)];
      float err = (theta*pA - pB).squaredNorm();
//      std::cout << assoc_[i].transpose() << " "
//        << pA.transpose() << " " << pB.transpose() << " "
//        << err << std::endl;
      if (J) {
        Jse3.leftCols(3) = -SO3mat<float>::invVee(theta*pcA_[assoc_[i](0)]);
        *J += Jhuber(err, delta_) * Jse3.transpose() * 2.*(theta*pcA_[assoc_[i](0)] - pcB_[assoc_[i](1)]);
      }
      if (f) *f += HuberLoss(err, delta_);
    }
    if (f) *f /= assoc_.Area();
    if (J) *J /= assoc_.Area();
  };

  static float HuberLoss(float a, float delta) {
    return (fabs(a) <= delta ? 0.5*a*a : delta*(fabs(a)-0.5*delta));
  }
  static float Jhuber(float a, float delta) {
    return (fabs(a) <= delta ? a : (a < 0 ? -delta : delta));
  }

 protected:
  const Image<Vector3fda>& pcA_;
  const Image<Vector3fda>& pcB_;
  const Image<Vector2ida>& assoc_;
  float delta_;
};

template <typename T>
Huber3D3D<T>::Huber3D3D(const Image<Vector3fda>& pcA, 
    const Image<Vector3fda>& pcB, 
    const Image<Vector2ida>& assoc,
    float delta) 
  : pcA_(pcA), pcB_(pcB), assoc_(assoc), delta_(delta) {
  this->c_=0.1;
  this->t_=0.3;
}

}

