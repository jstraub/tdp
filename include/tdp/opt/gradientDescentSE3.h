#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SE3.h>
#include <tdp/manifold/gradientDescent.h>

namespace tdp {

template <typename T>
class GDSE3 : public GD<T,6,SE3<T>> {
 public:
  GDSE3();
  ~GDSE3() {};
  virtual void ComputeJacobian(const SE3<T>& theta,
      Eigen::Matrix<T,6,1>* J, T* f) = 0;
 protected:
};

template <typename T>
GDSE3<T>::GDSE3() {
  this->c_=0.1;
  this->t_=0.3;
}

}
