#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/gradientDescent.h>

namespace tdp {

template <typename T>
class GDSO3 : public GD<T,3,SO3<T>> {
 public:
  GDSO3();
  ~GDSO3() {};
  virtual void ComputeJacobian(const SO3<T>& theta, Eigen::Matrix<T,3,1>* J, T* f) = 0;
 protected:
};

template <typename T>
GDSO3<T>::GDSO3() {
  this->c_=0.1;
  this->t_=0.3;
}

}
