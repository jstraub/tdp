
#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SE3.h>
#include <tdp/manifold/newton.h>

namespace tdp {

template <typename T>
class NewtonSE3 : public Newton<T,6,SE3<T>> {
 public:
  NewtonSE3();
  ~NewtonSE3() {};
  virtual void ComputeJacobianAndHessian(const SE3<T>& theta,
      Eigen::Matrix<T,6,6>* H, Eigen::Matrix<T,6,1>* J, T* f) = 0;
 protected:
};

template <typename T>
NewtonSE3<T>::NewtonSE3() {
  this->c_=0.1;
  this->t_=0.5;
}

}
