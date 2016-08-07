
#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/newton.h>

namespace tdp {

template <typename T>
class NewtonSO3 : public Newton<T,6,SO3<T>> {
 public:
  NewtonSO3();
  ~NewtonSO3() {};
  virtual void ComputeJacobianAndHessian(const SO3<T>& theta,
      Eigen::Matrix<T,6,6>* H, Eigen::Matrix<T,6,1>* J, T* f) = 0;
 protected:
};

template <typename T>
NewtonSO3<T>::NewtonSO3() {
  this->c_=0.1;
  this->t_=0.5;
}

}
