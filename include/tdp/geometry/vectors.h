#pragma once
#include <Eigen/Dense>
#include <tdp/eigen/dense.h>

namespace tdp {

/// compute dot product at B 
template<typename Derived>
float DotABC(const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& b,
    const Eigen::MatrixBase<Derived>& c) {
  Eigen::MatrixBase<Derived> dirab = a-b;
  Eigen::MatrixBase<Derived> dircb = c-b;
  return (dirab.dot(dircb)/(dirab.norm()*dircb.norm()));
}

template<typename Derived>
float LengthOfAonB(const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& b) {
  return a.dot(b);
}

template<typename Derived>
float LengthOfAorthoToB(const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& b) {
  return RejectAfromB(a,b).norm();
}

template<typename Derived>
Eigen::MatrixBase<Derived> ProjectAontoB(const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& b) {
  return (a.dot(b)/b.dot(b))*b;
}

template<typename Derived>
Eigen::MatrixBase<Derived> RejectAfromB(const Eigen::MatrixBase<Derived>& a, 
    const Eigen::MatrixBase<Derived>& b) {
  return a - ProjectAontoB(a,b);
}

template<typename Derived>
Eigen::MatrixBase<Derived> Normalize(const Eigen::MatrixBase<Derived>& a) {
  return a/a.norm();
}

}
