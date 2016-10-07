#pragma once
#include <Eigen/Dense>
#include <tdp/geometry/vectors.h>

namespace tdp {

template<typename T, int Option>
Eigen::Matrix<T,3,3,Option> Orthonormalize(const Eigen::Matrix<T,3,1,Option>& a,
  const Eigen::Matrix<T,3,1,Option>& b) {
  Eigen::Matrix<T,3,3,Option> R;
  R.col(0) = a/a.norm();
  R.col(1) = RejectAfromB(b,a);
  R.col(1) /= R.col(1).norm();
  R.col(2) = R.col(0).cross(R.col(1));
  R.col(2) /= R.col(2).norm();
  return R;
}

template<typename T, int Option>
Eigen::Matrix<T,3,3,Option> OrthonormalizeFromYZ(
    const Eigen::Matrix<T,3,1,Option>& y,
    const Eigen::Matrix<T,3,1,Option>& z) {
  Eigen::Matrix<T,3,3,Option> R;
  R.col(2) = Normalize(z);
  R.col(1) = Normalize(RejectAfromB(y,z));
  R.col(0) = Normalize(R.col(1).cross(R.col(2)));
  return R;
}

}
