#pragma once

#include <gtest/gtest.h>
#include <Eigen/Dense>

#define EPS 1e-9


template <typename DerivedA, typename DerivedB>
bool IsAppox(const Eigen::MatrixBase<DerivedA>& a, 
    const Eigen::MatrixBase<DerivedB>& b,
    float eps=EPS) {
  bool same = a.isApprox(b,eps);
  if (!same) {
    std::cout << a << std::endl << b << std::endl;
  }
  return same;
}

