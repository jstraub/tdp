/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <random>
#include <Eigen/Dense>

template<typename T>
inline size_t sampleDisc(const Eigen::Matrix<T,Eigen::Dynamic,1>& pdf,
    std::mt19937& rnd) {
  std::uniform_real_distribution<T> unif(0.,1.);
  T u = unif(rnd);
  T cdf = pdf[0];
  size_t k;
  for (k=1; k< (size_t)pdf.rows(); ++k) {
    if (u <= cdf) {
      return k-1;
    }
    cdf += pdf[k];
  }
  return pdf.rows()-1;
}
