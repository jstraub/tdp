/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

namespace tdp {

template <typename T>
class Box {
 public:
  Box(const Eigen::Matrix<T,3,1>& p_min, const Eigen::Matrix<T,3,1>& p_max);
  ~Box() = default;
  std::vector<Box<T>> Subdivide() const;
  Eigen::Matrix<T,3,1> GetCenter() const;
  bool IsInside(const Eigen::Matrix<T,3,1>& t) const;
  void GetCorner(uint32_t i, Eigen::Matrix<T,3,1>& c) const { c = corners_.col(i);}
  void GetEdge(uint32_t i, Eigen::Matrix<T,3,1>& e0, 
      Eigen::Matrix<T,3,1>& d) const {
    d = edges_.col(i); e0 = corners_.col(i/3);
  }
  void GetSide(uint32_t i, Eigen::Matrix<T,3,1>& p0, 
      Eigen::Matrix<T,3,2>& E) const {
    p0 = corners_.col(i/2); E = sides_.middleCols(i*2,2);
  }
  Eigen::Matrix<T,3,1> GetSideLengths() const;
  T GetVolume() const { return (p_max_- p_min_).prod();}
 private:
  Eigen::Matrix<T, 3, 1> p_min_;
  Eigen::Matrix<T, 3, 1> p_max_;
  Eigen::Matrix<T, 3, 8> corners_;
  Eigen::Matrix<T, 3, 12> edges_;
  Eigen::Matrix<T, 3, 12> sides_;
};

}

