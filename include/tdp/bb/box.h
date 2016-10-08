/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

namespace tdp {

class Box {
 public:
  Box(const Eigen::Vector3f& p_min, const Eigen::Vector3f& p_max);
  ~Box() = default;
  std::vector<Box> Subdivide() const;
  Eigen::Vector3f GetCenter() const;
  bool IsInside(const Eigen::Vector3f& t) const;
  void GetCorner(uint32_t i, Eigen::Vector3f& c) const { c = corners_.col(i);}
  void GetEdge(uint32_t i, Eigen::Vector3f& e0, Eigen::Vector3f& d) const
  {d = edges_.col(i); e0 = corners_.col(i/3);}
  void GetSide(uint32_t i, Eigen::Vector3f& p0, Eigen::Matrix<float,3,2>& E) const
  {p0 = corners_.col(i/2); E = sides_.middleCols<2>(i*2);}
  Eigen::Vector3f GetSideLengths() const;
  float GetVolume() const { return (p_max_- p_min_).prod();}
 private:
  Eigen::Matrix<float, 3, 1> p_min_;
  Eigen::Matrix<float, 3, 1> p_max_;
  Eigen::Matrix<float, 3, 8> corners_;
  Eigen::Matrix<float, 3, 12> edges_;
  Eigen::Matrix<float, 3, 12> sides_;
};


}

