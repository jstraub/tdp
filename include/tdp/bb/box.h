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
  Box(const Eigen::Vector3d& p_min, const Eigen::Vector3d& p_max);
  ~Box() = default;
  std::vector<Box> Subdivide() const;
  Eigen::Vector3d GetCenter() const;
  bool IsInside(const Eigen::Vector3d& t) const;
  void GetCorner(uint32_t i, Eigen::Vector3d& c) const { c = corners_.col(i);}
  void GetEdge(uint32_t i, Eigen::Vector3d& e0, Eigen::Vector3d& d) const
  {d = edges_.col(i); e0 = corners_.col(i/3);}
  void GetSide(uint32_t i, Eigen::Vector3d& p0, Eigen::Matrix<double,3,2>& E) const
  {p0 = corners_.col(i/2); E = sides_.middleCols<2>(i*2);}
  Eigen::Vector3d GetSideLengths() const;
  double GetVolume() const { return (p_max_- p_min_).prod();}
 private:
  Eigen::Matrix<double, 3, 1> p_min_;
  Eigen::Matrix<double, 3, 1> p_max_;
  Eigen::Matrix<double, 3, 8> corners_;
  Eigen::Matrix<double, 3, 12> edges_;
  Eigen::Matrix<double, 3, 12> sides_;
};


}

