/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/box.h>

namespace tdp {
Box::Box(const Eigen::Vector3f& p_min, const Eigen::Vector3f& p_max) 
  : p_min_(p_min), p_max_(p_max) {
  // The first 4 are on purpose the way they are layed out to allow
  // fast access for edges and sides.
  corners_.col(0) = p_min;
  corners_.col(1) << p_max(0), p_max(1), p_min(2);
  corners_.col(2) << p_max(0), p_min(1), p_max(2);
  corners_.col(3) << p_min(0), p_max(1), p_max(2);
  corners_.col(4) << p_max(0), p_min(1), p_min(2);
  corners_.col(5) << p_min(0), p_max(1), p_min(2);
  corners_.col(6) = p_max;
  corners_.col(7) << p_min(0), p_min(1), p_max(2);

  edges_.col(0)  = corners_.col(4) - corners_.col(0);
  edges_.col(1)  = corners_.col(5) - corners_.col(0);
  edges_.col(2)  = corners_.col(7) - corners_.col(0);
  edges_.col(3)  = corners_.col(5) - corners_.col(1);
  edges_.col(4)  = corners_.col(4) - corners_.col(1);
  edges_.col(5)  = corners_.col(6) - corners_.col(1);
  edges_.col(6)  = corners_.col(7) - corners_.col(2);
  edges_.col(7)  = corners_.col(6) - corners_.col(2);
  edges_.col(8)  = corners_.col(4) - corners_.col(2);
  edges_.col(9)  = corners_.col(6) - corners_.col(3);
  edges_.col(10) = corners_.col(7) - corners_.col(3);
  edges_.col(11) = corners_.col(5) - corners_.col(3);

  sides_.col(0)  = corners_.col(5) - corners_.col(0);
  sides_.col(1)  = corners_.col(7) - corners_.col(0);
  sides_.col(2)  = corners_.col(4) - corners_.col(0);
  sides_.col(3)  = corners_.col(5) - corners_.col(0);

  sides_.col(4)  = corners_.col(6) - corners_.col(1);
  sides_.col(5)  = corners_.col(4) - corners_.col(1);
  sides_.col(6)  = corners_.col(6) - corners_.col(1);
  sides_.col(7)  = corners_.col(5) - corners_.col(1);

  sides_.col(8)  = corners_.col(6) - corners_.col(2);
  sides_.col(9)  = corners_.col(7) - corners_.col(2);
  sides_.col(10) = corners_.col(4) - corners_.col(2);
  sides_.col(11) = corners_.col(7) - corners_.col(2);

}

Eigen::Vector3f Box::GetCenter() const {
  return 0.5*(corners_.col(0) + corners_.col(6));
}

std::vector<Box> Box::Subdivide() const {
  std::vector<Box> boxs;
  boxs.reserve(8);
  // Lower half.
  boxs.push_back(Box(corners_.col(0),
        0.5*(corners_.col(0)+corners_.col(6))));
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(5)), 
        0.5*(corners_.col(5)+corners_.col(6))));
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(4)), 
        0.5*(corners_.col(4)+corners_.col(6))));
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(1)), 
        0.5*(corners_.col(1)+corners_.col(6))));
  // Upper half.
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(7)), 
        0.5*(corners_.col(7)+corners_.col(6))));
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(3)), 
        0.5*(corners_.col(3)+corners_.col(6))));
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(2)), 
        0.5*(corners_.col(2)+corners_.col(6))));
  boxs.push_back(Box(0.5*(corners_.col(0)+corners_.col(6)), 
        corners_.col(6)));
  return boxs;
}

bool Box::IsInside(const Eigen::Vector3f& t) const {
  return (p_min_.array() <= t.array()).all() 
    && (t.array() <= p_max_.array()).all();
}

Eigen::Vector3f Box::GetSideLengths() const {
  Eigen::Vector3f ls;
  ls << edges_.col(0).norm(), edges_.col(1).norm(), edges_.col(2).norm(); 
  return ls;
}

}
