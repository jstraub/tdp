/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/box.h>

namespace tdp {

template <typename T>
Box<T>::Box(const Eigen::Matrix<T,3,1>& p_min, 
    const Eigen::Matrix<T,3,1>& p_max) 
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

template <typename T>
Eigen::Matrix<T,3,1> Box<T>::GetCenter() const {
  return 0.5*(corners_.col(0) + corners_.col(6));
}

template <typename T>
std::vector<Box<T>> Box<T>::Subdivide() const {
  std::vector<Box<T>> boxs;
  boxs.reserve(8);
  // Lower half.
  boxs.push_back(Box<T>(corners_.col(0),
        0.5*(corners_.col(0)+corners_.col(6))));
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(5)), 
        0.5*(corners_.col(5)+corners_.col(6))));
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(4)), 
        0.5*(corners_.col(4)+corners_.col(6))));
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(1)), 
        0.5*(corners_.col(1)+corners_.col(6))));
  // Upper half.
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(7)), 
        0.5*(corners_.col(7)+corners_.col(6))));
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(3)), 
        0.5*(corners_.col(3)+corners_.col(6))));
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(2)), 
        0.5*(corners_.col(2)+corners_.col(6))));
  boxs.push_back(Box<T>(0.5*(corners_.col(0)+corners_.col(6)), 
        corners_.col(6)));
  return boxs;
}

template <typename T>
bool Box<T>::IsInside(const Eigen::Matrix<T,3,1>& t) const {
  return (p_min_.array() <= t.array()).all() 
    && (t.array() <= p_max_.array()).all();
}

template <typename T>
Eigen::Matrix<T,3,1> Box<T>::GetSideLengths() const {
  Eigen::Matrix<T,3,1> ls;
  ls << edges_.col(0).norm(), edges_.col(1).norm(), edges_.col(2).norm(); 
  return ls;
}

template class Box<float>;
template class Box<double>;

}
