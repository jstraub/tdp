/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node_AA.h>

namespace tdp {

template<typename T>
NodeAA<T>::NodeAA(const Box<T>& box, std::vector<uint32_t> ids) 
  : NodeLin<T>(box,ids)
{
  this->Linearize(box);
}

template<typename T>
NodeAA<T>::NodeAA(const NodeAA<T>& node) 
  : NodeLin<T>(node)
{ }

template<typename T>
Eigen::Quaternion<T> NodeAA<T>::Project(const Eigen::Matrix<T,3,1>& c) const {
  Eigen::Quaternion<T> q;
  T theta = c.norm();
  if (theta > 1e-9) {
    q.w() = cos(theta*0.5);
    q.vec() = -c*sin(theta*0.5)/theta;
  } else {
    q.w() = 1.;
    q.vec().fill(0.);
  }
  return q;
}

template<typename T>
std::vector<NodeAA<T>> NodeAA<T>::Branch() const {
  std::vector<NodeAA<T>> nodes;
  nodes.reserve(8);
  std::vector<NodeR3<T>> boxs = this->nodeLin_.Branch();
  for (uint32_t i=0; i < boxs.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeAA<T>(boxs[i].GetBox(), ids));
  }
  return nodes;
}

template<typename T>
std::list<NodeAA<T>> TessellateAA() {
//  Eigen::Matrix<T,3,1> p_min(-M_PI,-M_PI,-M_PI);
//  Eigen::Matrix<T,3,1> p_max( M_PI, M_PI, M_PI);
  // split into 4 cubes for level 4 so that at level 2 we have 256
  // cubes which is close to the 270 of the 600-cell tessellation.
  Eigen::Matrix<T,3,1> p_min(-M_PI,-M_PI,-M_PI);
  Eigen::Matrix<T,3,1> p_max( 0., 0., M_PI);
  NodeAA<T> root00(Box<T>(p_min, p_max),std::vector<uint32_t>(1,0));
  p_min << -M_PI,0.,-M_PI;
  p_max << 0., M_PI, M_PI;
  NodeAA<T> root01(Box<T>(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0., -M_PI,-M_PI;
  p_max << M_PI, 0., M_PI;
  NodeAA<T> root10(Box<T>(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0.,0.,-M_PI;
  p_max << M_PI, M_PI, M_PI;
  NodeAA<T> root11(Box<T>(p_min, p_max),std::vector<uint32_t>(1,1));
//  std::cout << root.ToString() << std::endl;
  std::vector<NodeAA<T>> l1 = root00.Branch();
  std::vector<NodeAA<T>> a = root01.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root10.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root11.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  std::list<NodeAA<T>> nodes;
  for (auto& node1 : l1) {
    std::vector<NodeAA<T>> l2 = node1.Branch();
//    for (auto& node2 : l2) {
//      std::vector<NodeAA<T>> l3 = node2.Branch();
//      for (auto& node3 : l3) {
//        std::vector<NodeAA<T>> l4 = node3.Branch();
        nodes.insert(nodes.end(), l2.begin(), l2.end());
//      }
//    }
  }
  return nodes;
}

template class NodeAA<float>;
template class NodeAA<double>;

}
