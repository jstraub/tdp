/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node_TpS3.h>

namespace tdp {

template<typename T>
NodeTpS3<T>::NodeTpS3(const Box<T>& box, std::vector<uint32_t> ids) 
  : NodeLin<T>(box, ids)
{ 
  this->Linearize(box);
}

template<typename T>
NodeTpS3<T>::NodeTpS3(const NodeTpS3<T>& node) 
  : NodeLin<T>(node)
{ }

template<typename T>
Eigen::Quaternion<T> NodeTpS3<T>::Project(const Eigen::Matrix<T,3,1>& c) const {
  // In accordance with the other tessellation approaches
  static Eigen::Matrix<T,4,1> north(1.,0.,0.,0.);
  static S<T,4> s3(north);
  Eigen::Matrix<T,4,1> qvec = s3.Exp(s3.ToAmbient(c)).vector();
  return Eigen::Quaternion<T>(qvec(0),qvec(1),qvec(2),qvec(3));
}

template<typename T>
std::vector<NodeTpS3<T>> NodeTpS3<T>::Branch() const {
  std::vector<NodeTpS3<T>> nodes;
  nodes.reserve(8);
  std::vector<NodeR3<T>> boxs = this->nodeLin_.Branch();
  for (uint32_t i=0; i < boxs.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeTpS3<T>(boxs[i].GetBox(), ids));
  }
  return nodes;
}

template<typename T>
std::list<NodeTpS3<T>> TessellateTpS3() {
  // split into 4 cubes for level 4 so that at level 2 we have 256
  // cubes which is close to the 270 of the 600-cell tessellation.
  Eigen::Matrix<T,3,1> p_min(-0.5*M_PI,-0.5*M_PI,-0.5*M_PI);
  Eigen::Matrix<T,3,1> p_max( 0., 0., 0.5*M_PI);
  NodeTpS3<T> root00(Box<T>(p_min, p_max),std::vector<uint32_t>(1,0));
  p_min << -0.5*M_PI,0.,-0.5*M_PI;
  p_max << 0., 0.5*M_PI, 0.5*M_PI;
  NodeTpS3<T> root01(Box<T>(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0., -0.5*M_PI,-0.5*M_PI;
  p_max << 0.5*M_PI, 0., 0.5*M_PI;
  NodeTpS3<T> root10(Box<T>(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0.,0.,-0.5*M_PI;
  p_max << 0.5*M_PI, 0.5*M_PI, 0.5*M_PI;
  NodeTpS3<T> root11(Box<T>(p_min, p_max),std::vector<uint32_t>(1,1));
//  std::cout << root.ToString() << std::endl;
  std::vector<NodeTpS3<T>> l1 = root00.Branch();
  std::vector<NodeTpS3<T>> a = root01.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root10.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root11.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  std::list<NodeTpS3<T>> nodes;
  for (auto& node1 : l1) {
    std::vector<NodeTpS3<T>> l2 = node1.Branch();
//    for (auto& node2 : l2) {
//      std::vector<NodeTpS3<T>> l3 = node2.Branch();
//      for (auto& node3 : l3) {
//        std::vector<NodeTpS3<T>> l4 = node3.Branch();
        nodes.insert(nodes.end(), l2.begin(), l2.end());
//      }
//    }
  }
  return nodes;
}

template class NodeTpS3<float>;
template class NodeTpS3<double>;

}
