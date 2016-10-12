/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node_Lin.h>

namespace tdp {

template<typename T>
NodeLin<T>::NodeLin(const Box<T>& box, std::vector<uint32_t> ids) 
  : BaseNode<T>(ids), nodeLin_(box, ids)
{ }

template<typename T>
NodeLin<T>::NodeLin(const NodeLin<T>& node) 
  : BaseNode<T>(node.GetIds(), node.GetLB(), node.GetUB()),
  nodeLin_(node.nodeLin_), qs_(node.qs_), q_lb_(node.q_lb_)
{ }

//void NodeLin::Linearize(const Box<T>& box) {
//  // subdivide box in Lin space into 4 tetrahedra
//  // https://www.ics.uci.edu/~eppstein/projects/tetra/
//  nodeS3s_.reserve(5);
//  // NodeS3 1: 0 4 5 7
//  Tetrahedron4D t = TetraFromBox(box, 0, 4, 5, 7);
//  std::vector<uint32_t> idsInternal(1,0);
//  nodeS3s_.push_back(NodeS3(t, idsInternal));
//  // NodeS3 2: 1 4 5 6
//  t = TetraFromBox(box, 1, 4, 5, 6);
//  idsInternal = std::vector<uint32_t>(1,1);
//  nodeS3s_.push_back(NodeS3(t, idsInternal));
//  // NodeS3 3: 2 4 6 7
//  t = TetraFromBox(box, 2, 4, 6, 7);
//  idsInternal = std::vector<uint32_t>(1,2);
//  nodeS3s_.push_back(NodeS3(t, idsInternal));
//  // NodeS3 4: 3 5 6 7
//  t = TetraFromBox(box, 3, 5, 6, 7);
//  idsInternal = std::vector<uint32_t>(1,3);
//  nodeS3s_.push_back(NodeS3(t, idsInternal));
//  // NodeS3 5: 0 1 2 3
//  t = TetraFromBox(box, 0, 1, 2, 3);
//  idsInternal = std::vector<uint32_t>(1,4);
//  nodeS3s_.push_back(NodeS3(t, idsInternal));
//}
template<typename T>
NodeS3<T> NodeLin<T>::GetNodeS3() const {
  Eigen::Matrix<T,4,4> Q;
  for (uint32_t i=0; i<4; ++i) {
    Q(0,i) = qs_[i].w();
    Q.block(1,i,3,1) = qs_[i].vec();
  }
  Tetrahedron4D<T> t(Q);
  return NodeS3<T>(t, this->ids_);
}

template<typename T>
void NodeLin<T>::Linearize(const Box<T>& box) {
  qs_.reserve(8);
  for (uint32_t i=0; i<8; ++i) {
    Eigen::Matrix<T,3,1> c;
    box.GetCorner(i,c);
    qs_.push_back(Project(c));
  }
}

template<typename T>
Eigen::Quaternion<T> NodeLin<T>::GetCenter() const {
  return Project(nodeLin_.GetBox().GetCenter());
}

template<typename T>
Eigen::Matrix<T,4,1> NodeLin<T>::QuaternionToVec(const Eigen::Quaternion<T>& q) {
  return Eigen::Matrix<T,4,1>(q.w(),q.x(),q.y(),q.z());
}

template<typename T>
Tetrahedron4D<T> NodeLin<T>::TetraFromBox(const Box<T>& box, uint32_t i,
    uint32_t j, uint32_t k, uint32_t l) const {
  Eigen::Matrix<T,3,1> a;
  box.GetCorner(i,a);
  Eigen::Matrix<T,3,1> b;
  box.GetCorner(j,b);
  Eigen::Matrix<T,3,1> c;
  box.GetCorner(k,c);
  Eigen::Matrix<T,3,1> d;
  box.GetCorner(l,d);
  return Tetrahedron4D<T>(QuaternionToVec(Project(a)),
      QuaternionToVec(Project(b)),
      QuaternionToVec(Project(c)),
      QuaternionToVec(Project(d)));
}

template<typename T>
T NodeLin<T>::GetVolume_() const { 
  // subdivide box in Lin space into 4 tetrahedra and sum their volumes
  // https://www.ics.uci.edu/~eppstein/projects/tetra/
  // NodeS3 1: 0 4 5 7
  T V = TetraFromBox(nodeLin_.GetBox(), 0, 4, 5, 7).GetVolume();
  // NodeS3 2: 1 4 5 6
  V += TetraFromBox(nodeLin_.GetBox(), 1, 4, 5, 6).GetVolume();
  // NodeS3 3: 2 4 6 7
  V += TetraFromBox(nodeLin_.GetBox(), 2, 4, 6, 7).GetVolume();
  // NodeS3 4: 3 5 6 7
  V += TetraFromBox(nodeLin_.GetBox(), 3, 5, 6, 7).GetVolume();
  // NodeS3 5: 0 1 2 3
  V += TetraFromBox(nodeLin_.GetBox(), 0, 1, 2, 3).GetVolume();
  return V;
}

template<typename T>
std::string NodeLin<T>::ToString() const {
  std::stringstream ss;
  ss  << " in lin space: " << nodeLin_.ToString() 
//    << " V=" << GetVolume()
    << std::endl;
  for (const auto& q : qs_) 
    ss << "\t " << q.coeffs().transpose() << std::endl;
  return ss.str();
};

template<typename T>
std::string NodeLin<T>::Serialize() const {
  return nodeLin_.Serialize();
};

template class NodeLin<float>;
template class NodeLin<double>;
}
