/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node_S3.h>

namespace tdp {

template <typename T>
NodeS3<T>::NodeS3(const Tetrahedron4D<T>& tetrahedron,
    std::vector<uint32_t> ids) : BaseNode<T>(ids),
  tetrahedron_(tetrahedron) {
}

template <typename T>
NodeS3<T>::NodeS3(const NodeS3<T>& node) : BaseNode<T>(node.GetIds(),
    node.GetLB(), node.GetUB()), tetrahedron_(node.GetTetrahedron()),
  q_lb_(node.GetLbArgument()) {
}

template <typename T>
std::vector<NodeS3<T>> NodeS3<T>::Branch() const {
  std::vector<NodeS3<T>> nodes;
  nodes.reserve(8);
  std::vector<Tetrahedron4D<T>> tetrahedra = tetrahedron_.Subdivide();
  for (uint32_t i=0; i < tetrahedra.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeS3<T>(tetrahedra[i], ids));
  }
  return nodes;
}

template <typename T>
std::string NodeS3<T>::ToString() const {
  std::stringstream out; 
  out << GetTetrahedron().GetCenter().transpose() << std::endl;

  for (uint32_t i=0; i < 4; ++i) 
    out << GetTetrahedron().GetVertex(i).transpose() << "|.| " <<
      GetTetrahedron().GetVertex(i).norm() << std::endl;
  out << "pairwise angles: ";
  for (uint32_t i=0; i < 4; ++i) 
    for (uint32_t j=0; j < 4; ++j) 
      if(i!=j)
        out << i << "," <<j<< ": "
          << GetTetrahedron().GetVertexQuaternion(i).angularDistance(
              GetTetrahedron().GetVertexQuaternion(j)) *180./M_PI<< " ";
  return out.str();
};

template <typename T>
std::string NodeS3<T>::Serialize() const {
  std::stringstream out; 
  Eigen::Matrix<T,4,1> v;
  for (uint32_t i=0; i<4; ++i) {
    v = GetTetrahedron().GetVertex(i);
    out << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << std::endl;
  }
  return out.str();
};

template class NodeS3<float>;
template class NodeS3<double>;

template <typename T>
std::list<NodeS3<T>> GenerateNotesThatTessellateS3() {
  std::vector<Tetrahedron4D<T>> tetrahedra = TessellateS3<T>();
  std::list<NodeS3<T>> nodes; 
//  nodes.reserve(tetrahedra.size());
  for (uint32_t i=0; i<tetrahedra.size(); ++i) {
    nodes.push_back(NodeS3<T>(tetrahedra[i], std::vector<uint32_t>(1,i)));
  }
  return nodes;
}

template std::list<NodeS3<float>> GenerateNotesThatTessellateS3();
template std::list<NodeS3<double>> GenerateNotesThatTessellateS3();

}
