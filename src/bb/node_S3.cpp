/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node_S3.h>

namespace tdp {

NodeS3::NodeS3(const Tetrahedron4D& tetrahedron,
    std::vector<uint32_t> ids) : BaseNode(ids),
  tetrahedron_(tetrahedron) {
}

NodeS3::NodeS3(const NodeS3& node) : BaseNode(node.GetIds(),
    node.GetLB(), node.GetUB()), tetrahedron_(node.GetTetrahedron()),
  q_lb_(node.GetLbArgument()) {
}

std::vector<NodeS3> NodeS3::Branch() const {
  std::vector<NodeS3> nodes;
  nodes.reserve(8);
  std::vector<Tetrahedron4D> tetrahedra = tetrahedron_.Subdivide();
  for (uint32_t i=0; i < tetrahedra.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeS3(tetrahedra[i], ids));
  }
  return nodes;
}

std::string NodeS3::ToString() const {
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

std::string NodeS3::Serialize() const {
  std::stringstream out; 
  Eigen::Vector4d v;
  for (uint32_t i=0; i<4; ++i) {
    v = GetTetrahedron().GetVertex(i);
    out << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << std::endl;
  }
  return out.str();
};

std::list<NodeS3> GenerateNotesThatTessellateS3() {
  std::vector<Tetrahedron4D> tetrahedra = TessellateS3();
  std::list<NodeS3> nodes; 
//  nodes.reserve(tetrahedra.size());
  for (uint32_t i=0; i<tetrahedra.size(); ++i) {
    nodes.push_back(NodeS3(tetrahedra[i], std::vector<uint32_t>(1,i)));
  }
  return nodes;
}


}
