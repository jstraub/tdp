/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include "bbTrans/node_TpS3.h"

namespace bb {

NodeTpS3::NodeTpS3(const Box& box, std::vector<uint32_t> ids) 
  : NodeLin(box, ids)
{ 
  Linearize(box);
}

NodeTpS3::NodeTpS3(const NodeTpS3& node) 
  : NodeLin(node)
{ }

Eigen::Quaterniond NodeTpS3::Project(const Eigen::Vector3d& c) const {
  // In accordance with the other tessellation approaches
  static Eigen::Vector4d north(1.,0.,0.,0.);
  static S<double,4> s3(north);
  Eigen::Vector4d qvec = s3.Exp(s3.ToAmbient(c)).vector();
  return Eigen::Quaterniond(qvec(0),qvec(1),qvec(2),qvec(3));
}

std::vector<NodeTpS3> NodeTpS3::Branch() const {
  std::vector<NodeTpS3> nodes;
  nodes.reserve(8);
  std::vector<NodeR3> boxs = nodeLin_.Branch();
  for (uint32_t i=0; i < boxs.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeTpS3(boxs[i].GetBox(), ids));
  }
  return nodes;
}

std::list<NodeTpS3> TessellateTpS3() {
  // split into 4 cubes for level 4 so that at level 2 we have 256
  // cubes which is close to the 270 of the 600-cell tessellation.
  Eigen::Vector3d p_min(-0.5*M_PI,-0.5*M_PI,-0.5*M_PI);
  Eigen::Vector3d p_max( 0., 0., 0.5*M_PI);
  NodeTpS3 root00(Box(p_min, p_max),std::vector<uint32_t>(1,0));
  p_min << -0.5*M_PI,0.,-0.5*M_PI;
  p_max << 0., 0.5*M_PI, 0.5*M_PI;
  NodeTpS3 root01(Box(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0., -0.5*M_PI,-0.5*M_PI;
  p_max << 0.5*M_PI, 0., 0.5*M_PI;
  NodeTpS3 root10(Box(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0.,0.,-0.5*M_PI;
  p_max << 0.5*M_PI, 0.5*M_PI, 0.5*M_PI;
  NodeTpS3 root11(Box(p_min, p_max),std::vector<uint32_t>(1,1));
//  std::cout << root.ToString() << std::endl;
  std::vector<NodeTpS3> l1 = root00.Branch();
  std::vector<NodeTpS3> a = root01.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root10.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root11.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  std::list<NodeTpS3> nodes;
  for (auto& node1 : l1) {
    std::vector<NodeTpS3> l2 = node1.Branch();
//    for (auto& node2 : l2) {
//      std::vector<NodeTpS3> l3 = node2.Branch();
//      for (auto& node3 : l3) {
//        std::vector<NodeTpS3> l4 = node3.Branch();
        nodes.insert(nodes.end(), l2.begin(), l2.end());
//      }
//    }
  }
  return nodes;
}

}
