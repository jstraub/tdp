/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include "bbTrans/node_AA.h"

namespace bb {

NodeAA::NodeAA(const Box& box, std::vector<uint32_t> ids) 
  : NodeLin(box,ids)
{
  Linearize(box);
}

NodeAA::NodeAA(const NodeAA& node) 
  : NodeLin(node)
{ }

Eigen::Quaterniond NodeAA::Project(const Eigen::Vector3d& c) const {
  Eigen::Quaterniond q;
  double theta = c.norm();
  if (theta > 1e-9) {
    q.w() = cos(theta*0.5);
    q.vec() = -c*sin(theta*0.5)/theta;
  } else {
    q.w() = 1.;
    q.vec().fill(0.);
  }
  return q;
}

std::vector<NodeAA> NodeAA::Branch() const {
  std::vector<NodeAA> nodes;
  nodes.reserve(8);
  std::vector<NodeR3> boxs = nodeLin_.Branch();
  for (uint32_t i=0; i < boxs.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeAA(boxs[i].GetBox(), ids));
  }
  return nodes;
}

std::list<NodeAA> TessellateAA() {
//  Eigen::Vector3d p_min(-M_PI,-M_PI,-M_PI);
//  Eigen::Vector3d p_max( M_PI, M_PI, M_PI);
  // split into 4 cubes for level 4 so that at level 2 we have 256
  // cubes which is close to the 270 of the 600-cell tessellation.
  Eigen::Vector3d p_min(-M_PI,-M_PI,-M_PI);
  Eigen::Vector3d p_max( 0., 0., M_PI);
  NodeAA root00(Box(p_min, p_max),std::vector<uint32_t>(1,0));
  p_min << -M_PI,0.,-M_PI;
  p_max << 0., M_PI, M_PI;
  NodeAA root01(Box(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0., -M_PI,-M_PI;
  p_max << M_PI, 0., M_PI;
  NodeAA root10(Box(p_min, p_max),std::vector<uint32_t>(1,1));
  p_min << 0.,0.,-M_PI;
  p_max << M_PI, M_PI, M_PI;
  NodeAA root11(Box(p_min, p_max),std::vector<uint32_t>(1,1));
//  std::cout << root.ToString() << std::endl;
  std::vector<NodeAA> l1 = root00.Branch();
  std::vector<NodeAA> a = root01.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root10.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  a = root11.Branch();
  l1.insert(l1.end(),a.begin(), a.end());
  std::list<NodeAA> nodes;
  for (auto& node1 : l1) {
    std::vector<NodeAA> l2 = node1.Branch();
//    for (auto& node2 : l2) {
//      std::vector<NodeAA> l3 = node2.Branch();
//      for (auto& node3 : l3) {
//        std::vector<NodeAA> l4 = node3.Branch();
        nodes.insert(nodes.end(), l2.begin(), l2.end());
//      }
//    }
  }
  return nodes;
}

}
