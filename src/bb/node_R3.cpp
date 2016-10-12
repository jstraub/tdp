/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node_R3.h>

namespace tdp {

template <typename T>
NodeR3<T>::NodeR3(const Box<T>& box, std::vector<uint32_t> ids) 
  : BaseNode<T>(ids), box_(box) {
}

template <typename T>
NodeR3<T>::NodeR3(const NodeR3<T>& node) : BaseNode<T>(node.GetIds(),
    node.GetLB(), node.GetUB()), box_(node.GetBox()), 
  t_lb_(node.GetLbArgument()) {
}

template <typename T>
std::vector<NodeR3<T>> NodeR3<T>::Branch() const {
  std::vector<NodeR3<T>> nodes;
  nodes.reserve(8);
  std::vector<Box<T>> boxs = box_.Subdivide();
  for (uint32_t i=0; i < boxs.size(); ++i) {
    std::vector<uint32_t> ids(this->ids_);
    ids.push_back(i);
    nodes.push_back(NodeR3<T>(boxs[i], ids));
  }
  return nodes;
}

template <typename T>
std::list<NodeR3<T>> GenerateNotesThatTessellateR3(const Eigen::Matrix<T,3,1>&
    min, const Eigen::Matrix<T,3,1>& max, T max_side_len) {
  NodeR3<T> node0(Box<T>(min, max), std::vector<uint32_t>(1,0));
  std::vector<std::vector<NodeR3<T>>> node_tree;
  node_tree.push_back(std::vector<NodeR3<T>>(1,node0));
  for (uint32_t lvl = 0; lvl < 20; ++lvl) {
    node_tree.push_back(node_tree[lvl][0].Branch());
    for (uint32_t i = 1; i < node_tree[lvl].size(); ++i) {
      std::vector<NodeR3<T>> nodes_new = node_tree[lvl][i].Branch();
      for (auto& node: nodes_new) node_tree[lvl+1].push_back(node);
    }
    std::cout << "@" << lvl+1 << ": # " << node_tree[lvl+1].size() <<  std::endl;
    if (node_tree[lvl+1][0].GetBox().GetSideLengths().maxCoeff() < max_side_len)
      break;
  }
  uint32_t lvl = node_tree.size() -1;
  return std::list<NodeR3<T>>(node_tree[lvl].begin(), node_tree[lvl].end());
}

template <typename T>
std::string NodeR3<T>::ToString() const {
  std::stringstream out; 
  out << GetBox().GetCenter().transpose();
  out << " V=" << GetBox().GetVolume();
  return out.str();
};

template <typename T>
std::string NodeR3<T>::Serialize() const {
  std::stringstream out; 
  Eigen::Matrix<T,3,1> c;
  for (uint32_t i=0; i<8; ++i) {
    GetBox().GetCorner(i, c);
    out << c(0) << " " << c(1) << " " << c(2) << std::endl;
  }
  return out.str();
};

template class NodeR3<float>;
template class NodeR3<double>;

}
