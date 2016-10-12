/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<class Node>
std::vector<uint32_t> CountBranchesInTree(const std::list<Node>& nodes) {
  uint32_t max_lvl = 0;
  for (auto& node : nodes) if (node.GetLevel() > max_lvl) 
    max_lvl = node.GetLevel();
  std::cout << "# nodes: " << nodes.size() 
    << " max depth in nodes: " << max_lvl << std::endl;
  std::vector<std::map<uint64_t, uint32_t>> counts(max_lvl+1);
  std::vector<std::map<uint64_t, const Node*>> nodesPerLvl(max_lvl+1);
  for (auto& node : nodes) {
    uint32_t node_lvl = node.GetLevel();
    for (uint32_t lvl = 0; lvl < node_lvl+1; ++lvl) {
      ++ counts[lvl][node.GetIdAtLevel(lvl)];
      if (lvl == node_lvl) 
        nodesPerLvl[lvl][node.GetIdAtLevel(lvl)] = &node;
      else 
        nodesPerLvl[lvl][node.GetIdAtLevel(lvl)] = NULL;
    }
  }
  std::vector<uint32_t> countsPerLevel(max_lvl,0);
  for (uint32_t lvl = 0; lvl < max_lvl; ++lvl) {
    std::cout << "@ lvl " << lvl << std::endl;
    for (auto& count : counts[lvl]) {
      std::cout << "  id: " << count.first << "\t#:" << count.second;
      if (nodesPerLvl[lvl][count.first])
        std::cout << "\tinfo: " << nodesPerLvl[lvl][count.first]->ToString()
        << std::endl;
      else
        std::cout << std::endl;
      ++ countsPerLevel[lvl];
    }
  }
  return countsPerLevel;
}

}
