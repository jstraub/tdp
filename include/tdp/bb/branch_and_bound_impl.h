/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template <class Node>
BranchAndBound<Node>::BranchAndBound(Bound<Node>& lower_bound,
    Bound<Node>& upper_bound)
  : lower_bound_(lower_bound), upper_bound_(upper_bound)
{}

template <class Node>
uint32_t BranchAndBound<Node>::BoundAndPrune(std::list<Node>& nodes, float& lb,
    float& ub, float eps) {
  lb = -1e40; ub = -1e40; 
  for (auto& node : nodes) {
    // Because of numerics...
    if (node.GetUB() < node.GetLB()) {
      std::cout << " ub < lb: " << node.GetUB() << " < " <<
        node.GetLB() << " - " << node.GetUB() - node.GetLB()
        << " lvl " << node.GetLevel()
        << std::endl;
      node.SetUB(node.GetLB()+eps); 
    }
    lb = std::max(lb, node.GetLB());
    ub = std::max(ub, node.GetUB());
  }
  // Prune
  nodes.remove_if(IsPrunableNode<Node>(lb));
  return std::distance(nodes.begin(), nodes.end());
}

template <class Node>
void BranchAndBound<Node>::WriteStats(std::ofstream& out,
    std::list<Node>& nodes, float lb, float ub, float dt,
    typename
    std::list<Node>::iterator& node_star) {
    float V = 0.; // Volume that the nodes explain
    for (auto& node : nodes) V += node.GetVolume(); 
    uint32_t n_nodes = std::distance(nodes.begin(), nodes.end());
    out << lb << " " << ub << " " << n_nodes << " " << V 
      << " " << dt
      << " " << node_star->GetLB()
      << " " << node_star->GetUB()
      << " " << node_star->GetLevel() << std::endl;
};

template <class Node>
void BranchAndBound<Node>::WriteNodes(std::ofstream& out, std::list<Node>& nodes, float lb, float ub) {
    float V = 0.; // Volume that the nodes explain
    for (auto& node : nodes) V += node.GetVolume(); 
    uint32_t n_nodes = std::distance(nodes.begin(), nodes.end());
    out << lb << " " << ub << " " << n_nodes << " " << V << std::endl;
    for (auto& node : nodes) {
      out << node.GetLB() << " " << node.GetUB() << " " << node.GetLevel()
        << " " << node.GetVolume() << std::endl;
      out << node.Serialize();
    }
};

template <class Node>
Node BranchAndBound<Node>::Compute(std::list<Node>& nodes, float eps,
    uint32_t max_lvl, uint32_t max_it) {

  // Prepare output
  bool write_stats = true;
  std::ofstream out, outNodes;
  if (write_stats) {
    std::stringstream ss;
    ss << "./bb_iteration_stats_" << nodes.begin()->GetSpace() << ".csv";
    out.open(ss.str());
    std::stringstream ssNodes;
    ssNodes << "./bb_nodes_per_iteration_" << nodes.begin()->GetSpace() << ".csv";
    outNodes.open(ssNodes.str());
  }
  Timer t0;
  // Get initial upper and lower bounds
  float lb = -1.e6;
  float ub = -1.e6;
  for (auto& node : nodes) {
    lower_bound_.EvaluateAndSet(node);
    upper_bound_.EvaluateAndSet(node);
    // Because of numerics in S3 case...
    if (node.GetUB() < node.GetLB()) {
      node.SetUB(node.GetLB()+eps); 
    }
  }
  typename std::list<Node>::iterator node_star = FindBestNode(nodes, eps);
  if (write_stats) WriteStats(out, nodes, lb, ub, t0.toc(), node_star);
  if (write_stats) WriteNodes(outNodes, nodes, lb, ub);
  uint32_t n_nodes = BoundAndPrune(nodes, lb, ub, eps);
  uint32_t it = 0;
  do  {
    t0.tic();
    // Find node with the biggest upper bound (the most promising node)
    // to explore further.
    auto node_i = FindBestNodeToExplore(nodes, eps);
    // Find the current best estimate as the maximizer of the lower
    // bounds
    node_star = FindBestNode(nodes, eps);
    if (node_star->GetLevel() >= max_lvl) break;
    // Find the node with the biggest lower bound (the most
    // conservative node to return).
    if (it%(max_it/10) == 0)
      std::cout << "@" << it << " # " << n_nodes 
        << ": exploring " << node_i->GetLB() << " < " << node_i->GetUB() 
        << " lvl " << node_i->GetLevel()
        << ": best " << node_star->GetLB() << " < " << node_star->GetUB() 
        << " lvl " << node_star->GetLevel()
        << "\t global "
        << lb << " < " << ub << " " << " |.| " << (ub - lb)/lb
        << std::endl;
      // Branch and check resulting nodes.
      std::vector<Node> new_nodes = node_i->Branch();
      for (auto& node : new_nodes) {
        upper_bound_.EvaluateAndSet(node);
        lower_bound_.EvaluateAndSet(node);
        // Because of numerics in S3 case...
        if (node.GetUB() < node.GetLB()) {
          if (node.GetLevel() < 8) 
            std::cout << " ub < lb: " << node.GetUB() << " < " <<
              node.GetLB() << " - " << node.GetUB() - node.GetLB() 
              << " lvl " << node.GetLevel() << "@" << it << " # " <<
              n_nodes << ": cur " << node_i->GetLB() << " < " <<
              node_i->GetUB() << " lvl " << node_i->GetLevel() 
              << "\t global " << lb << " < " << ub << " " << " |.| " <<
              (ub - lb)/lb << std::endl;
          node.SetUB(node.GetLB()+10*eps); 
        }
        if (node.GetUB() > lb) {
          // Remember this node since we cannot prune it.
          nodes.push_back(node);
          ++n_nodes;
        }
      }
    // Pop the examined node since we are branching here
    nodes.erase(node_i); 
    n_nodes = BoundAndPrune(nodes, lb, ub, eps);
    if (write_stats) WriteStats(out, nodes, lb, ub, t0.toc(), node_star);
    if (write_stats) WriteNodes(outNodes, nodes, lb, ub);
  } while (it++ < max_it && (ub - lb)/lb > eps && n_nodes >= 1);

  node_star = FindBestNode(nodes, eps);

  std::cout << "@" << it << " # " << n_nodes << ": global " 
    << lb << " < " << ub << " " << " |.| " << fabs(ub - lb)/lb << "\t selected "
    << node_star->GetLB() << " < " << node_star->GetUB() << " lvl "
    << node_star->GetLevel() << " " //<< node_star.GetIds()[0] << "\t" 
    << std::endl;

  if (write_stats) out.close();
  if (write_stats) outNodes.close();
  return *node_star;
}

template<class Node>
typename std::list<Node>::iterator BranchAndBound<Node>::FindBestNodeToExplore(std::list<Node>& nodes, float eps) {
  auto node_i = std::max_element(nodes.begin(), nodes.end(),
      LessThanNodeUB<Node>());
  uint32_t Neq = 0;
  auto node_istar = node_i;
  for (auto it=nodes.begin(); it!=nodes.end(); it++) 
    if (node_i->GetUB() - it->GetUB() < eps && it->GetLB() > node_istar->GetLB()) {
      ++ Neq;
      node_istar = it;
    }
  if (Neq > 1) {
    std::cout << " equal to explore node updates: " << Neq << std::endl;
  }
  return node_istar;
}

template<class Node>
typename std::list<Node>::iterator BranchAndBound<Node>::FindBestNode(std::list<Node>& nodes, float eps) {
  auto node_star = std::max_element(nodes.begin(), nodes.end(),
      LessThanNodeLB<Node>());
  uint32_t Neq = 0;
  auto node_starstar = node_star;
  for (auto it=nodes.begin(); it!=nodes.end(); it++) 
    if (node_star->GetLB() - it->GetLB() < eps && it->GetUB() < node_starstar->GetUB()) {
      node_starstar = it;
      ++ Neq;
    }
  if (Neq > 1) {
    std::cout << " equal best nodes: " << Neq << std::endl;
  }
  return node_starstar;
}

}
