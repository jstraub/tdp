#pragma once
#include <ANN/ANN.h>

// ANN wrapper
namespace tdp {

class ANN {
 public:

  ANN() : kdTree(nullptr) 
  {}
  ~ANN() {
    if (kdTree) delete kdTree;
  };

  void ComputeKDtree(const Image<Vector3fda>& pc) {
    if (kdTree) delete kdTree;
    kdTree = new ANNkd_tree( pc.ptr_, pc.w_, 3);					
  }

  void Search(const Vector3fda& query, int k, float eps, 
       Eigen::VectorXi& nnIdx, Eigen::VectorXf& dists) {
    assert(nnIdx.size() == k);
    assert(dists.size() == k);
    kdTree->annkSearch(&query(0), k, &nnIdx(0), &dists(0), eps);
  }

 private:
  ANNkd_tree* kdTree;
};

}
