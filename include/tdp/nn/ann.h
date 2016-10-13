#pragma once
#include <ANN/ANN.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>

// ANN wrapper
namespace tdp {

class ANN {
 public:

  ANN() : pc_(nullptr), N_(0), kdTree_(nullptr) 
  {}
  ~ANN() {
    if (kdTree_) delete kdTree_;
    if (pc_) delete[] pc_;
  };

  void ComputeKDtree(Image<Vector3fda>& pc) {
    if (kdTree_) delete kdTree_;
    if (pc_) delete[] pc_;
    // build array of pointers to the data points because thats how ANN
    // wants the data
    N_ = pc.Area();
    pc_ = new ANNpoint[N_];
    for (size_t i=0; i<pc.Area(); ++i) pc_[i] = &pc[i](0);
    // build KD tree
    kdTree_ = new ANNkd_tree(pc_, N_, 3, 1, ANN_KD_SUGGEST);
  }

  void Search(Vector3fda& query, int k, float eps, 
       Eigen::VectorXi& nnIds, Eigen::VectorXf& dists) {
    assert(nnIds.size() == k);
    assert(dists.size() == k);
    kdTree_->annkSearch(&query(0), k, &nnIds(0), &dists(0), eps);
  }

 private:
  ANNpointArray pc_;
  int N_;
  ANNkd_tree* kdTree_;
};

}
