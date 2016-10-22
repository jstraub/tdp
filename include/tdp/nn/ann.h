#pragma once
#include <ANN/ANN.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>

// ANN wrapper
namespace tdp {

class ANN {
 public:

  ANN() : N_(0), pc_(nullptr), kdTree_(nullptr) 
  {}
  ~ANN() {
    if (kdTree_) delete kdTree_;
    if (pc_) delete[] pc_;
  };

  void ComputeKDtree(Image<Vector3fda>& pc, int stride=1) {
    if (kdTree_) delete kdTree_;
    if (pc_) delete[] pc_;
    idMap_.clear();
    // build array of pointers to the data points because thats how ANN
    // wants the data
    N_ = pc.Area();
    pc_ = new ANNpoint[N_];
    idMap_.reserve(N_);
    size_t j=0;
    for (size_t i=0; i<pc.Area(); i+=stride) {
      if (IsValidData(pc[i])) {
        pc_[j++] = &pc[i](0);
        idMap_.push_back(i);
      }
    }
    N_ = j;
//    std::cout << "building ann PC data structure " << N_ 
//      << " of " << pc.Area() << std::endl;
    // build KD tree
    kdTree_ = new ANNkd_tree(pc_, N_, 3, 1, ANN_KD_SUGGEST);
  }

  void Search(Vector3fda& query, int k, float eps, 
       Eigen::VectorXi& nnIds, Eigen::VectorXf& dists) {
    assert(nnIds.size() == k);
    assert(dists.size() == k);
    kdTree_->annkSearch(&query(0), k, &nnIds(0), &dists(0), eps);
    for (int i=0; i<k; ++i) { 
      nnIds(i) = nnIds(i) >= 0 ? idMap_[nnIds(i)] : nnIds(i);
    }
  }

  int N_;
 private:
  std::vector<int> idMap_;
  ANNpointArray pc_;
  ANNkd_tree* kdTree_;
};

}
