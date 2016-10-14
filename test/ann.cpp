#include <tdp/testing/testing.h>
#include <tdp/nn/ann.h>
#include <tdp/nn/ann.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/utils/timer.hpp>

TEST(smallSearch, ann) {
  
  for (size_t N=10000; N<1000000; N += 100000) {

    tdp::ANN ann;
    tdp::ManagedHostImage<tdp::Vector3fda> pcA(N,1);
    tdp::ManagedHostImage<tdp::Vector3fda> pcB(N,1);

    for (size_t i=0; i<pcA.w_; ++i) {
      pcA[i] = 10.*tdp::Vector3fda::Random();
      pcB[i] = 10.*tdp::Vector3fda::Random();
      if (i%100==0) {
        pcA[i](0) = NAN;
        pcA[i](1) = NAN;
        pcA[i](2) = NAN;
        pcB[i](0) = NAN;
        pcB[i](1) = NAN;
        pcB[i](2) = NAN;
      }
    }
    tdp::Timer t0;
    ann.ComputeKDtree(pcA);
    t0.toctic("building KD tree");

    std::cout << "running ANN with " << ann.N_ 
      << " of " <<  N << " points " << std::endl;

    int k = 10;
    Eigen::VectorXi nnIds(k);
    Eigen::VectorXf dists(k);
    for (size_t i=0; i<pcB.w_; ++i) {
      ann.Search(pcB[i], k, 0., nnIds, dists);
    }

    t0.toctic("search");
  }

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
