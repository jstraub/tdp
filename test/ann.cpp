#include <tdp/testing/testing.h>
#include <tdp/nn/ann.h>
#include <tdp/nn/ann.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/utils/timer.hpp>

TEST(smallSearch, ann) {
  
  tdp::ANN ann;

  for (size_t N=10000; N<1000000; N += 100000) {
  tdp::ManagedHostImage<tdp::Vector3fda> pcA(N,1);
  tdp::ManagedHostImage<tdp::Vector3fda> pcB(N,1);

  for (size_t i=0; i<pcA.w_; ++i) {
    pcA[i] = 10.*tdp::Vector3fda::Random();
    pcB[i] = 10.*tdp::Vector3fda::Random();
  }
  
  std::cout << "running ANN with " << N << " points " << std::endl;
  tdp::Timer t0;
  ann.ComputeKDtree(pcA);
  t0.toctic("building KD tree");

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
