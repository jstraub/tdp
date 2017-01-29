#include <tdp/testing/testing.h>
#include <tdp/nn_cuda/nn_cuda.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/utils/timer.hpp>

#include <tdp/nn/ann.h>

/*
TEST(speedTest, nn_cuda) {
  // randomly initialize a point cloud
  size_t N = 1000000;
  tdp::ManagedHostImage<tdp::Vector3fda> pc(N, 1);
  for (size_t i = 0; i < N; i++) {
    pc[i] = 10. * tdp::Vector3fda::Random();
  }

  // Randomly initialize a query cloud
  size_t M = 1000000;
  tdp::ManagedHostImage<tdp::Vector3fda> qc(M, 1);
  for (size_t i = 0; i < M; i++) {
    qc[i] = 10. * tdp::Vector3fda::Random();
  }

  tdp::Timer timer;
  tdp::NN_Cuda nn;
  timer.toctic("Start");

  nn.reinitialise(pc);
  timer.toctic("Reinitializing the NN_Cuda object");

  int k = 1;
  Eigen::VectorXi nnIds(k);
  Eigen::VectorXf dists(k);

  for (size_t i = 0; i < M; i++) {
    nn.search(qc[i], k, nnIds, dists);
  }

  timer.toctic("1000000 searches");
}
*/

TEST(correctness, nn_cuda) {
  size_t N = 100;
  tdp::ManagedHostImage<tdp::Vector3fda> pc(N, 1);
  for (size_t i = 0; i < N; i++) {
    pc[i] = 10. * tdp::Vector3fda::Random();
  }

  size_t M = 1;
  tdp::ManagedHostImage<tdp::Vector3fda> qc(M, 1);
  for (size_t i = 0; i < M; i++) {
    qc[i] = 10. * tdp::Vector3fda::Random();
  }

  int k = 10;
  Eigen::VectorXi nnIds(k);
  Eigen::VectorXf dists(k);
  tdp::NN_Cuda nn;

  nn.reinitialise(pc);
  for (size_t i = 0; i < M; i++) {
    nn.search(qc[i], k, nnIds, dists);
    for (size_t j = 0; j < k; j++) {
      std::cout << nnIds(j) << " " << dists(j) << std::endl;
    }
  }
}

/*
TEST(relativeCorrectness, nn_cuda) {
  // randomly initialize a point cloud
  size_t N = 1000000;
  tdp::ManagedHostImage<tdp::Vector3fda> pc(N, 1);
  for (size_t i = 0; i < N; i++) {
    pc[i] = 10. * tdp::Vector3fda::Random();
  }

  // Randomly initialize a query cloud
  size_t M = 1000;
  tdp::ManagedHostImage<tdp::Vector3fda> qc(M, 1);
  for (size_t i = 0; i < M; i++) {
    qc[i] = 10. * tdp::Vector3fda::Random();
  }

  int k = 1;
  Eigen::VectorXi nnIds(k);
  Eigen::VectorXf dists(k);
  tdp::NN_Cuda nn;

  Eigen::VectorXi nnIds_old(k);
  Eigen::VectorXf dists_old(k);
  tdp::ANN ann;

  nn.reinitialise(pc);
  ann.ComputeKDtree(pc);

  for (size_t i = 0; i < M; i++) {
    nn.search(qc[i], k, nnIds, dists);
    ann.Search(qc[i], k, 0., nnIds_old, dists_old);

    if (nnIds(0) != nnIds_old(0)) {
      std::cout << "ERROR on search " << i << std::endl;
      std::cout << "\t Found " << nnIds(0) << std::endl;
      std::cout << "\t Expected " << nnIds_old(0) << std::endl;
    }
  }
}
*/

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
