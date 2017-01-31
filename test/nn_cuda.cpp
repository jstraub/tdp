#include <tdp/testing/testing.h>
#include <tdp/nn_cuda/nn_cuda.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/utils/timer.hpp>

#include <tdp/nn/ann.h>

void resizeAndRandomlyFill(
     tdp::ManagedHostImage<tdp::Vector3fda>* image,
     size_t newSize
) {
  image->Reinitialise(newSize, 1);
  for (size_t i = 0; i < newSize; i++) {
    (*image)[i] = 10. * tdp::Vector3fda::Random();
  }
}

TEST(nn_cuda, speedTest) {
  // randomly initialize a point cloud
  size_t N = 10000000;
  tdp::ManagedHostImage<tdp::Vector3fda> pc;
  resizeAndRandomlyFill(&pc, N);

  // Randomly initialize a query cloud
  size_t M = 100;
  tdp::ManagedHostImage<tdp::Vector3fda> qc;
  resizeAndRandomlyFill(&qc, M);

  int k = 1;
  Eigen::VectorXi nnIds(k);
  Eigen::VectorXf dists(k);

  tdp::Timer timer;
  timer.toctic("Start");
  tdp::NN_Cuda nn;
  nn.reinitialise(pc);
  timer.toctic("Reinitialized");
  for (size_t i = 0; i < M; i++) {
    nn.search(qc[i], k, nnIds, dists);
  }
  timer.toctic("NN: 10,000,000 points, 100 queries");
  float nn_cudaTime = timer.toc();

  Eigen::VectorXi nnIds_old(k);
  Eigen::VectorXf dists_old(k);

  tdp::Timer timer2;
  timer2.toctic("Start");
  tdp::ANN ann;
  ann.ComputeKDtree(pc);
  timer2.toctic("Reinitialized");
  for (size_t i = 0; i < M; i++) {
    nn.search(qc[i], k, nnIds, dists);
  }
  timer2.toctic("ANN: 10,000,000 points, 100 queries");
  float annTime = timer.toc();

  EXPECT_TRUE(nn_cudaTime < annTime);
}

TEST(nn_cuda, correctness) {
  size_t N = 10;
  tdp::ManagedHostImage<tdp::Vector3fda> pc;
  resizeAndRandomlyFill(&pc, N);

  size_t M = 1;
  tdp::ManagedHostImage<tdp::Vector3fda> qc;
  resizeAndRandomlyFill(&qc, M);

  int k = 10;
  Eigen::VectorXi nnIds(k);
  Eigen::VectorXf dists(k);
  tdp::NN_Cuda nn;

  nn.reinitialise(pc);
  nn.search(qc[0], k, nnIds, dists);

  for (size_t i = 0; i < k; i++) {
    tdp::Vector3fda diff = pc[nnIds(i)] - qc[0];
    float dist = diff.dot(diff);
    float vdiff = dist - dists(i);
    EXPECT_TRUE(std::abs(vdiff) < 1e-4);

    if (i < k - 1) {
      EXPECT_TRUE(dists(i) <= dists(i + 1));
    }
  }
}

// Tests for proper ordering wrt original ANN code
TEST(nn_cuda, relativeCorrectness) {
  size_t N = 1000;
  tdp::ManagedHostImage<tdp::Vector3fda> pc;
  resizeAndRandomlyFill(&pc, N);

  size_t M = 1000;
  tdp::ManagedHostImage<tdp::Vector3fda> qc;
  resizeAndRandomlyFill(&qc, M);

  int k = 10;
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

    for (size_t j = 0; j < k; j++) {
      EXPECT_TRUE(nnIds(j) == nnIds_old(j));
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
