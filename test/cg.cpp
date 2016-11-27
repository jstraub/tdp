#include <tdp/testing/testing.h>
#include <tdp/cg/cg.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

TEST(CG, small) {
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(100,12); 
  Eigen::VectorXf b = Eigen::VectorXf::Random(100); 

  Eigen::MatrixXf ATA = A.transpose()*A;
  Eigen::VectorXf ATb = A.transpose()*b;

  Eigen::VectorXf x = ATA.ldlt().solve(ATb);
  std::cout << x.transpose() << std::endl;

  Eigen::SparseMatrix<float> Asp = A.sparseView();
  Eigen::VectorXf xCg = Eigen::VectorXf::Zero(12);
  tdp::CG::ComputeCpu(Asp, b, 100, xCg);
  std::cout << xCg.transpose() << std::endl;

  Eigen::VectorXf xPcg = Eigen::VectorXf::Zero(12);
  Eigen::VectorXf Mdiag = ATA.diagonal();
  tdp::PCG::ComputeCpu(Asp, b, Mdiag, 100, xPcg);
  std::cout << xPcg.transpose() << std::endl;

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
