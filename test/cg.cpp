#include <tdp/testing/testing.h>
#include <tdp/cg/cg.h>
#include <tdp/utils/timer.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

TEST(CG, small) {
  size_t Nvar = 6*3;
  size_t Nobs = 100;
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(Nobs,Nvar); 
  Eigen::VectorXf b = Eigen::VectorXf::Random(Nobs); 

  Eigen::MatrixXf ATA = A.transpose()*A;
  Eigen::VectorXf ATb = A.transpose()*b;

  Eigen::VectorXf x = ATA.ldlt().solve(ATb);
  std::cout << x.transpose() << std::endl;

  Eigen::SparseMatrix<float> Asp = A.sparseView();
  Eigen::VectorXf xCg = Eigen::VectorXf::Zero(Nvar);
  tdp::CG::ComputeCpu(Asp, b, 100, 1e-6, xCg);
  std::cout << xCg.transpose() << std::endl;

  Eigen::VectorXf xPcg = Eigen::VectorXf::Zero(Nvar);
  Eigen::VectorXf Mdiag = ATA.diagonal();
  tdp::PCG::ComputeCpu(Asp, b, Mdiag, 100, 1e-6, xPcg);
  std::cout << xPcg.transpose() << std::endl;

}

TEST(CG, large) {
  size_t Nvar = 6*100;
  size_t Nobs = 1000000;
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(Nobs,Nvar); 
  Eigen::VectorXf b = Eigen::VectorXf::Random(Nobs); 

  tdp::Timer t0;
  Eigen::MatrixXf ATA = A.transpose()*A;
  Eigen::VectorXf ATb = A.transpose()*b;
  Eigen::VectorXf x = ATA.ldlt().solve(ATb);
  t0.toctic("ldlt solve");

  Eigen::SparseMatrix<float> Asp = A.sparseView();
  Eigen::VectorXf xCg = Eigen::VectorXf::Zero(Nvar);

  t0.tic();
  tdp::CG::ComputeCpu(Asp, b, 100, 1e-6, xCg);
  t0.toctic("cg");
  std::cout << "CG err:\t" << (x-xCg).norm() << std::endl;

  Eigen::VectorXf xPcg = Eigen::VectorXf::Zero(Nvar);
  Eigen::VectorXf Mdiag = ATA.diagonal();
  t0.tic();
  tdp::PCG::ComputeCpu(Asp, b, Mdiag, 100, 1e-6, xPcg);
  t0.toctic("pcg");
  std::cout << "PCG err:\t" << (x-xPcg).norm() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
