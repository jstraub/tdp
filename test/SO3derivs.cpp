#include <tdp/testing/testing.h>
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/rotation.h>
#include <tdp/manifold/SO3.h>
#include <tdp/eigen/dense.h>

using namespace tdp;

TEST(SO3, deriv) {
  float eps = 1e-3;

  for (size_t i=0; i<100; ++i) {
    Eigen::Vector3f p_c = Eigen::Vector3f::Random();
    tdp::SO3f T_wc = tdp::SO3f::Random();
    for (size_t j=0; j<3; ++j) {
      Eigen::Matrix<float,3,1> delta = Eigen::Matrix<float,3,1>::Zero();
      delta(j) = eps;
      tdp::SO3f T_wcDelta = T_wc.Exp(delta);
      Eigen::Vector3f diffGt = T_wcDelta*p_c - T_wc*p_c ; 
      Eigen::Matrix<float,3,3> J = -T_wc.matrix()*tdp::SO3f::invVee(p_c);
      Eigen::Vector3f diffJ = J*delta;
      std::cout << j << ": " << (diffGt-diffJ).norm() << std::endl;
//        << ";\t" << diffGt.transpose() << " " << diffJ.transpose() << std::endl;
    }
  }
}

TEST(SO3, derivofInverse) {
  float eps = 1e-3;

  for (size_t i=0; i<100; ++i) {
    Eigen::Vector3f p_w = Eigen::Vector3f::Random();
    tdp::SO3f T_wc = tdp::SO3f::Random();
    for (size_t j=0; j<3; ++j) {
      Eigen::Matrix<float,3,1> delta = Eigen::Matrix<float,3,1>::Zero();
      delta(j) = eps;
      tdp::SO3f T_wcDelta = T_wc.Exp(delta);
      Eigen::Vector3f diffGt = T_wcDelta.Inverse()*p_w- T_wc.Inverse()*p_w; 
      Eigen::Matrix<float,3,3> J = tdp::SO3f::invVee(T_wc.Inverse()*p_w);
      Eigen::Vector3f diffJ = J*delta;
      std::cout << j << ": " << (diffGt-diffJ).norm() 
        << ";\t" << diffGt.transpose() << " " << diffJ.transpose() << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
