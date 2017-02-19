#include <tdp/testing/testing.h>
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/rotation.h>
#include <tdp/manifold/SE3.h>
#include <tdp/eigen/dense.h>

using namespace tdp;

TEST(SE3, deriv) {
  float eps = 1e-5;

  for (size_t i=0; i<100; ++i) {
    Eigen::Vector3f p_c = Eigen::Vector3f::Random();
    tdp::SE3f T_wc = tdp::SE3f::Random();
    for (size_t j=0; j<6; ++j) {
      Eigen::Matrix<float,6,1> delta = Eigen::Matrix<float,6,1>::Zero();
      delta(j) = eps;
      tdp::SE3f T_wcDelta = T_wc.Exp(delta);
      Eigen::Vector3f diffGt = T_wcDelta*p_c - T_wc*p_c ; 
      Eigen::Matrix<float,3,6> J;
      J << -T_wc.rotation().matrix()* tdp::SO3f::invVee(p_c), Eigen::Matrix3f::Identity();
      Eigen::Vector3f diffJ = J*delta;
      std::cout << j << ": " << (diffGt-diffJ).norm() << std::endl;
//        << ";\t" << diffGt.transpose() << " " << diffJ.transpose() << std::endl;
    }
  }
}

TEST(SE3, derivofInverse) {
  float eps = 1e-3;

  for (size_t i=0; i<100; ++i) {
    Eigen::Vector3f p_w = Eigen::Vector3f::Random();
    tdp::SE3f T_wc = tdp::SE3f::Random();
    for (size_t j=0; j<6; ++j) {
      Eigen::Matrix<float,6,1> delta = Eigen::Matrix<float,6,1>::Zero();
      delta(j) = eps;
      tdp::SE3f T_wcDelta = T_wc.Exp(delta);
      Eigen::Vector3f diffGt = T_wcDelta.Inverse()*p_w- T_wc.Inverse()*p_w; 
      Eigen::Matrix<float,3,6> J;
      J << tdp::SO3f::invVee(T_wc.rotation().Inverse()*(p_w-T_wc.translation())), - T_wc.rotation().Inverse().matrix();
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
