#include <tdp/testing/testing.h>
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/rotation.h>
#include <tdp/manifold/SE3.h>

using namespace tdp;


TEST(SE3, setup) {
  
  SE3d T;
  std::cout << T << std::endl;

  double theta = 15.*M_PI/180.;
  Eigen::Matrix4d Tmu_;
  Tmu_ << 1, 0, 0, 0,
         0, cos(theta), sin(theta), 0,
         0, -sin(theta), cos(theta), 0,
           0,0,0,1;
  SE3d Tmu(Tmu_);
  
  std::cout << Tmu << std::endl;
  std::cout << T*Tmu << std::endl;
  std::cout << T << std::endl;
  std::cout << Tmu*T << std::endl;

  std::cout << Tmu.Inverse()*Tmu << std::endl;

  SE3f T1(SO3f::Rx(ToRad(10.)), Eigen::Vector3f(0,0,1));
  std::cout << T1 << std::endl;
  std::cout << T1*Eigen::Vector3f(1,0,0) << std::endl;
  std::cout << T1*Eigen::Vector3f(0,1,0) << std::endl;

  SE3f T2(SO3f::Rx(ToRad(10.)));

//  std::cout << T-Tmu << std::endl;
//  Eigen::Matrix<double,6,1> w = T-Tmu;
//  std::cout << Tmu.Exp(w) << std::endl;
//  std::cout << Tmu-T << std::endl;

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

