#include <tdp/testing/testing.h>

#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SO3mat.h>

using namespace tdp;
TEST(SO3, exp) {

  const float eps = 1e-5;
  for (size_t i=0; i<1000; ++i) {
    Eigen::Vector3f x = 1e-3*Eigen::Vector3f::Random();
    Eigen::Matrix3f R0 = SO3f::Exp_(x).matrix();
    Eigen::Matrix3f R1 = SO3mat<float>::Exp_(x).matrix();

    for (size_t i=0; i<9; ++i)
      ASSERT_NEAR(R0(i), R1(i), eps);

    Eigen::Vector3f x0 = SO3f::Log_(R0);
    Eigen::Vector3f x1 = SO3mat<float>::Log_(R1);

    for (size_t i=0; i<3; ++i)
      ASSERT_NEAR(x0(i), x1(i), eps);
    for (size_t i=0; i<3; ++i)
      ASSERT_NEAR(x0(i), x(i), eps);
    for (size_t i=0; i<3; ++i)
      ASSERT_NEAR(x(i), x1(i), eps);


  }
}

TEST(SO3, opt) {

  
  SO3d R;
  std::cout << R << std::endl;

  double theta = 15.*M_PI/180.;
  Eigen::Matrix3d Rmu_;
  Rmu_ << 1, 0, 0,
         0, cos(theta), sin(theta),
         0, -sin(theta), cos(theta);
  SO3d Rmu(Rmu_);
  
  std::cout << Rmu << std::endl;
//  std::cout << R+Rmu << std::endl;
  std::cout << R << std::endl;
//  std::cout << Rmu+R << std::endl;
//  std::cout << R-Rmu << std::endl;

//  Eigen::Vector3d w = R-Rmu;
//  std::cout << Rmu.Exp(w) << std::endl;

//  std::cout << Rmu-R << std::endl;
  
  double delta = 0.1;
  double f_prev = 1e99;
  double f = (Rmu.Inverse() * R).matrix().trace();
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<100; ++it) {
    Eigen::Matrix3d J = -0.5*((R.Inverse() * Rmu).matrix() - (Rmu.Inverse() * R).matrix()); 
    Eigen::Vector3d Jw = SO3d::vee(J);
//    R = R.Exp(-delta*Jw);
    R *= SO3d::Exp_(-delta*Jw);
//    std::cout << Jw << std::endl;
    f_prev = f;
    f = (Rmu.Inverse() * R).matrix().trace();
//    std::cout << "f=" << f << " df/f=" << (f_prev - f)/f 
//      << std::endl;
  }
  std::cout << Rmu << std::endl;
  std::cout << R << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

