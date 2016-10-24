#include <tdp/testing/testing.h> 
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SO3mat.h>

using namespace tdp;

TEST(SO3, log) {
  const float eps = 1e-4;

  for (size_t i=0; i<1000; ++i) {
    Eigen::Vector2f rand = Eigen::Vector2f::Random();
    SO3f Rx0 = SO3f::Rx(ToRad(rand(0)));
    SO3matf Rx0_ = SO3matf::Rx(ToRad(rand(0)));

    ASSERT_TRUE(IsAppox(Rx0.matrix(), Rx0_.matrix(), eps));
    ASSERT_NEAR(fabs(rand(0)), ToDeg(SO3f::Log_(Rx0).norm()), eps);
    ASSERT_TRUE(rand(0)<0.? SO3f::Log_(Rx0)(0) < 0. : SO3f::Log_(Rx0)(0) >= 0. );

  }

  for (size_t i=0; i<1000; ++i) {
    SO3f R0 = SO3f::Random();
    Eigen::Vector3f x0 = SO3f::Log_(R0);
    Eigen::Vector3f x1 = SO3mat<float>::Log_(R0.matrix());

    if (!IsAppox(x0,x1,eps)) {
      std::cout << R0.matrix() << std::endl;
      Eigen::Vector3f axis;
      float angle;
      R0.ToAxisAngle(axis, angle);
      std::cout << axis.transpose() << " angle " << angle 
        << " " << ToDeg(angle) << std::endl;
      std::cout << ToDeg(x0.norm()) << " " << ToDeg(x1.norm()) << std::endl;
      std::cout << R0 << std::endl;
    }
    EXPECT_TRUE(IsAppox(x0,x1,eps));
//    ASSERT_NEAR(x0.norm(),x1.norm(),eps);
  }
}

TEST(SO3, exp) {

  const float eps = 1e-5;
  for (size_t i=0; i<1000; ++i) {
    Eigen::Vector3f x = 1e-3*Eigen::Vector3f::Random();
    Eigen::Matrix3f R0 = SO3f::Exp_(x).matrix();
    Eigen::Matrix3f R1 = SO3mat<float>::Exp_(x).matrix();

    ASSERT_TRUE(IsAppox(R0,R1,eps));

    Eigen::Vector3f x0 = SO3f::Log_(R0);
    Eigen::Vector3f x1 = SO3mat<float>::Log_(R1);
    
    if (!IsAppox(x0,x1,eps)) {
      std::cout << R0 << std::endl << R1 << std::endl;
      std::cout << "so3:  " << x.transpose() << std::endl;
      std::cout << "Quat: " << SO3f::Exp_(x) << std::endl;
    }

    ASSERT_TRUE(IsAppox(x0,x1,eps));
    ASSERT_TRUE(IsAppox(x,x1,eps));
    ASSERT_TRUE(IsAppox(x0,x,eps));

  }
}

TEST(SO3, Rz) {
  const float eps = 1e-4;
  Eigen::Matrix3f R0t = Eigen::Matrix3f::Identity();
  SO3f R0 = SO3f::Rz(0.);
  ASSERT_TRUE(IsAppox(R0.matrix(), R0t, eps));

  SO3f R0_1;
  for (size_t i=0; i<36; ++i)
    R0_1 = SO3f::Rz(ToRad(10.)) * R0_1;
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));

  R0_1 = SO3f();
  for (size_t i=0; i<360; ++i)
    R0_1 = SO3f::Rz(ToRad(1.)) * R0_1;
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));

  R0_1 = SO3f();
  for (size_t i=0; i<3600; ++i)
    R0_1 = SO3f::Rz(ToRad(0.1)) * R0_1;
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));
}

TEST(SO3, Ry) {
  const float eps = 1e-4;
  Eigen::Matrix3f R0t = Eigen::Matrix3f::Identity();
  SO3f R0 = SO3f::Ry(0.);
  ASSERT_TRUE(IsAppox(R0.matrix(), R0t, eps));

  SO3f R0_1;
  for (size_t i=0; i<36; ++i)
    R0_1 *= SO3f::Ry(ToRad(10.));
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));

  R0_1 = SO3f();
  for (size_t i=0; i<360; ++i)
    R0_1 *= SO3f::Ry(ToRad(1.));
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));

  R0_1 = SO3f();
  for (size_t i=0; i<3600; ++i)
    R0_1 *= SO3f::Ry(ToRad(0.1));
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));
}

TEST(SO3, Rx) {
  const float eps = 1e-4;
  Eigen::Matrix3f R0t = Eigen::Matrix3f::Identity();
  SO3f R0 = SO3f::Rx(0.);
  ASSERT_TRUE(IsAppox(R0.matrix(), R0t, eps));

  ASSERT_TRUE(IsAppox(SO3f::Rx(ToRad(10.)).matrix(), 
        SO3mat<float>::Rx(ToRad(10.)).matrix(),eps));
  SO3f R0_1;
  for (size_t i=0; i<36; ++i) 
    R0_1 *= SO3f::Rx(ToRad(10));
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));

  R0_1 = SO3f();
  for (size_t i=0; i<360; ++i)
    R0_1 *= SO3f::Rx(ToRad(1.));
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));

  R0_1 = SO3f();
  for (size_t i=0; i<3600; ++i)
    R0_1 *= SO3f::Rx(ToRad(0.1));
  ASSERT_TRUE(IsAppox(R0_1.matrix(), R0t, eps));
}

TEST(SO3, composition) {

  const float eps = 1e-5;
  for (size_t i=0; i<1000; ++i) {
    SO3f Rw0 = SO3f::Random();
    SO3f Rw1 = SO3f::Random();
    Eigen::Matrix3f Rw0mat = Rw0.matrix();
    Eigen::Matrix3f Rw1mat = Rw1.matrix();

    SO3f R01 = Rw0.Inverse() * Rw1;
    Eigen::Matrix3f R01mat = Rw0mat.transpose()*Rw1mat;
    ASSERT_TRUE(R01mat.isApprox(R01.matrix(),eps));

    SO3f Rw0w1 = Rw0 * Rw1;
    Eigen::Matrix3f Rw0w1mat = Rw0mat*Rw1mat;
    ASSERT_TRUE(Rw0w1mat.isApprox(Rw0w1.matrix(),eps));
  }

  SO3f Rw0;
  Eigen::Matrix3f Rw0mat = Eigen::Matrix3f::Identity();
  for (size_t i=0; i<1000; ++i) {
    Eigen::Vector3f x0 = 1e-3*Eigen::Vector3f::Random();
    Rw0 = Rw0 * SO3f::Exp_(x0);
    Rw0mat = Rw0mat * SO3mat<float>::Exp_(x0).matrix();
    ASSERT_TRUE(Rw0mat.isApprox(Rw0.matrix(),eps));
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

