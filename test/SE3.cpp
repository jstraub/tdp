#include <tdp/testing/testing.h>
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/rotation.h>
#include <tdp/manifold/SE3.h>

#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/pc.h>

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

TEST(SE3, inverse) {

  const float eps = 1e-6;
  for (size_t i=0; i<10000; ++i) {
    Eigen::Matrix<float,3,1> p0 = Eigen::Matrix<float,3,1>::Random();
    Eigen::Matrix<float,6,1> x0 = Eigen::Matrix<float,6,1>::Random();
    SE3f T = SE3f::Exp_(x0);
    Eigen::Matrix4f Tmat = T.matrix();
    
    Eigen::Matrix4f TmatInv = Tmat.inverse();
    Eigen::Matrix4f Tinv = T.Inverse().matrix();

    Eigen::Vector3f p1 = TmatInv.topLeftCorner(3,3)*p0 + TmatInv.topRightCorner(3,1);
    Eigen::Vector3f p2 = T.Inverse()*p0;

    ASSERT_NEAR(p2(0), p1(0), eps);
    ASSERT_NEAR(p2(1), p1(1), eps);
    ASSERT_NEAR(p2(2), p1(2), eps);

    for (size_t i=0; i<16; ++i)
      ASSERT_NEAR(TmatInv(i), Tinv(i), eps);

    SE3f Tse3Inv = T.Inverse();
    
    Eigen::Matrix4f Tinvinvmat = Tse3Inv.Inverse().matrix();

    for (size_t i=0; i<16; ++i)
      ASSERT_NEAR(Tmat(i), Tinvinvmat(i), eps);

  }
}

TEST(SE3, exp) {

  const float eps = 1e-5;

  Eigen::Matrix<float,6,1> x0;
  x0 << 0,0,ToRad(10.),0,0.1,0.1;
  SE3f T0 = SE3f::Exp_(x0);
  Eigen::Matrix<float,6,1> x1 = SE3f::Log_(T0);
  std::cout << x0.transpose() << std::endl;
  std::cout << x1.transpose() << std::endl;

  ASSERT_NEAR(x0(0), x1(0), eps);
  ASSERT_NEAR(x0(1), x1(1), eps);
  ASSERT_NEAR(x0(2), x1(2), eps);
  ASSERT_NEAR(x0(3), x1(3), eps);
  ASSERT_NEAR(x0(4), x1(4), eps);
  ASSERT_NEAR(x0(5), x1(5), eps);

  x0 << 0,0,0.,0,0.1,0.1;
  T0 = SE3f::Exp_(x0);
  x1 = SE3f::Log_(T0);
  std::cout << x0.transpose() << std::endl;
  std::cout << x1.transpose() << std::endl;

  ASSERT_NEAR(x0(0), x1(0), eps);
  ASSERT_NEAR(x0(1), x1(1), eps);
  ASSERT_NEAR(x0(2), x1(2), eps);
  ASSERT_NEAR(x0(3), x1(3), eps);
  ASSERT_NEAR(x0(4), x1(4), eps);
  ASSERT_NEAR(x0(5), x1(5), eps);

  for (size_t i=0; i<10000; ++i) {
    x0 = Eigen::Matrix<float,6,1>::Random();
    T0 = SE3f::Exp_(x0);
    x1 = SE3f::Log_(T0);

    ASSERT_NEAR(x0(0), x1(0), eps);
    ASSERT_NEAR(x0(1), x1(1), eps);
    ASSERT_NEAR(x0(2), x1(2), eps);
    ASSERT_NEAR(x0(3), x1(3), eps);
    ASSERT_NEAR(x0(4), x1(4), eps);
    ASSERT_NEAR(x0(5), x1(5), eps);

  }

}

#ifdef CUDA_FOUND
TEST(SE3, gpu) {

  const float eps = 1e-6;

  for (size_t it=0; it<100; ++it) {
    SE3f T(SO3f::R_rpy(Eigen::Vector3f::Random()), Eigen::Vector3f::Random());

    ManagedHostImage<Vector3fda> x(1000,1);
    ManagedHostImage<Vector3fda> xAfter(1000,1);
    ManagedDeviceImage<Vector3fda> cuX(1000,1);

    for (size_t i=0; i<1000; ++i) {
      x[i] = Vector3fda::Random();
      xAfter[i] = T*x[i];
    }
    cuX.CopyFrom(x, cudaMemcpyHostToDevice);
    TransformPc(T, cuX);
    x.CopyFrom(cuX, cudaMemcpyDeviceToHost);

    for (size_t i=0; i<1000; ++i) {
      ASSERT_NEAR(x[i](0), xAfter[i](0), eps);
      ASSERT_NEAR(x[i](1), xAfter[i](1), eps);
      ASSERT_NEAR(x[i](2), xAfter[i](2), eps);
    }
  }
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

