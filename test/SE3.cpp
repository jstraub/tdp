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

//  SE3f T1(SO3f::Rx(ToRad(10.)), Eigen::Vector3f(0,0,1));
//  std::cout << T1 << std::endl;
//  std::cout << T1*Eigen::Vector3f(1,0,0) << std::endl;
//  std::cout << T1*Eigen::Vector3f(0,1,0) << std::endl;
//
//  SE3f T2(SO3f::Rx(ToRad(10.)));

//  std::cout << T-Tmu << std::endl;
//  Eigen::Matrix<double,6,1> w = T-Tmu;
//  std::cout << Tmu.Exp(w) << std::endl;
//  std::cout << Tmu-T << std::endl;

}

TEST(SE3, transform) {

  const float eps = 1e-5;
  for (size_t i=0; i<10000; ++i) {
    Eigen::Matrix<float,3,1> p0 = Eigen::Matrix<float,3,1>::Random();
    SE3f T = SE3f::Random();
    Eigen::Matrix4f Tmat = T.matrix();
    Eigen::Matrix4f TmatInv = Tmat.inverse();

    Eigen::Vector3f p1 = TmatInv.topLeftCorner(3,3)*p0 + TmatInv.topRightCorner(3,1);
    Eigen::Vector3f p2 = T.Inverse()*p0;
    ASSERT_TRUE(IsAppox(p1, p2, eps));

    p1 = Tmat.topLeftCorner(3,3)*p0 + Tmat.topRightCorner(3,1);
    p2 = T*p0;
    ASSERT_TRUE(IsAppox(p1, p2, eps));

  }
}

TEST(SE3, inverse) {

  const float eps = 1e-5;
  for (size_t i=0; i<10000; ++i) {
    Eigen::Matrix<float,6,1> x0 = Eigen::Matrix<float,6,1>::Random();
    SE3f T = SE3f::Exp_(x0);
    Eigen::Matrix4f Tmat = T.matrix();
    
    Eigen::Matrix4f TmatInv = Tmat.inverse();
    Eigen::Matrix4f Tinv = T.Inverse().matrix();
    ASSERT_TRUE(IsAppox(TmatInv, Tinv, eps));

    SE3f Tse3Inv = T.Inverse();
    Eigen::Matrix4f Tinvinvmat = Tse3Inv.Inverse().matrix();
    ASSERT_TRUE(IsAppox(Tinvinvmat, Tmat, eps));

  }
}

TEST(SE3, expLog) {

  const float eps = 1e-5;
  for (size_t i=0; i<10000; ++i) {
    Eigen::Matrix<float,6,1> x0 = Eigen::Matrix<float,6,1>::Random();
    SE3f T = SE3f::Exp_(x0);
    Eigen::Matrix<float,6,1> x1 = SE3f::Log_(T);
    ASSERT_TRUE(IsAppox(x0, x1, eps));

//    Eigen::Matrix4f Tmat = T.matrix();
//    
//    Eigen::Matrix4f TmatInv = Tmat.inverse();
//    Eigen::Matrix4f Tinv = T.Inverse().matrix();
//    ASSERT_TRUE(IsAppox(TmatInv, Tinv, eps));
//
//    SE3f Tse3Inv = T.Inverse();
//    Eigen::Matrix4f Tinvinvmat = Tse3Inv.Inverse().matrix();
//    ASSERT_TRUE(IsAppox(Tinvinvmat, Tmat, eps));

  }
}

TEST(SE3, composition) {

  const float eps = 1e-5;
  for (size_t i=0; i<1000; ++i) {
    SE3f Tw0 = SE3f::Random();
    SE3f Tw1 = SE3f::Random();
    Eigen::Matrix4f Tw0mat = Tw0.matrix();
    Eigen::Matrix4f Tw1mat = Tw1.matrix();

    SE3f T01 = Tw0.Inverse() * Tw1;
    Eigen::Matrix4f T01mat = Tw0mat.inverse()*Tw1mat;

    Eigen::Matrix4f T01_mat = T01.matrix();
    ASSERT_TRUE(IsAppox(T01mat,T01_mat,eps));

    SE3f Tw0w1 = Tw0 * Tw1;
    Eigen::Matrix4f Tw0w1mat = Tw0mat*Tw1mat;
    ASSERT_TRUE(Tw0w1mat.isApprox(Tw0w1.matrix(),eps));


  }

  SE3f Tw0;
  Eigen::Matrix4f Tw0mat = Eigen::Matrix4f::Identity();
  SE3f Tw1;
  Eigen::Matrix4f Tw1mat = Eigen::Matrix4f::Identity();
  for (size_t i=0; i<1000; ++i) {
    Eigen::Matrix<float,6,1> x0 = 1e-3*Eigen::Matrix<float,6,1>::Random();
    Tw0 = Tw0 * SE3f::Exp_(x0);
    Tw0mat = Tw0mat * SE3f::Exp_(x0).matrix();
    ASSERT_TRUE(Tw0mat.isApprox(Tw0.matrix(),eps));

    Tw1 = SE3f::Exp_(x0) * Tw1;
    Tw1mat = SE3f::Exp_(x0).matrix() * Tw1mat;
    ASSERT_TRUE(Tw1mat.isApprox(Tw1.matrix(),eps));
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

  const float eps = 1e-5;

  for (size_t it=0; it<100; ++it) {
    SE3f T = SE3f::Random();
//    SO3f R (Eigen::Quaternion<float,Eigen::DontAlign>(T.rotation().vector()));
    SO3matf R (T.rotation());
//    std::cout << T << std::endl;
//    std::cout << T.rotation() << std::endl;
//    std::cout << R << std::endl;

    ManagedHostImage<Vector3fda> x(1000,1);
    ManagedHostImage<Vector3fda> xBefore(1000,1);
    ManagedHostImage<Vector3fda> xAfter(1000,1);
    ManagedHostImage<Vector3fda> xAfterInv(1000,1);
    ManagedHostImage<Vector3fda> xAfterInvRot(1000,1);
    ManagedHostImage<Vector3fda> xAfterRot(1000,1);
    xAfter.Fill(Vector3fda::Zero());
    xAfterInv.Fill(Vector3fda::Zero());
    xAfterInvRot.Fill(Vector3fda::Zero());
    xAfterRot.Fill(Vector3fda::Zero());

    ManagedDeviceImage<Vector3fda> cuX(1000,1);
    ManagedDeviceImage<Vector3fda> cuXinv(1000,1);
    ManagedDeviceImage<Vector3fda> cuXinvRot(1000,1);
    ManagedDeviceImage<Vector3fda> cuXrot(1000,1);

    cudaMemset(cuX.ptr_, 0, cuX.SizeBytes());
    cudaMemset(cuXinv.ptr_, 0, cuXinv.SizeBytes());
    cudaMemset(cuXinvRot.ptr_, 0, cuXinvRot.SizeBytes());
    cudaMemset(cuXrot.ptr_, 0, cuXrot.SizeBytes());

    for (size_t i=0; i<1000; ++i) {
      x[i] = Vector3fda::Random();
      xBefore[i] = x[i];
      xAfter[i] = T*x[i];
      xAfterInv[i] = T.Inverse()*x[i];
      xAfterRot[i] = T.rotation()*x[i];
      xAfterInvRot[i] = T.rotation().Inverse()*x[i];
    }
    cuX.CopyFrom(x, cudaMemcpyHostToDevice);
    cuXinv.CopyFrom(x, cudaMemcpyHostToDevice);
    cuXinvRot.CopyFrom(x, cudaMemcpyHostToDevice);
    cuXrot.CopyFrom(x, cudaMemcpyHostToDevice);

    TransformPc(T, cuX);
    InverseTransformPc(T, cuXinv);
    InverseTransformPc(R, cuXinvRot);
    TransformPc(R, cuXrot);

    x.CopyFrom(cuX, cudaMemcpyDeviceToHost);
    for (size_t i=0; i<1000; ++i) {
      ASSERT_TRUE(IsAppox(x[i],xAfter[i], eps));
    }
    x.CopyFrom(cuXrot, cudaMemcpyDeviceToHost);
    for (size_t i=0; i<1000; ++i) {
      ASSERT_TRUE(IsAppox(x[i],xAfterRot[i], eps));
    }
    x.CopyFrom(cuXinvRot, cudaMemcpyDeviceToHost);
    for (size_t i=0; i<1000; ++i) {
      ASSERT_TRUE(IsAppox(x[i],xAfterInvRot[i], eps));
    }
    x.CopyFrom(cuXinv, cudaMemcpyDeviceToHost);
    for (size_t i=0; i<1000; ++i) {
      if(!IsAppox(x[i],xAfterInv[i], eps)) {
        std::cout << xBefore[i].transpose() << std::endl;    
      }
      ASSERT_TRUE(IsAppox(x[i],xAfterInv[i], eps));
    }
  }
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

