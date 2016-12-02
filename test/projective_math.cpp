#include <tdp/testing/testing.h>
#include <tdp/manifold/SE3.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/projective_math.h>

TEST(projective_math, TransformHomography) {
  tdp::SE3f T_rd(Eigen::Matrix4f::Identity());
  tdp::Camera<float> camR(Eigen::Vector4f(275,275,159.5,119.5)); 
  tdp::Camera<float> camD(Eigen::Vector4f(550,550,319.5,239.5)); 
  Eigen::Vector3f nd_r(0,0,-1);

  Eigen::Vector2f u_d(camD.params_(2),camD.params_(3));
  Eigen::Vector2f u_r = TransformHomography(u_d, T_rd, camR, camD, nd_r);

  std::cout << T_rd << std::endl;
  Eigen::Matrix3f H = (T_rd.rotation().matrix()-T_rd.translation()*nd_r.transpose());
  std::cout << H << std::endl;

  ASSERT_NEAR(u_r(0),camR.params_(2),EPS);
  ASSERT_NEAR(u_r(1),camR.params_(3),EPS);

  u_d << 0,0;
  u_r = TransformHomography(u_d, T_rd, camR, camD, nd_r);
  ASSERT_NEAR(u_r(0),0.,EPS);
  ASSERT_NEAR(u_r(1),0.,EPS);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

