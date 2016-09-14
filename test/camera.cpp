#include <tdp/testing/testing.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>

TEST(setup, camera) {
  tdp::Cameraf::Parameters pf(4);
  pf << 550, 550, 319.5, 239.5;
  tdp::Cameraf cf(pf);

  tdp::Camerad::Parameters pd(4);
  pd << 550, 550, 319.5, 239.5;
  tdp::Camerad cd(pd);
}

TEST(project, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);

  Eigen::Vector3f x1(0,0,1);
  Eigen::Vector3f x2(0,0,2);
  Eigen::Vector2f u1 = c.Project(x1);
  Eigen::Vector2f u2 = c.Project(x2);
  ASSERT_NEAR(u1(0),u2(0),EPS);
  ASSERT_NEAR(u1(1),u2(1),EPS);
}

TEST(unproject, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);

  Eigen::Vector3f x1 = c.Unproject(10,10,100);
  Eigen::Vector3f x2 = c.Unproject(10,10,100);
  ASSERT_NEAR(x1(0),x2(0),EPS);
  ASSERT_NEAR(x1(1),x2(1),EPS);
  ASSERT_NEAR(x1(2),x2(2),EPS);
}

TEST(projectUnprojectThereAndBack, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);

  Eigen::Vector3f x1(0,0,1);
  Eigen::Vector2f u1 = c.Project(x1);
  Eigen::Vector3f x1_ = c.Unproject(u1(0),u1(1),x1(2));
  ASSERT_NEAR(x1(0),x1_(0),EPS);
  ASSERT_NEAR(x1(1),x1_(1),EPS);
  ASSERT_NEAR(x1(2),x1_(2),EPS);
}

TEST(projectUnprojectThereAndBack, poly3) {
  const float eps = 1e-7;
  tdp::CameraPoly3f::Parameters p(7);
  p << 550, 550, 319.5, 239.5, 0.1, 0.1, 0.1;
  tdp::CameraPoly3f c(p);

  Eigen::Vector3f x1(0,0.1,1);
  Eigen::Vector2f u1 = c.Project(x1);
  //std::cout << u1.transpose() << std::endl;
  Eigen::Vector3f x1_ = c.Unproject(u1(0),u1(1),x1(2));
  //std::cout << x1_.transpose() << std::endl;
  ASSERT_NEAR(x1(0),x1_(0),eps);
  ASSERT_NEAR(x1(1),x1_(1),eps);
  ASSERT_NEAR(x1(2),x1_(2),eps);

  c.params_(4) = 0.;
  c.params_(5) = 0.;
  c.params_(6) = 0.;

  u1 = c.Project(x1);
  x1_ = c.Unproject(u1(0),u1(1),x1(2));
  ASSERT_NEAR(x1(0),x1_(0),eps);
  ASSERT_NEAR(x1(1),x1_(1),eps);
  ASSERT_NEAR(x1(2),x1_(2),eps);
}

TEST(json, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);
  pangolin::json::value cam = c.ToJson();
  p << 0, 0, 0, 0;
  tdp::Cameraf c2(p);
  c2.FromJson(cam);
  
  ASSERT_NEAR(c.params_(0), c2.params_(0), EPS);
  ASSERT_NEAR(c.params_(1), c2.params_(1), EPS);
  ASSERT_NEAR(c.params_(2), c2.params_(2), EPS);
  ASSERT_NEAR(c.params_(3), c2.params_(3), EPS);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
